// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package skills

import (
	"context"
	"log/slog"
	"sort"
	"sync"
	"time"

	"amplio/internal/config"
	"amplio/internal/embed"
	"amplio/internal/vec"
)

// Hit is one search result.
type Hit struct {
	Entry Entry
	Score float64 // cosine similarity in [-1, 1]
}

// Index is an in-memory cosine-similarity index over skill descriptions. Build
// once at startup (disk-cached embeddings); query repeatedly. Search is brute
// force over a few hundred rows — fast enough without a vector index.
type Index struct {
	sources  []Source
	embedder embed.Embedder
	cache    Cache

	mu      sync.RWMutex
	entries map[string]Entry
	names   []string        // row order
	matrix  [][]float32     // L2-normalized vectors, row per name
	pinned  map[string]bool // names of entries from a pinned source
	built   bool

	// initial is how many relevance-ranked skills the session-start seed lists
	// (see Initial). An instance-wide recall policy carried on the index — like
	// lessons.Index's recall switch — because the index is the one per-instance
	// recall object every agent already receives. Set once at startup.
	initial int
}

// NewIndex constructs an index; call Build to populate (does I/O + embeds).
func NewIndex(sources []Source, embedder embed.Embedder, cache Cache) *Index {
	return &Index{sources: sources, embedder: embedder, cache: cache, initial: config.DefaultSkillsInitial}
}

// SetInitialLimit sets how many relevance-ranked skills the session-start seed
// lists, in addition to pinned ones ([skills].initial). Call once at startup.
func (ix *Index) SetInitialLimit(n int) { ix.initial = max(n, 0) }

// InitialLimit is how many relevance-ranked skills the session-start seed lists.
func (ix *Index) InitialLimit() int { return ix.initial }

func embedText(e Entry) string { return e.Name + ": " + e.Description }

// LoadCached hydrates the index purely from the persisted cache — no file scan,
// no embedding API. Fast: just a single DB read + matrix construction. After
// this returns successfully with cached entries, IsBuilt() reports true and
// Search/Load serve from the cached snapshot.
//
// Intended as Stage 1 of a two-stage startup: call this synchronously so the
// agent has recall immediately, then call Build in a goroutine to reconcile
// the in-memory state against the current on-disk corpus. The atomic swap at
// Build's end replaces this snapshot with the up-to-date one.
//
// Returns the number of entries hydrated (0 means the cache was empty — the
// caller should treat this as cold-start and Build synchronously). A failure
// to load the cache is logged-and-ignored; the index stays empty and Build
// will recover from disk.
func (ix *Index) LoadCached(ctx context.Context) int {
	cached, err := ix.cache.Load(ctx, ix.embedder.ModelID())
	if err != nil {
		slog.Warn("skill cache hydrate failed; starting empty", "error", err)
		return 0
	}
	if len(cached) == 0 {
		return 0
	}
	// Sort by name for stable ordering (the file-scan path uses a sort, too —
	// keeping both deterministic means Search results don't churn between
	// stages 1 and 2 just because of insertion order).
	names := make([]string, 0, len(cached))
	for n := range cached {
		names = append(names, n)
	}
	sort.Strings(names)
	entryMap := make(map[string]Entry, len(cached))
	matrix := make([][]float32, 0, len(cached))
	for _, n := range names {
		c := cached[n]
		entryMap[n] = Entry{
			Name:        n,
			Description: c.Description,
			Body:        c.Body,
			Path:        c.Path,
			ContentHash: c.Hash,
		}
		matrix = append(matrix, vec.Normalize(c.Vector))
	}
	pinned := pinnedNames(ix.sources, entryMap)
	ix.mu.Lock()
	ix.entries, ix.names, ix.matrix, ix.pinned, ix.built = entryMap, names, matrix, pinned, true
	ix.mu.Unlock()
	slog.Info("skill index hydrated from cache (background reconcile to follow)", "skills", len(names), "pinned", len(pinned))
	return len(names)
}

// Build scans the sources, embeds new/changed skills (reusing cached vectors for
// unchanged ones), persists the cache, and swaps in the in-memory state. It
// retries a transient embedding failure (skills matter) up to maxAttempts before
// returning the error; the caller decides whether to fail or degrade to no recall.
func (ix *Index) Build(ctx context.Context) error {
	const maxAttempts = 3
	var err error
	for attempt := 1; attempt <= maxAttempts; attempt++ {
		if err = ix.buildOnce(ctx); err == nil {
			return nil
		}
		slog.Warn("skill index build attempt failed", "attempt", attempt, "max", maxAttempts, "error", err)
		if attempt == maxAttempts {
			break
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(time.Duration(attempt) * 2 * time.Second):
		}
	}
	return err
}

func (ix *Index) buildOnce(ctx context.Context) error {
	entries := scanSources(ix.sources)
	model := ix.embedder.ModelID()

	cached, err := ix.cache.Load(ctx, model)
	if err != nil {
		slog.Warn("skill cache load failed; re-embedding all", "error", err)
		cached = nil
	}

	var toEmbedNames, toEmbedTexts []string
	reuse := make(map[string][]float32)
	for _, e := range entries {
		if c, ok := cached[e.Name]; ok && c.Hash == e.ContentHash {
			reuse[e.Name] = c.Vector
			continue
		}
		toEmbedNames = append(toEmbedNames, e.Name)
		toEmbedTexts = append(toEmbedTexts, embedText(e))
	}

	fresh := make(map[string][]float32)
	if len(toEmbedTexts) > 0 {
		vecs, err := ix.embedder.Embed(ctx, toEmbedTexts)
		if err != nil {
			return err
		}
		for i, n := range toEmbedNames {
			fresh[n] = vecs[i]
		}
	}

	entryMap := make(map[string]Entry, len(entries))
	names := make([]string, 0, len(entries))
	matrix := make([][]float32, 0, len(entries))
	save := make(map[string]CacheEntry, len(entries))
	for _, e := range entries {
		embVec, ok := reuse[e.Name]
		if !ok {
			embVec, ok = fresh[e.Name]
		}
		if !ok {
			continue // unreachable: every entry was reused or embedded
		}
		entryMap[e.Name] = e
		names = append(names, e.Name)
		matrix = append(matrix, vec.Normalize(embVec))
		save[e.Name] = CacheEntry{
			Hash:        e.ContentHash,
			Vector:      embVec,
			Description: e.Description,
			Path:        e.Path,
			Body:        e.Body,
		}
	}

	if err := ix.cache.Save(ctx, model, save); err != nil {
		slog.Warn("skill cache save failed", "error", err)
	}

	pinned := pinnedNames(ix.sources, entryMap)
	ix.mu.Lock()
	ix.entries, ix.names, ix.matrix, ix.pinned, ix.built = entryMap, names, matrix, pinned, true
	ix.mu.Unlock()
	slog.Info("skill index built", "skills", len(names), "pinned", len(pinned), "embedded", len(fresh), "reused", len(reuse))
	return nil
}

// IsBuilt reports whether Build has completed.
func (ix *Index) IsBuilt() bool {
	ix.mu.RLock()
	defer ix.mu.RUnlock()
	return ix.built
}

// Size is the number of indexed skills.
func (ix *Index) Size() int {
	ix.mu.RLock()
	defer ix.mu.RUnlock()
	return len(ix.names)
}

// PinnedCount is the number of indexed skills that come from a pinned source.
func (ix *Index) PinnedCount() int {
	ix.mu.RLock()
	defer ix.mu.RUnlock()
	return len(ix.pinned)
}

// Load returns the parsed entry for name (in-memory; no file read).
func (ix *Index) Load(name string) (Entry, bool) {
	ix.mu.RLock()
	defer ix.mu.RUnlock()
	e, ok := ix.entries[name]
	return e, ok
}

// Search returns up to k skills most similar to query, by cosine descending.
// A degenerate query vector (norm below vec.MinNorm — e.g. the embedder
// collapsed a whitespace-only string) returns nil rather than every-row-
// tied-at-zero junk from an arbitrary unstable sort.
func (ix *Index) Search(ctx context.Context, query string, k int) ([]Hit, error) {
	ix.mu.RLock()
	names, matrix, entries := ix.names, ix.matrix, ix.entries
	ix.mu.RUnlock()

	if k <= 0 || len(names) == 0 || query == "" {
		return nil, nil
	}
	scores, err := ix.score(ctx, query, matrix)
	if err != nil || scores == nil {
		return nil, err
	}
	return topK(names, entries, scores, k, nil), nil
}

// Initial selects the skills listed at session start: EVERY skill from a pinned
// source, plus the k most relevant of the rest (pinned ones never take one of
// the k slots, and never appear twice). Pinned skills are ordered by relevance
// to query, or by name when there is no query (a chat session) or ranking
// failed. The query is embedded once for both.
//
// On an embedding error it still returns the pinned skills (by name) along
// with the error: they don't depend on the ranking, and the caller can log the
// failure and list them anyway.
func (ix *Index) Initial(ctx context.Context, query string, k int) (pinned []Entry, relevant []Hit, err error) {
	ix.mu.RLock()
	names, matrix, entries, pinnedSet := ix.names, ix.matrix, ix.entries, ix.pinned
	ix.mu.RUnlock()

	var scores []float64
	if query != "" && len(names) > 0 && (k > 0 || len(pinnedSet) > 0) {
		scores, err = ix.score(ctx, query, matrix)
	}

	if len(pinnedSet) > 0 {
		if scores != nil {
			for _, h := range topK(names, entries, scores, len(names), func(n string) bool { return pinnedSet[n] }) {
				pinned = append(pinned, h.Entry)
			}
		} else {
			byName := make([]string, 0, len(pinnedSet))
			for n := range pinnedSet {
				byName = append(byName, n)
			}
			sort.Strings(byName)
			for _, n := range byName {
				pinned = append(pinned, entries[n])
			}
		}
	}
	if scores != nil && k > 0 {
		relevant = topK(names, entries, scores, k, func(n string) bool { return !pinnedSet[n] })
	}
	return pinned, relevant, err
}

// score embeds query and returns its cosine similarity to every row of matrix,
// or nil for a degenerate query vector (norm below vec.MinNorm — e.g. the
// embedder collapsed a whitespace-only string): every row tied at zero would
// be arbitrary-order junk.
func (ix *Index) score(ctx context.Context, query string, matrix [][]float32) ([]float64, error) {
	vecs, err := ix.embedder.Embed(ctx, []string{query})
	if err != nil {
		return nil, err
	}
	qv := vec.NormalizeOrNil(vecs[0])
	if qv == nil {
		return nil, nil
	}
	scores := make([]float64, len(matrix))
	for i, row := range matrix {
		scores[i] = vec.Dot(row, qv)
	}
	return scores, nil
}

// topK returns up to k entries by descending score, considering only names for
// which keep reports true (nil keep = all).
func topK(names []string, entries map[string]Entry, scores []float64, k int, keep func(string) bool) []Hit {
	idx := make([]int, 0, len(names))
	for i, n := range names {
		if keep == nil || keep(n) {
			idx = append(idx, i)
		}
	}
	sort.SliceStable(idx, func(a, b int) bool { return scores[idx[a]] > scores[idx[b]] })
	if k > len(idx) {
		k = len(idx)
	}
	out := make([]Hit, 0, k)
	for _, i := range idx[:k] {
		out = append(out, Hit{Entry: entries[names[i]], Score: scores[i]})
	}
	return out
}
