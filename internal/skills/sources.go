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
	"log/slog"
	"path/filepath"
	"strings"
)

// Source is a directory tree of skills to scan, with an optional per-source
// blocklist of skill names to exclude.
type Source struct {
	Name    string
	Path    string
	Blocked []string
	// Pinned marks every skill from this source to be listed at session start
	// regardless of relevance (see Index.Initial).
	Pinned bool
}

// sourceOf returns the index of the source an entry belongs to — the source
// whose Path is the longest directory prefix of the entry's SKILL.md path — or
// -1 if none matches. Path-based rather than recorded at scan time so the same
// rule attributes entries hydrated from the cache (which stores the path, not
// the source) and entries from a fresh scan. For an overridden name the entry
// is the winning (later) definition, so it resolves to the winning source.
func sourceOf(sources []Source, entryPath string) int {
	p := filepath.Clean(entryPath)
	best, bestLen := -1, -1
	for i, src := range sources {
		root := filepath.Clean(src.Path)
		if (p == root || strings.HasPrefix(p, root+string(filepath.Separator))) && len(root) > bestLen {
			best, bestLen = i, len(root)
		}
	}
	return best
}

// pinnedNames returns the names of the entries that belong to a pinned source.
func pinnedNames(sources []Source, entries map[string]Entry) map[string]bool {
	pinned := make(map[string]bool)
	for name, e := range entries {
		if i := sourceOf(sources, e.Path); i >= 0 && sources[i].Pinned {
			pinned[name] = true
		}
	}
	return pinned
}

// scanSources scans every source in order and merges into a flat entry list.
// Same-named skills resolve last-wins (a later source overrides an earlier one),
// so callers can layer an override dir after a base dir.
func scanSources(sources []Source) []Entry {
	merged := make(map[string]Entry)
	order := make([]string, 0) // first-seen order, stable across overrides
	for _, src := range sources {
		blocked := make(map[string]bool, len(src.Blocked))
		for _, b := range src.Blocked {
			blocked[b] = true
		}
		for _, e := range scanSkills(src.Path) {
			if blocked[e.Name] {
				slog.Info("skipping blocked skill", "name", e.Name, "source", src.Name)
				continue
			}
			if _, exists := merged[e.Name]; exists {
				slog.Info("skill overrides earlier definition", "name", e.Name, "source", src.Name)
			} else {
				order = append(order, e.Name)
			}
			merged[e.Name] = e
		}
	}
	out := make([]Entry, 0, len(order))
	for _, n := range order {
		out = append(out, merged[n])
	}
	return out
}
