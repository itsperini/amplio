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
	"errors"
	"path/filepath"
	"slices"
	"testing"

	"amplio/internal/db/sqlite"
	"amplio/internal/embed"
)

// pinnedFixture: a shared library (unpinned) and a small domain dir (pinned).
func pinnedFixture(t *testing.T) (lib, mine string, cache Cache) {
	t.Helper()
	root := t.TempDir()
	lib, mine = filepath.Join(root, "lib"), filepath.Join(root, "mine")
	writeSkill(t, lib, "spanner", "query spanner sql databases", "b")
	writeSkill(t, lib, "gmail", "read and send email messages", "b")
	writeSkill(t, lib, "bazel", "build run and test code with bazel", "b")
	writeSkill(t, mine, "tune-model", "tune hyperparameters of our model", "b")
	writeSkill(t, mine, "deploy-model", "deploy our model to serving", "b")
	store, err := sqlite.Open(":memory:")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })
	return lib, mine, NewDBCache(store)
}

func entryNames(es []Entry) []string {
	out := make([]string, len(es))
	for i, e := range es {
		out[i] = e.Name
	}
	return out
}

func hitNames(hs []Hit) []string {
	out := make([]string, len(hs))
	for i, h := range hs {
		out[i] = h.Entry.Name
	}
	return out
}

// Every pinned skill is listed regardless of relevance, the k relevant slots go
// to unpinned skills only, and nothing appears twice.
func TestIndex_InitialPinnedPlusRelevant(t *testing.T) {
	lib, mine, cache := pinnedFixture(t)
	ctx := context.Background()
	ix := NewIndex([]Source{{Name: "lib", Path: lib}, {Name: "mine", Path: mine, Pinned: true}}, embed.Mock{}, cache)
	if err := ix.Build(ctx); err != nil {
		t.Fatal(err)
	}
	if ix.PinnedCount() != 2 {
		t.Fatalf("PinnedCount = %d, want 2", ix.PinnedCount())
	}

	pinned, relevant, err := ix.Initial(ctx, "query spanner sql", 1)
	if err != nil {
		t.Fatal(err)
	}
	if got := entryNames(pinned); !slices.Equal(slices.Sorted(slices.Values(got)), []string{"deploy-model", "tune-model"}) {
		t.Errorf("pinned = %v, want both domain skills", got)
	}
	if got := hitNames(relevant); !slices.Equal(got, []string{"spanner"}) {
		t.Errorf("relevant = %v, want [spanner]", got)
	}

	// A k larger than the pool: every UNPINNED skill, still no pinned ones.
	_, relevant, _ = ix.Initial(ctx, "query spanner sql", 10)
	if got := hitNames(relevant); len(got) != 3 || slices.Contains(got, "tune-model") || slices.Contains(got, "deploy-model") {
		t.Errorf("relevant with large k = %v, want the 3 library skills only", got)
	}

	// Pinned skills are ordered by relevance when there is a query.
	pinned, _, _ = ix.Initial(ctx, "deploy our model to serving", 1)
	if got := entryNames(pinned); got[0] != "deploy-model" {
		t.Errorf("pinned order = %v, want the relevant one (deploy-model) first", got)
	}

	// k = 0 ([skills].initial = 0): pinned only.
	pinned, relevant, _ = ix.Initial(ctx, "query spanner sql", 0)
	if len(pinned) != 2 || len(relevant) != 0 {
		t.Errorf("k=0: pinned=%v relevant=%v, want 2 pinned and no relevant", entryNames(pinned), hitNames(relevant))
	}
}

// No query (a chat session): pinned skills by name, no relevance ranking.
func TestIndex_InitialNoQuery(t *testing.T) {
	lib, mine, cache := pinnedFixture(t)
	ctx := context.Background()
	emb := &countingEmbedder{inner: embed.Mock{}}
	ix := NewIndex([]Source{{Name: "lib", Path: lib}, {Name: "mine", Path: mine, Pinned: true}}, emb, cache)
	if err := ix.Build(ctx); err != nil {
		t.Fatal(err)
	}
	before := emb.calls
	pinned, relevant, err := ix.Initial(ctx, "", 5)
	if err != nil {
		t.Fatal(err)
	}
	if got := entryNames(pinned); !slices.Equal(got, []string{"deploy-model", "tune-model"}) {
		t.Errorf("pinned = %v, want by name", got)
	}
	if relevant != nil {
		t.Errorf("relevant = %v, want none without a query", hitNames(relevant))
	}
	if emb.calls != before {
		t.Errorf("embedded %d texts with no query, want 0", emb.calls-before)
	}
}

// A cache-hydrated index attributes pinned skills the same as a fresh scan.
func TestIndex_PinnedAfterCacheHydrate(t *testing.T) {
	lib, mine, cache := pinnedFixture(t)
	ctx := context.Background()
	sources := []Source{{Name: "lib", Path: lib}, {Name: "mine", Path: mine, Pinned: true}}
	if err := NewIndex(sources, embed.Mock{}, cache).Build(ctx); err != nil {
		t.Fatal(err)
	}
	ix := NewIndex(sources, embed.Mock{}, cache)
	if n := ix.LoadCached(ctx); n != 5 {
		t.Fatalf("hydrated %d, want 5", n)
	}
	if ix.PinnedCount() != 2 {
		t.Errorf("PinnedCount after hydrate = %d, want 2", ix.PinnedCount())
	}
}

// An overridden name is pinned iff the WINNING (later) definition comes from a
// pinned source.
func TestIndex_PinnedFollowsOverride(t *testing.T) {
	lib, mine, cache := pinnedFixture(t)
	writeSkill(t, mine, "spanner", "our own spanner guide", "b") // same name as lib's
	ctx := context.Background()

	// mine (pinned) overrides lib: its spanner wins and is pinned.
	ix := NewIndex([]Source{{Name: "lib", Path: lib}, {Name: "mine", Path: mine, Pinned: true}}, embed.Mock{}, cache)
	if err := ix.Build(ctx); err != nil {
		t.Fatal(err)
	}
	if pinned, _, _ := ix.Initial(ctx, "", 0); !slices.Contains(entryNames(pinned), "spanner") {
		t.Errorf("pinned = %v, want spanner (the winning copy is from the pinned source)", entryNames(pinned))
	}

	// lib (unpinned) layered after mine: lib's spanner wins and is not pinned.
	ix = NewIndex([]Source{{Name: "mine", Path: mine, Pinned: true}, {Name: "lib", Path: lib}}, embed.Mock{}, cache)
	if err := ix.Build(ctx); err != nil {
		t.Fatal(err)
	}
	if pinned, _, _ := ix.Initial(ctx, "", 0); slices.Contains(entryNames(pinned), "spanner") {
		t.Errorf("pinned = %v, want no spanner (the winning copy is from the unpinned source)", entryNames(pinned))
	}
}

// Attribution is by DIRECTORY prefix: /x/skills must not claim /x/skills-extra.
func TestSourceOf_LongestDirectoryPrefix(t *testing.T) {
	sources := []Source{{Path: "/x/skills"}, {Path: "/x/skills-extra"}, {Path: "/x/skills/nested/"}}
	cases := map[string]int{
		"/x/skills/a/SKILL.md":        0,
		"/x/skills-extra/a/SKILL.md":  1,
		"/x/skills/nested/a/SKILL.md": 2,
		"/elsewhere/a/SKILL.md":       -1,
	}
	for path, want := range cases {
		if got := sourceOf(sources, path); got != want {
			t.Errorf("sourceOf(%q) = %d, want %d", path, got, want)
		}
	}
}

// failingEmbedder embeds fine until armed, then errors (a query-time outage).
type failingEmbedder struct {
	embed.Mock
	fail bool
}

func (f *failingEmbedder) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	if f.fail {
		return nil, errors.New("embedder down")
	}
	return f.Mock.Embed(ctx, texts)
}

// A ranking failure still yields the pinned skills (by name), plus the error.
func TestIndex_InitialEmbedFailureKeepsPinned(t *testing.T) {
	lib, mine, cache := pinnedFixture(t)
	ctx := context.Background()
	emb := &failingEmbedder{}
	ix := NewIndex([]Source{{Name: "lib", Path: lib}, {Name: "mine", Path: mine, Pinned: true}}, emb, cache)
	if err := ix.Build(ctx); err != nil {
		t.Fatal(err)
	}
	emb.fail = true
	pinned, relevant, err := ix.Initial(ctx, "query spanner sql", 5)
	if err == nil {
		t.Error("want the embedding error surfaced")
	}
	if got := entryNames(pinned); !slices.Equal(got, []string{"deploy-model", "tune-model"}) || relevant != nil {
		t.Errorf("pinned=%v relevant=%v, want both pinned by name and no relevant", got, hitNames(relevant))
	}
}
