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

package main

import (
	"slices"
	"testing"

	"amplio/internal/config"
)

// [skills].pinned marks matching dirs (compared after cleaning, so a trailing
// slash still matches) and reports the entries that name no configured dir.
func TestSkillSources_Pinned(t *testing.T) {
	sources, unknown := skillSources(
		[]string{"/team/skills", "/home/me/mine"},
		config.SkillsConfig{Pinned: []string{"/home/me/mine/", "/home/me/typo"}, Blocked: []string{"noisy"}},
	)
	if len(sources) != 2 || sources[0].Pinned || !sources[1].Pinned {
		t.Fatalf("sources = %+v, want only /home/me/mine pinned", sources)
	}
	if !slices.Equal(sources[0].Blocked, []string{"noisy"}) || !slices.Equal(sources[1].Blocked, []string{"noisy"}) {
		t.Errorf("the blocklist must still apply to every source: %+v", sources)
	}
	if !slices.Equal(unknown, []string{"/home/me/typo"}) {
		t.Errorf("unknown pinned = %v, want [/home/me/typo]", unknown)
	}
}
