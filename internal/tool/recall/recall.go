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

// Package recall exposes the skill + lesson knowledge corpus to agents:
// recall_search finds relevant guides, recall_load fetches one's full body.
// Hits carry typed handles (skill:<name> / lesson:<id>) so the kind is explicit
// in the trajectory and recall_load can dispatch on it. Either corpus may be
// absent; the tools surface whichever indexes are built.
package recall

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"strings"

	"amplio/internal/db"
	"amplio/internal/lessons"
	"amplio/internal/skills"
	"amplio/internal/tool"
	"amplio/internal/util"
)

const (
	skillPrefix  = "skill:"
	lessonPrefix = "lesson:"

	// descPreviewMax bounds each search hit's description so a wide result set
	// stays readable in the tool output.
	descPreviewMax = 300

	// initialLessonHits is how many lessons the session-start seed shows —
	// intentionally terser than the recall_search default (10) since it's
	// unsolicited bootstrap context, not an explicit query. (The skill count is
	// the operator's [skills].initial, carried on the skill index.)
	initialLessonHits = 5
)

func skillReady(ix *skills.Index) bool { return ix != nil && ix.IsBuilt() }

// lessonReady gates every agent-facing use of the lesson corpus: the search
// tool, the lesson: branch of recall_load, and the run-start seed. An instance
// started with lesson search off (see lessons.Index.DisableRecall) reads as
// "no lesson corpus" here while mining and scoring carry on elsewhere.
func lessonReady(ix *lessons.Index) bool { return ix != nil && ix.IsBuilt() && !ix.RecallDisabled() }

// lessonsOff distinguishes "switched off for this instance" from "not built
// yet", which read the same to lessonReady but should not read the same to the
// operator or the model.
func lessonsOff(ix *lessons.Index) bool { return ix != nil && ix.RecallDisabled() }

type searchParams struct {
	Query string `json:"query" jsonschema:"required" jsonschema_description:"Natural-language description of what you need to do"`
	Limit int    `json:"limit,omitempty" jsonschema_description:"Max results per corpus (default 10)"`
}

// Search returns the recall_search tool over the skill and lesson indexes
// (either may be nil/unbuilt — that corpus is skipped).
func Search(skillIx *skills.Index, lessonIx *lessons.Index) *tool.Tool {
	// Describe only the corpora this instance will actually search: promising
	// lessons that can never be returned would send the model looking for them.
	corpora := "the skill library and the lessons mined from past runs"
	if lessonsOff(lessonIx) {
		corpora = "the skill library"
	}
	return &tool.Tool{
		Name: "recall_search",
		Description: "Search " + corpora + " for guides relevant to a task. " +
			"Returns handles + previews; pass a handle to recall_load to read the full guide. Use this before attempting " +
			"unfamiliar tools, internal systems, CLIs, or resource paths.",
		ParamType: &searchParams{},
		Execute: func(ctx context.Context, args json.RawMessage) (*tool.Result, error) {
			p, errResult := tool.ParseArgs[searchParams](args)
			if errResult != nil {
				return errResult, nil
			}
			limit := p.Limit
			if limit <= 0 {
				limit = 10
			}
			var b strings.Builder
			hitCount := 0

			if skillReady(skillIx) {
				hits, err := skillIx.Search(ctx, p.Query, limit)
				if err != nil {
					return &tool.Result{Content: "Error: " + err.Error(), IsError: true}, nil
				}
				if len(hits) > 0 {
					b.WriteString("Skills:\n")
					for _, h := range hits {
						fmt.Fprintf(&b, "  %s%s — %s\n", skillPrefix, h.Entry.Name, preview(h.Entry.Description))
					}
					hitCount += len(hits)
				}
			}
			if lessonReady(lessonIx) {
				hits, err := lessonIx.Search(ctx, p.Query, limit)
				if err != nil {
					return &tool.Result{Content: "Error: " + err.Error(), IsError: true}, nil
				}
				if len(hits) > 0 {
					if b.Len() > 0 {
						b.WriteString("\n")
					}
					b.WriteString("Lessons (mined from past runs):\n")
					for _, h := range hits {
						fmt.Fprintf(&b, "  %s%s — %s: %s\n", lessonPrefix, h.Lesson.LessonID, h.Lesson.Title, preview(h.Lesson.Description))
					}
					hitCount += len(hits)
				}
			}

			if hitCount == 0 {
				what := "skills or lessons"
				if lessonsOff(lessonIx) {
					what = "skills"
				}
				return &tool.Result{Content: fmt.Sprintf("No matching %s for %q.", what, p.Query)}, nil
			}
			b.WriteString("\nLoad one with recall_load(handle=\"skill:<name>\" or \"lesson:<id>\").")
			return &tool.Result{Content: b.String()}, nil
		},
	}
}

type loadParams struct {
	Handle string `json:"handle" jsonschema:"required" jsonschema_description:"A handle from recall_search, e.g. \"skill:blaze\" or \"lesson:abc123\""`
}

// Load returns the recall_load tool, dispatching on the handle prefix.
func Load(skillIx *skills.Index, lessonIx *lessons.Index) *tool.Tool {
	return &tool.Tool{
		Name:        "recall_load",
		Description: "Load the full content of a skill or lesson by its handle from recall_search.",
		ParamType:   &loadParams{},
		Execute: func(ctx context.Context, args json.RawMessage) (*tool.Result, error) {
			p, errResult := tool.ParseArgs[loadParams](args)
			if errResult != nil {
				return errResult, nil
			}
			handle := strings.TrimSpace(p.Handle)

			if name, ok := strings.CutPrefix(handle, skillPrefix); ok {
				if !skillReady(skillIx) {
					return &tool.Result{Content: "Skill recall is unavailable.", IsError: true}, nil
				}
				e, ok := skillIx.Load(name)
				if !ok {
					return &tool.Result{Content: fmt.Sprintf("No skill named %q.", name), IsError: true}, nil
				}
				return &tool.Result{Content: fmt.Sprintf("# Skill: %s\nPath: %s\n\n%s", e.Name, e.Path, e.Body)}, nil
			}
			if id, ok := strings.CutPrefix(handle, lessonPrefix); ok {
				if lessonsOff(lessonIx) {
					return &tool.Result{
						Content: "Lesson recall is disabled on this instance; skills are still available.",
						IsError: true,
					}, nil
				}
				if !lessonReady(lessonIx) {
					return &tool.Result{Content: "Lesson recall is unavailable.", IsError: true}, nil
				}
				l, ok := lessonIx.Load(id)
				if !ok {
					return &tool.Result{Content: fmt.Sprintf("No lesson with id %q.", id), IsError: true}, nil
				}
				// Usage tracking is best-effort and must not block returning the body.
				if err := lessonIx.RecordLoad(ctx, id); err != nil {
					slog.Warn("recall_load: record lesson load failed", "lesson_id", id, "error", err)
				}
				return &tool.Result{Content: formatLesson(l)}, nil
			}
			return &tool.Result{
				Content: fmt.Sprintf("Unknown handle %q; expected \"skill:<name>\" or \"lesson:<id>\".", p.Handle),
				IsError: true,
			}, nil
		},
	}
}

func formatLesson(l db.LessonRecord) string {
	src := l.SourceRunID
	if src == "" {
		src = "unknown"
	}
	return fmt.Sprintf("# Lesson: %s\n# Id: %s\n# Source run: %s\n# Score: %d (retrieved %d time(s))\n\n%s",
		l.Title, l.LessonID, src, l.Score, l.LoadedCount, l.Body)
}

// Seeder returns the session-start recall seed for an agent factory to wire as
// its InitialRecall, or nil when neither corpus exists. Shared so every agent
// type seeds the same way.
func Seeder(skillIx *skills.Index, lessonIx *lessons.Index) func(ctx context.Context, task string) string {
	if skillIx == nil && lessonIx == nil {
		return nil
	}
	return func(ctx context.Context, task string) string {
		return InitialContent(ctx, skillIx, lessonIx, task)
	}
}

// InitialContent returns the block seeded at session start so the agent sees
// applicable skills/lessons without searching first:
//
//   - every skill from a pinned source ([skills].pinned) — always, even with no
//     task (a chat session), since pinning doesn't depend on a query;
//   - the [skills].initial most relevant other skills, and the most relevant
//     lessons — only for a non-empty task, which is the query.
//
// Returns "" when there is nothing to list.
func InitialContent(ctx context.Context, skillIx *skills.Index, lessonIx *lessons.Index, task string) string {
	var b strings.Builder
	if skillReady(skillIx) {
		pinned, relevant, err := skillIx.Initial(ctx, task, skillIx.InitialLimit())
		if err != nil {
			// Pinned skills don't depend on the ranking; list them regardless.
			slog.Warn("initial skill recall: ranking failed; listing pinned skills only", "error", err)
		}
		if len(pinned) > 0 {
			b.WriteString("Skills pinned by the operator — always listed; read one with recall_load:\n")
			for _, e := range pinned {
				fmt.Fprintf(&b, "  %s%s — %s\n", skillPrefix, e.Name, preview(e.Description))
			}
		}
		if len(relevant) > 0 {
			if len(pinned) > 0 {
				b.WriteString("\nOther skills that may be relevant — read one with recall_load, or recall_search for more:\n")
			} else {
				b.WriteString("Skills that may be relevant — read one with recall_load, or recall_search for more:\n")
			}
			for _, h := range relevant {
				fmt.Fprintf(&b, "  %s%s — %s\n", skillPrefix, h.Entry.Name, preview(h.Entry.Description))
			}
		}
	}
	if task != "" && lessonReady(lessonIx) {
		if hits, err := lessonIx.Search(ctx, task, initialLessonHits); err == nil && len(hits) > 0 {
			if b.Len() > 0 {
				b.WriteString("\n")
			}
			b.WriteString("Lessons from past runs that may be relevant:\n")
			for _, h := range hits {
				fmt.Fprintf(&b, "  %s%s — %s: %s\n", lessonPrefix, h.Lesson.LessonID, h.Lesson.Title, preview(h.Lesson.Description))
			}
		}
	}
	return b.String()
}

func preview(s string) string {
	s = strings.Join(strings.Fields(s), " ") // collapse whitespace/newlines
	// Rune-safe truncation so a multibyte description isn't cut mid-rune.
	return util.TruncateRunes(s, descPreviewMax)
}
