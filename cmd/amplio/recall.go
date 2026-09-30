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
	"context"
	"fmt"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"time"

	"amplio/internal/config"
	"amplio/internal/db"
	"amplio/internal/embed"
	"amplio/internal/lessons"
	"amplio/internal/runtime"
	"amplio/internal/skills"
)

// Startup wiring for the recall subsystem (skill + lesson search), and the
// operator-facing status block that reports whether it came up.

// recallSubsystem captures whether one recall subsystem (skills or knowledge)
// came up, with a short human-readable detail (a count, or why it's disabled).
type recallSubsystem struct {
	enabled bool
	detail  string
}

func recallEnabled(detail string) recallSubsystem {
	return recallSubsystem{enabled: true, detail: detail}
}
func recallDisabled(reason string) recallSubsystem {
	return recallSubsystem{enabled: false, detail: reason}
}

// recallStatus is the startup outcome of the recall subsystem, rendered once by
// printRecallStatus so the operator can see at a glance whether skills +
// knowledge are live.
type recallStatus struct {
	embedModel string
	skills     recallSubsystem
	knowledge  recallSubsystem
}

// printRecallStatus writes an operator-facing block mirroring cli.PrintStatus:
// a green ✓ for a live subsystem, a yellow ✗ + reason for a disabled one. Color
// is used only when w is a TTY and NO_COLOR is unset.
func printRecallStatus(w io.Writer, st recallStatus) {
	const (
		reset  = "\x1b[0m"
		green  = "\x1b[32m"
		yellow = "\x1b[33m"
	)
	color := false
	if f, ok := w.(*os.File); ok && os.Getenv("NO_COLOR") == "" {
		if fi, err := f.Stat(); err == nil && fi.Mode()&os.ModeCharDevice != 0 {
			color = true
		}
	}
	paint := func(s, code string) string {
		if !color {
			return s
		}
		return code + s + reset
	}
	line := func(name string, sub recallSubsystem) {
		if sub.enabled {
			msg := "available"
			if sub.detail != "" {
				msg += " (" + sub.detail + ")"
			}
			fmt.Fprintf(w, "  %s %-9s %s\n", paint("✓", green), name, msg)
		} else {
			fmt.Fprintf(w, "  %s %-9s %s\n", paint("✗", yellow), name, paint("disabled — "+sub.detail, yellow))
		}
	}
	header := "Recall subsystem (agent skill + knowledge search):"
	if st.embedModel != "" {
		header = fmt.Sprintf("Recall subsystem (embed model %s):", st.embedModel)
	}
	fmt.Fprintln(w, header)
	line("skills", st.skills)
	line("knowledge", st.knowledge)
}

// pinnedSkillsWarn is the pinned-skill count above which startup warns: every
// session (sub-agents included) lists every pinned skill, so a large pinned set
// costs context on every agent. A warning, not a cap — the operator asked.
const pinnedSkillsWarn = 50

// skillSources maps the configured skill dirs to index sources, marking the
// ones named in [skills].pinned. It also returns the pinned entries that match
// no configured dir (compared after cleaning), which the caller reports and
// ignores.
func skillSources(dirs []string, sc config.SkillsConfig) (sources []skills.Source, unknownPinned []string) {
	pinned := make(map[string]bool, len(sc.Pinned))
	for _, p := range sc.Pinned {
		pinned[filepath.Clean(p)] = true
	}
	matched := make(map[string]bool, len(pinned))
	sources = make([]skills.Source, 0, len(dirs))
	for _, d := range dirs {
		isPinned := pinned[filepath.Clean(d)]
		if isPinned {
			matched[filepath.Clean(d)] = true
		}
		sources = append(sources, skills.Source{Name: d, Path: d, Blocked: sc.Blocked, Pinned: isPinned})
	}
	for _, p := range sc.Pinned {
		if !matched[filepath.Clean(p)] {
			unknownPinned = append(unknownPinned, p)
		}
	}
	return sources, unknownPinned
}

// skillCounts renders the skills status detail ("828 skills", or "828 skills, 20
// pinned"), warning when the pinned set is large enough to matter.
func skillCounts(ix *skills.Index, total int) string {
	n := ix.PinnedCount()
	if n == 0 {
		return fmt.Sprintf("%d skills", total)
	}
	if n > pinnedSkillsWarn {
		slog.Warn("many pinned skills: every session lists all of them at start", "pinned", n)
	}
	return fmt.Sprintf("%d skills, %d pinned", total, n)
}

// setupRecall builds the skill and lesson recall indexes from config, installs
// them on the manager, and returns them (nil when unavailable). Lessons build
// SYNCHRONOUSLY (DB-only — fast: ~ms).
//
// Skills use a two-stage build to keep startup snappy without sacrificing
// recall availability:
//
//  1. LoadCached (sync, ~ms): hydrate the in-memory Index from the DB cache.
//     IsBuilt() returns true immediately and Search/Load serve from the cached
//     snapshot, so freshly-spawned agents have full recall right away.
//  2. Reconcile (background, slow): full file scan against the skill source
//     tree, re-embed any changed/new skills, atomic-swap in the fresh index.
//     The skill tree often lives on a network FS where per-file reads are
//     tens of seconds cold; doing this in background means startup is
//     responsive even on a slow-srcfs day.
//
// Cold-start exception: if LoadCached found nothing (empty cache), we run
// Build synchronously instead — backgrounding would leave Search returning
// empty results for the entire scan duration, which is a much worse UX than
// the one-time wait. Subsequent restarts hit the warm cache and are instant.
//
// Degrades to no recall (logged) when the embedder is unavailable (e.g. no
// Vertex creds).
func setupRecall(ctx context.Context, mgr *runtime.RunManager, store db.Store, cfg config.Config) (*skills.Index, *lessons.Index, embed.Embedder) {
	var st recallStatus
	defer func() { printRecallStatus(os.Stderr, st) }()

	embedModel := cfg.EmbedModelOrDefault()
	if embedModel == "" {
		// No embedder configured: recall needs embeddings, so it's disabled.
		// Set --embed-model / $AMPLIO_EMBED_MODEL / embed_model.
		st.skills = recallDisabled("no embed model configured")
		st.knowledge = recallDisabled("no embed model configured")
		return nil, nil, nil
	}
	st.embedModel = embedModel
	embedder, err := createEmbedder(ctx, embedModel)
	if err != nil {
		reason := fmt.Sprintf("embedder unavailable: %s", err)
		st.skills = recallDisabled(reason)
		st.knowledge = recallDisabled(reason)
		return nil, nil, nil
	}

	// Lessons ("knowledge"): synchronous; reads from the DB, populated by
	// end-of-run mining.
	lessonIx := lessons.NewIndex(store, embedder)
	// Instance-wide isolation switch. The index is still built and still handed
	// to the manager: mining, its near-duplicate check and lesson scoring all
	// need it. Only the agent-facing surface (recall_search / recall_load /
	// the run-start seed) is closed off.
	if !cfg.LessonSearchEnabled() {
		lessonIx.DisableRecall()
		slog.Info("lesson search disabled: agents cannot read lessons from past runs; mining and scoring still run")
	}
	switch err := lessonIx.Build(ctx); {
	case err != nil:
		st.knowledge = recallDisabled(fmt.Sprintf("index build failed: %s", err))
	case !cfg.LessonSearchEnabled():
		st.knowledge = recallDisabled("search off (--lesson-search=false); mining + scoring still run")
	default:
		st.knowledge = recallEnabled("")
	}
	mgr.SetLessonIndex(lessonIx)

	// Skills: only when dirs are configured.
	dirs := cfg.SkillDirs()
	var skillIx *skills.Index
	if len(dirs) == 0 {
		st.skills = recallDisabled("no skill dirs configured")
		return skillIx, lessonIx, embedder
	}
	sources, unknownPinned := skillSources(dirs, cfg.Skills)
	if len(unknownPinned) > 0 {
		// A typo must not stop startup, but it must not pass silently either.
		slog.Warn("[skills] pinned names directories that are not skill dirs; ignored",
			"pinned", unknownPinned, "dirs", dirs)
	}
	skillIx = skills.NewIndex(sources, embedder, skills.NewDBCache(store))
	skillIx.SetInitialLimit(cfg.SkillsInitial())
	mgr.SetSkillIndex(skillIx)

	// Stage 1: hydrate from cache. Fast even with hundreds of skills.
	hydrated := skillIx.LoadCached(ctx)
	if hydrated == 0 {
		// Cold start: no cache to lean on. Block — empty Search results would
		// be worse than the one-time wait, and this only happens on the very
		// first run (or after a manual DB wipe).
		slog.Info("skill cache empty; indexing synchronously (cold start)", "dirs", dirs)
		if err := skillIx.Build(ctx); err != nil {
			st.skills = recallDisabled(fmt.Sprintf("index build failed: %s", err))
		} else {
			st.skills = recallEnabled(skillCounts(skillIx, skillIx.Size()))
		}
		return skillIx, lessonIx, embedder
	}

	// Stage 2: background reconcile against the on-disk corpus. During this
	// window the index serves cached results (possibly slightly stale for
	// skills whose SKILL.md changed since the last run); the atomic swap at
	// Build's end refreshes everything.
	st.skills = recallEnabled(skillCounts(skillIx, hydrated) + ", reconciling in background")
	go func() {
		start := time.Now()
		if err := skillIx.Build(ctx); err != nil {
			slog.Error("skill reconcile failed; serving stale cached index", "error", err)
			return
		}
		slog.Info("skill index reconciled", "elapsed", time.Since(start).Round(time.Second))
	}()
	return skillIx, lessonIx, embedder
}
