/// Riverpod providers wrapping the inference + planner + persistence layers.
///
/// Architecture:
///   1. Singletons (one-time init):
///      - databaseProvider     → AppDatabase (SQLite)
///      - modelAssetsProvider  → loaded TFLite + JSONs
///      - deepGainProvider     → DeepGain wired to assets
///      - strengthPriorsProvider, plannerMetaProvider
///
///   2. Persisted state (watched from db):
///      - anchorsProvider      → bench/squat/deadlift kg, null until onboarding
///      - historyProvider      → completed sets last 14d
///      - settingsProvider     → rest_sec, default rir, locale, theme
///
///   3. Computed:
///      - mpcProvider          → predictMpc(history, now) keyed off history+anchors
///      - plannerProvider      → KnapsackPlanner factory (anchors-dependent)
library;

import 'package:deepgain_app/data/database.dart';
import 'package:deepgain_app/inference/deepgain.dart';
import 'package:deepgain_app/inference/model_assets.dart';
import 'package:deepgain_app/inference/strength.dart';
import 'package:deepgain_app/inference/types.dart';
import 'package:deepgain_app/planner/knapsack_planner.dart';
import 'package:deepgain_app/planner/planner_meta.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

// ──────────────────────────────────────────────────────────────────────────────
// Singletons
// ──────────────────────────────────────────────────────────────────────────────

/// Upper bound on any single startup step the [_AppGate] splash waits on.
/// If a step (model load, asset decode, DB open) ever stalls instead of
/// completing or throwing, the `.timeout` turns the silent hang into a
/// [TimeoutException] so the gate shows an actionable error + retry rather
/// than spinning forever. This is what App Review flagged on iPad: a startup
/// future that never completed left the spinner running indefinitely.
const _startupTimeout = Duration(seconds: 15);

final databaseProvider = Provider<AppDatabase>((ref) {
  final db = AppDatabase();
  ref.onDispose(db.close);
  return db;
});

final modelAssetsProvider = FutureProvider<ModelAssets>((ref) async {
  final assets = await ModelAssets.load().timeout(_startupTimeout);
  ref.onDispose(assets.close);
  return assets;
});

final strengthPriorsProvider = FutureProvider<StrengthPriors>((ref) async {
  return StrengthPriors.load().timeout(_startupTimeout);
});

final plannerMetaProvider = FutureProvider<PlannerMeta>((ref) async {
  return PlannerMeta.load().timeout(_startupTimeout);
});

final deepGainProvider = FutureProvider<DeepGain>((ref) async {
  final assets = await ref.watch(modelAssetsProvider.future);
  return DeepGain.fromAssets(assets);
});

// ──────────────────────────────────────────────────────────────────────────────
// Persisted state
// ──────────────────────────────────────────────────────────────────────────────

/// Three-anchor 1RM in kg, or null if onboarding hasn't completed.
///
/// Originally a [StreamProvider] backed by drift's `watchAnchors()`. Switched
/// to [FutureProvider] because drift's cross-isolate watch propagation on
/// Android (NativeDatabase.createInBackground) is unreliable enough that the
/// onboarding screen's "save → gate sees new anchors" handoff was failing
/// silently (write committed, watch never fired, gate stayed on onboarding).
///
/// Write paths must call `ref.invalidate(anchorsProvider)` after `setAnchors`
/// to refresh — see `OnboardingScreen._start` and `SettingsScreen`.
final anchorsProvider = FutureProvider<AnchorsKg?>((ref) async {
  final db = ref.watch(databaseProvider);
  // Timed out so a stalled DB open (e.g. drift's background isolate failing to
  // come up) surfaces as an error instead of an endless splash. See
  // [_startupTimeout].
  final row = await db.getAnchors().timeout(_startupTimeout);
  if (row == null) return null;
  return [row.benchPressKg, row.squatKg, row.deadliftKg];
});

final hasOnboardedProvider = Provider<bool>((ref) {
  final anchors = ref.watch(anchorsProvider);
  return anchors.maybeWhen(data: (a) => a != null, orElse: () => false);
});

/// All completed sets in the last 14 days, oldest first.
final historyProvider = StreamProvider<List<WorkoutSet>>((ref) {
  final db = ref.watch(databaseProvider);
  return db.watchHistoryLastDays(days: 14).map((rows) => rows
      .map((r) => WorkoutSet(
            exercise: r.exerciseId,
            weightKg: r.weightKg,
            reps: r.reps,
            rir: r.rir,
            timestamp: r.timestamp,
          ))
      .toList());
});

final settingsProvider = StreamProvider<AppSetting?>((ref) {
  final db = ref.watch(databaseProvider);
  return db.watchSettings();
});

/// Recent sessions for the History tab. FutureProvider (not Stream) for the
/// same reason as anchorsProvider — invalidate-and-refresh is more reliable
/// across Android isolates than relying on drift's watch.
final sessionsProvider = FutureProvider<List<Session>>((ref) async {
  final db = ref.watch(databaseProvider);
  return db.recentSessions(limit: 60);
});

// ──────────────────────────────────────────────────────────────────────────────
// Computed
// ──────────────────────────────────────────────────────────────────────────────

/// Current MPC for all 15 muscles, computed from the last 14d of history.
/// Recomputed whenever history or anchors change.
final mpcProvider = FutureProvider<Mpc>((ref) async {
  final dg = await ref.watch(deepGainProvider.future);
  final priors = await ref.watch(strengthPriorsProvider.future);
  final anchors = ref.watch(anchorsProvider).valueOrNull;
  final history = ref.watch(historyProvider).valueOrNull ?? const [];

  return dg.predictMpc(
    history: history,
    timestamp: DateTime.now(),
    anchorsKg: anchors,
    strengthPriors: priors,
  );
});

/// A KnapsackPlanner ready to plan, scoped to current anchors + meta + model.
/// Re-created when anchors change.
final plannerProvider = FutureProvider<KnapsackPlanner>((ref) async {
  final dg = await ref.watch(deepGainProvider.future);
  final priors = await ref.watch(strengthPriorsProvider.future);
  final meta = await ref.watch(plannerMetaProvider.future);
  final settings = ref.watch(settingsProvider).valueOrNull;
  final anchors = ref.watch(anchorsProvider).valueOrNull ?? priors.defaultAnchorsKg;

  return KnapsackPlanner(
    deepgain: dg,
    strengthPriors: priors,
    meta: meta,
    strengthAnchors: anchors,
    restBetweenSetsSec: settings?.restBetweenSetsSec ?? 180,
  );
});
