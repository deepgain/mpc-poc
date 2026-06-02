/// Local SQLite database — single source of truth for app state.
///
/// Tables:
///   - [Anchors]         — current bench/squat/deadlift in kg + last update.
///                         Single-row table (id always 0).
///   - [Sessions]        — one row per training session (start, optional end).
///   - [CompletedSets]   — every set the user marked done (linked to a session
///                         when one was active, otherwise standalone history).
///   - [DismissedLog]    — analytics: which exercises got dropped during a session.
///   - [Settings]        — single-row settings (rest_sec, target_rir defaults).
///
/// MPC is NEVER persisted — it's recomputed from history via DeepGain.predictMpc.
library;

import 'dart:io';

import 'package:drift/drift.dart';
import 'package:drift/native.dart';
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';

part 'database.g.dart';

// ──────────────────────────────────────────────────────────────────────────────
// Schema
// ──────────────────────────────────────────────────────────────────────────────

class Anchors extends Table {
  IntColumn get id => integer().withDefault(const Constant(0))();
  RealColumn get benchPressKg => real()();
  RealColumn get squatKg => real()();
  RealColumn get deadliftKg => real()();
  DateTimeColumn get updatedAt => dateTime()();

  @override
  Set<Column> get primaryKey => {id};
}

class Sessions extends Table {
  IntColumn get id => integer().autoIncrement()();
  DateTimeColumn get startedAt => dateTime()();
  DateTimeColumn get endedAt => dateTime().nullable()();
  IntColumn get timeBudgetSec => integer().nullable()();
  IntColumn get targetRir => integer().nullable()();
}

class CompletedSets extends Table {
  IntColumn get id => integer().autoIncrement()();
  IntColumn get sessionId =>
      integer().nullable().references(Sessions, #id, onDelete: KeyAction.setNull)();
  TextColumn get exerciseId => text()();
  RealColumn get weightKg => real()();
  IntColumn get reps => integer()();
  RealColumn get rir => real()();
  DateTimeColumn get timestamp => dateTime()();

  // Anchors snapshot AT the time the set was completed. Required for
  // training-data-quality export — anchors evolve over time, so historical
  // sets need their then-current 1RMs preserved (matches the
  // config_1rm_{bench_press,squat,deadlift} columns in the training CSV).
  RealColumn get anchorsBenchKg =>
      real().withDefault(const Constant(100.0))();
  RealColumn get anchorsSquatKg =>
      real().withDefault(const Constant(140.0))();
  RealColumn get anchorsDeadliftKg =>
      real().withDefault(const Constant(180.0))();
}

class DismissedLog extends Table {
  IntColumn get id => integer().autoIncrement()();
  IntColumn get sessionId =>
      integer().references(Sessions, #id, onDelete: KeyAction.cascade)();
  TextColumn get exerciseId => text()();
  DateTimeColumn get timestamp => dateTime()();
}

class AppSettings extends Table {
  IntColumn get id => integer().withDefault(const Constant(0))();
  IntColumn get restBetweenSetsSec => integer().withDefault(const Constant(180))();
  IntColumn get defaultTargetRir => integer().withDefault(const Constant(2))();
  TextColumn get themeMode => text().withDefault(const Constant('system'))();

  // Stable per-device user ID. Generated on first launch (UUID v4) so that
  // exported training data attributes sets to a consistent identity. Empty
  // string until set by ensureUserId().
  TextColumn get userId => text().withDefault(const Constant(''))();

  @override
  Set<Column> get primaryKey => {id};
}

// ──────────────────────────────────────────────────────────────────────────────
// Database
// ──────────────────────────────────────────────────────────────────────────────

@DriftDatabase(tables: [Anchors, Sessions, CompletedSets, DismissedLog, AppSettings])
class AppDatabase extends _$AppDatabase {
  AppDatabase() : super(_open());

  // For tests: pass a fresh in-memory NativeDatabase.
  AppDatabase.forTesting(super.executor);

  @override
  int get schemaVersion => 2;

  @override
  MigrationStrategy get migration => MigrationStrategy(
        onUpgrade: (m, from, to) async {
          if (from < 2) {
            // v2: add user_id to settings; anchor snapshot columns to completed_sets.
            // Existing completed_sets rows get the current anchors (best we can do
            // for sets recorded before the snapshot existed).
            await m.addColumn(appSettings, appSettings.userId);
            await m.addColumn(completedSets, completedSets.anchorsBenchKg);
            await m.addColumn(completedSets, completedSets.anchorsSquatKg);
            await m.addColumn(completedSets, completedSets.anchorsDeadliftKg);

            final cur = await getAnchors();
            if (cur != null) {
              await customStatement(
                'UPDATE completed_sets SET '
                'anchors_bench_kg = ?, anchors_squat_kg = ?, anchors_deadlift_kg = ?',
                [cur.benchPressKg, cur.squatKg, cur.deadliftKg],
              );
            }
          }
        },
      );

  // ── Anchors ────────────────────────────────────────────────────────────

  Future<Anchor?> getAnchors() async =>
      (select(anchors)..where((t) => t.id.equals(0))).getSingleOrNull();

  Future<void> setAnchors({
    required double benchKg,
    required double squatKg,
    required double deadliftKg,
  }) async {
    // Single-row table semantics — wipe + insert in a transaction. The
    // previous "insertOnConflictUpdate without explicit id" approach left
    // rows at SQLite-assigned IDs that getAnchors WHERE id=0 couldn't see,
    // so the gate stayed stuck on onboarding.
    await transaction(() async {
      await delete(anchors).go();
      await into(anchors).insert(AnchorsCompanion.insert(
        id: const Value(0),
        benchPressKg: benchKg,
        squatKg: squatKg,
        deadliftKg: deadliftKg,
        updatedAt: DateTime.now(),
      ));
    });
  }

  Stream<Anchor?> watchAnchors() =>
      (select(anchors)..where((t) => t.id.equals(0))).watchSingleOrNull();

  // ── Sessions ───────────────────────────────────────────────────────────

  Future<int> startSession({
    required int timeBudgetSec,
    required int targetRir,
  }) async {
    return into(sessions).insert(SessionsCompanion.insert(
      startedAt: DateTime.now(),
      timeBudgetSec: Value(timeBudgetSec),
      targetRir: Value(targetRir),
    ));
  }

  Future<void> endSession(int sessionId) async {
    await (update(sessions)..where((t) => t.id.equals(sessionId))).write(
      SessionsCompanion(endedAt: Value(DateTime.now())),
    );
  }

  Future<List<Session>> recentSessions({int limit = 30}) =>
      (select(sessions)
            ..orderBy([(t) => OrderingTerm.desc(t.startedAt)])
            ..limit(limit))
          .get();

  // ── Completed sets ─────────────────────────────────────────────────────

  Future<void> addCompletedSet({
    int? sessionId,
    required String exerciseId,
    required double weightKg,
    required int reps,
    required double rir,
    required DateTime timestamp,
  }) async {
    // Snapshot the user's CURRENT anchors at the moment the set is recorded.
    // These persist on the row even if the user later updates their 1RMs —
    // required for training-data-quality export.
    final currentAnchors = await getAnchors();
    await into(completedSets).insert(CompletedSetsCompanion.insert(
      sessionId: Value(sessionId),
      exerciseId: exerciseId,
      weightKg: weightKg,
      reps: reps,
      rir: rir,
      timestamp: timestamp,
      anchorsBenchKg: Value(currentAnchors?.benchPressKg ?? 100.0),
      anchorsSquatKg: Value(currentAnchors?.squatKg ?? 140.0),
      anchorsDeadliftKg: Value(currentAnchors?.deadliftKg ?? 180.0),
    ));
  }

  /// Last [days] of completed sets, oldest first — used to seed predict_mpc.
  Future<List<CompletedSet>> historyLastDays({int days = 14}) {
    final cutoff = DateTime.now().subtract(Duration(days: days));
    return (select(completedSets)
          ..where((t) => t.timestamp.isBiggerThanValue(cutoff))
          ..orderBy([(t) => OrderingTerm.asc(t.timestamp)]))
        .get();
  }

  Stream<List<CompletedSet>> watchHistoryLastDays({int days = 14}) {
    final cutoff = DateTime.now().subtract(Duration(days: days));
    return (select(completedSets)
          ..where((t) => t.timestamp.isBiggerThanValue(cutoff))
          ..orderBy([(t) => OrderingTerm.asc(t.timestamp)]))
        .watch();
  }

  Future<List<CompletedSet>> setsForSession(int sessionId) =>
      (select(completedSets)
            ..where((t) => t.sessionId.equals(sessionId))
            ..orderBy([(t) => OrderingTerm.asc(t.timestamp)]))
          .get();

  // ── Dismissed log ──────────────────────────────────────────────────────

  Future<void> logDismissed(int sessionId, String exerciseId) async {
    await into(dismissedLog).insert(DismissedLogCompanion.insert(
      sessionId: sessionId,
      exerciseId: exerciseId,
      timestamp: DateTime.now(),
    ));
  }

  // ── Settings ───────────────────────────────────────────────────────────

  Future<AppSetting> getOrCreateSettings() async {
    var row = await (select(appSettings)..where((t) => t.id.equals(0)))
        .getSingleOrNull();
    if (row == null) {
      // See note on setAnchors — id=0 must be explicit for single-row tables.
      await into(appSettings).insert(
        const AppSettingsCompanion(id: Value(0)),
      );
      row = await (select(appSettings)..where((t) => t.id.equals(0))).getSingle();
    }
    return row;
  }

  Stream<AppSetting?> watchSettings() =>
      (select(appSettings)..where((t) => t.id.equals(0))).watchSingleOrNull();

  Future<void> updateSettings({
    int? restBetweenSetsSec,
    int? defaultTargetRir,
    String? themeMode,
    String? userId,
  }) async {
    await getOrCreateSettings();
    await (update(appSettings)..where((t) => t.id.equals(0))).write(
      AppSettingsCompanion(
        restBetweenSetsSec: restBetweenSetsSec != null
            ? Value(restBetweenSetsSec) : const Value.absent(),
        defaultTargetRir: defaultTargetRir != null
            ? Value(defaultTargetRir) : const Value.absent(),
        themeMode: themeMode != null ? Value(themeMode) : const Value.absent(),
        userId: userId != null ? Value(userId) : const Value.absent(),
      ),
    );
  }

  /// Generate a stable user_id (UUID v4) on first call; return existing on
  /// subsequent calls. Used by training-data CSV export.
  Future<String> ensureUserId(String Function() generate) async {
    final settings = await getOrCreateSettings();
    if (settings.userId.isNotEmpty) return settings.userId;
    final newId = generate();
    await updateSettings(userId: newId);
    return newId;
  }
}

LazyDatabase _open() {
  return LazyDatabase(() async {
    final dir = await getApplicationDocumentsDirectory();
    return NativeDatabase.createInBackground(File(p.join(dir.path, 'deepgain.db')));
  });
}
