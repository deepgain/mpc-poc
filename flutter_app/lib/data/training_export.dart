/// Export the user's completed-set history as a CSV in the **DeepGain Data
/// Standard v2** format, the same one the model was trained on.
///
/// Per `dataset/generate_training_data.py`:
/// ```
/// user_id, exercise, weight_kg, reps, rir, timestamp,
/// config_1rm_bench_press, config_1rm_squat, config_1rm_deadlift
/// ```
///
/// timestamp is ISO 8601, RIR is rounded to int (the trainer expects 0–5 int),
/// each row's `config_1rm_*` columns hold the **anchor snapshot at the time
/// the set was recorded** — not the user's current anchors. This is critical
/// for retraining: the same user with evolving 1RMs needs the historical
/// context preserved.
library;

import 'dart:io';

import 'package:deepgain_app/data/database.dart';
import 'package:drift/drift.dart' show OrderingTerm;
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';
import 'package:uuid/uuid.dart';

class TrainingDataExporter {
  final AppDatabase db;
  static const _uuid = Uuid();

  TrainingDataExporter(this.db);

  /// Build the CSV string and write it to a temp file (so it can be shared via
  /// the system share sheet). Returns the file path.
  Future<File> exportToFile() async {
    final csv = await buildCsv();
    final tmp = await getTemporaryDirectory();
    final ts = DateTime.now()
        .toIso8601String()
        .replaceAll(':', '-')
        .substring(0, 19);
    final f = File(p.join(tmp.path, 'deepgain_training_data_$ts.csv'));
    await f.writeAsString(csv);
    return f;
  }

  Future<String> buildCsv() async {
    final userId = await db.ensureUserId(_uuid.v4);
    final rows = await (db.select(db.completedSets)
          ..orderBy([(t) => OrderingTerm.asc(t.timestamp)]))
        .get();

    final buf = StringBuffer();
    buf.writeln('user_id,exercise,weight_kg,reps,rir,timestamp,'
        'config_1rm_bench_press,config_1rm_squat,config_1rm_deadlift');
    for (final r in rows) {
      buf.writeln([
        _csv(userId),
        _csv(r.exerciseId),
        r.weightKg.toStringAsFixed(2),
        r.reps,
        // The trainer treats rir as int (0–5). We store it as double for
        // future flexibility but emit it as int here for compatibility.
        r.rir.round().clamp(0, 5),
        _csv(r.timestamp.toUtc().toIso8601String()),
        r.anchorsBenchKg.toStringAsFixed(1),
        r.anchorsSquatKg.toStringAsFixed(1),
        r.anchorsDeadliftKg.toStringAsFixed(1),
      ].join(','));
    }
    return buf.toString();
  }

  /// Number of completed sets — useful for "Export 247 sets" UI.
  Future<int> completedSetsCount() async {
    final rows = await db.select(db.completedSets).get();
    return rows.length;
  }

  /// Wrap a value in double quotes if it contains commas or quotes.
  /// (Exercise IDs are safe snake_case; timestamps are ISO; user_id is UUID;
  /// none should ever need quoting in practice, but we're defensive.)
  static String _csv(String s) {
    if (s.contains(',') || s.contains('"') || s.contains('\n')) {
      return '"${s.replaceAll('"', '""')}"';
    }
    return s;
  }
}
