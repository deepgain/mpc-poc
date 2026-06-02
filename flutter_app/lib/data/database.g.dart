// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'database.dart';

// ignore_for_file: type=lint
class $AnchorsTable extends Anchors with TableInfo<$AnchorsTable, Anchor> {
  @override
  final GeneratedDatabase attachedDatabase;
  final String? _alias;
  $AnchorsTable(this.attachedDatabase, [this._alias]);
  static const VerificationMeta _idMeta = const VerificationMeta('id');
  @override
  late final GeneratedColumn<int> id = GeneratedColumn<int>(
      'id', aliasedName, false,
      type: DriftSqlType.int,
      requiredDuringInsert: false,
      defaultValue: const Constant(0));
  static const VerificationMeta _benchPressKgMeta =
      const VerificationMeta('benchPressKg');
  @override
  late final GeneratedColumn<double> benchPressKg = GeneratedColumn<double>(
      'bench_press_kg', aliasedName, false,
      type: DriftSqlType.double, requiredDuringInsert: true);
  static const VerificationMeta _squatKgMeta =
      const VerificationMeta('squatKg');
  @override
  late final GeneratedColumn<double> squatKg = GeneratedColumn<double>(
      'squat_kg', aliasedName, false,
      type: DriftSqlType.double, requiredDuringInsert: true);
  static const VerificationMeta _deadliftKgMeta =
      const VerificationMeta('deadliftKg');
  @override
  late final GeneratedColumn<double> deadliftKg = GeneratedColumn<double>(
      'deadlift_kg', aliasedName, false,
      type: DriftSqlType.double, requiredDuringInsert: true);
  static const VerificationMeta _updatedAtMeta =
      const VerificationMeta('updatedAt');
  @override
  late final GeneratedColumn<DateTime> updatedAt = GeneratedColumn<DateTime>(
      'updated_at', aliasedName, false,
      type: DriftSqlType.dateTime, requiredDuringInsert: true);
  @override
  List<GeneratedColumn> get $columns =>
      [id, benchPressKg, squatKg, deadliftKg, updatedAt];
  @override
  String get aliasedName => _alias ?? actualTableName;
  @override
  String get actualTableName => $name;
  static const String $name = 'anchors';
  @override
  VerificationContext validateIntegrity(Insertable<Anchor> instance,
      {bool isInserting = false}) {
    final context = VerificationContext();
    final data = instance.toColumns(true);
    if (data.containsKey('id')) {
      context.handle(_idMeta, id.isAcceptableOrUnknown(data['id']!, _idMeta));
    }
    if (data.containsKey('bench_press_kg')) {
      context.handle(
          _benchPressKgMeta,
          benchPressKg.isAcceptableOrUnknown(
              data['bench_press_kg']!, _benchPressKgMeta));
    } else if (isInserting) {
      context.missing(_benchPressKgMeta);
    }
    if (data.containsKey('squat_kg')) {
      context.handle(_squatKgMeta,
          squatKg.isAcceptableOrUnknown(data['squat_kg']!, _squatKgMeta));
    } else if (isInserting) {
      context.missing(_squatKgMeta);
    }
    if (data.containsKey('deadlift_kg')) {
      context.handle(
          _deadliftKgMeta,
          deadliftKg.isAcceptableOrUnknown(
              data['deadlift_kg']!, _deadliftKgMeta));
    } else if (isInserting) {
      context.missing(_deadliftKgMeta);
    }
    if (data.containsKey('updated_at')) {
      context.handle(_updatedAtMeta,
          updatedAt.isAcceptableOrUnknown(data['updated_at']!, _updatedAtMeta));
    } else if (isInserting) {
      context.missing(_updatedAtMeta);
    }
    return context;
  }

  @override
  Set<GeneratedColumn> get $primaryKey => {id};
  @override
  Anchor map(Map<String, dynamic> data, {String? tablePrefix}) {
    final effectivePrefix = tablePrefix != null ? '$tablePrefix.' : '';
    return Anchor(
      id: attachedDatabase.typeMapping
          .read(DriftSqlType.int, data['${effectivePrefix}id'])!,
      benchPressKg: attachedDatabase.typeMapping
          .read(DriftSqlType.double, data['${effectivePrefix}bench_press_kg'])!,
      squatKg: attachedDatabase.typeMapping
          .read(DriftSqlType.double, data['${effectivePrefix}squat_kg'])!,
      deadliftKg: attachedDatabase.typeMapping
          .read(DriftSqlType.double, data['${effectivePrefix}deadlift_kg'])!,
      updatedAt: attachedDatabase.typeMapping
          .read(DriftSqlType.dateTime, data['${effectivePrefix}updated_at'])!,
    );
  }

  @override
  $AnchorsTable createAlias(String alias) {
    return $AnchorsTable(attachedDatabase, alias);
  }
}

class Anchor extends DataClass implements Insertable<Anchor> {
  final int id;
  final double benchPressKg;
  final double squatKg;
  final double deadliftKg;
  final DateTime updatedAt;
  const Anchor(
      {required this.id,
      required this.benchPressKg,
      required this.squatKg,
      required this.deadliftKg,
      required this.updatedAt});
  @override
  Map<String, Expression> toColumns(bool nullToAbsent) {
    final map = <String, Expression>{};
    map['id'] = Variable<int>(id);
    map['bench_press_kg'] = Variable<double>(benchPressKg);
    map['squat_kg'] = Variable<double>(squatKg);
    map['deadlift_kg'] = Variable<double>(deadliftKg);
    map['updated_at'] = Variable<DateTime>(updatedAt);
    return map;
  }

  AnchorsCompanion toCompanion(bool nullToAbsent) {
    return AnchorsCompanion(
      id: Value(id),
      benchPressKg: Value(benchPressKg),
      squatKg: Value(squatKg),
      deadliftKg: Value(deadliftKg),
      updatedAt: Value(updatedAt),
    );
  }

  factory Anchor.fromJson(Map<String, dynamic> json,
      {ValueSerializer? serializer}) {
    serializer ??= driftRuntimeOptions.defaultSerializer;
    return Anchor(
      id: serializer.fromJson<int>(json['id']),
      benchPressKg: serializer.fromJson<double>(json['benchPressKg']),
      squatKg: serializer.fromJson<double>(json['squatKg']),
      deadliftKg: serializer.fromJson<double>(json['deadliftKg']),
      updatedAt: serializer.fromJson<DateTime>(json['updatedAt']),
    );
  }
  @override
  Map<String, dynamic> toJson({ValueSerializer? serializer}) {
    serializer ??= driftRuntimeOptions.defaultSerializer;
    return <String, dynamic>{
      'id': serializer.toJson<int>(id),
      'benchPressKg': serializer.toJson<double>(benchPressKg),
      'squatKg': serializer.toJson<double>(squatKg),
      'deadliftKg': serializer.toJson<double>(deadliftKg),
      'updatedAt': serializer.toJson<DateTime>(updatedAt),
    };
  }

  Anchor copyWith(
          {int? id,
          double? benchPressKg,
          double? squatKg,
          double? deadliftKg,
          DateTime? updatedAt}) =>
      Anchor(
        id: id ?? this.id,
        benchPressKg: benchPressKg ?? this.benchPressKg,
        squatKg: squatKg ?? this.squatKg,
        deadliftKg: deadliftKg ?? this.deadliftKg,
        updatedAt: updatedAt ?? this.updatedAt,
      );
  Anchor copyWithCompanion(AnchorsCompanion data) {
    return Anchor(
      id: data.id.present ? data.id.value : this.id,
      benchPressKg: data.benchPressKg.present
          ? data.benchPressKg.value
          : this.benchPressKg,
      squatKg: data.squatKg.present ? data.squatKg.value : this.squatKg,
      deadliftKg:
          data.deadliftKg.present ? data.deadliftKg.value : this.deadliftKg,
      updatedAt: data.updatedAt.present ? data.updatedAt.value : this.updatedAt,
    );
  }

  @override
  String toString() {
    return (StringBuffer('Anchor(')
          ..write('id: $id, ')
          ..write('benchPressKg: $benchPressKg, ')
          ..write('squatKg: $squatKg, ')
          ..write('deadliftKg: $deadliftKg, ')
          ..write('updatedAt: $updatedAt')
          ..write(')'))
        .toString();
  }

  @override
  int get hashCode =>
      Object.hash(id, benchPressKg, squatKg, deadliftKg, updatedAt);
  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      (other is Anchor &&
          other.id == this.id &&
          other.benchPressKg == this.benchPressKg &&
          other.squatKg == this.squatKg &&
          other.deadliftKg == this.deadliftKg &&
          other.updatedAt == this.updatedAt);
}

class AnchorsCompanion extends UpdateCompanion<Anchor> {
  final Value<int> id;
  final Value<double> benchPressKg;
  final Value<double> squatKg;
  final Value<double> deadliftKg;
  final Value<DateTime> updatedAt;
  const AnchorsCompanion({
    this.id = const Value.absent(),
    this.benchPressKg = const Value.absent(),
    this.squatKg = const Value.absent(),
    this.deadliftKg = const Value.absent(),
    this.updatedAt = const Value.absent(),
  });
  AnchorsCompanion.insert({
    this.id = const Value.absent(),
    required double benchPressKg,
    required double squatKg,
    required double deadliftKg,
    required DateTime updatedAt,
  })  : benchPressKg = Value(benchPressKg),
        squatKg = Value(squatKg),
        deadliftKg = Value(deadliftKg),
        updatedAt = Value(updatedAt);
  static Insertable<Anchor> custom({
    Expression<int>? id,
    Expression<double>? benchPressKg,
    Expression<double>? squatKg,
    Expression<double>? deadliftKg,
    Expression<DateTime>? updatedAt,
  }) {
    return RawValuesInsertable({
      if (id != null) 'id': id,
      if (benchPressKg != null) 'bench_press_kg': benchPressKg,
      if (squatKg != null) 'squat_kg': squatKg,
      if (deadliftKg != null) 'deadlift_kg': deadliftKg,
      if (updatedAt != null) 'updated_at': updatedAt,
    });
  }

  AnchorsCompanion copyWith(
      {Value<int>? id,
      Value<double>? benchPressKg,
      Value<double>? squatKg,
      Value<double>? deadliftKg,
      Value<DateTime>? updatedAt}) {
    return AnchorsCompanion(
      id: id ?? this.id,
      benchPressKg: benchPressKg ?? this.benchPressKg,
      squatKg: squatKg ?? this.squatKg,
      deadliftKg: deadliftKg ?? this.deadliftKg,
      updatedAt: updatedAt ?? this.updatedAt,
    );
  }

  @override
  Map<String, Expression> toColumns(bool nullToAbsent) {
    final map = <String, Expression>{};
    if (id.present) {
      map['id'] = Variable<int>(id.value);
    }
    if (benchPressKg.present) {
      map['bench_press_kg'] = Variable<double>(benchPressKg.value);
    }
    if (squatKg.present) {
      map['squat_kg'] = Variable<double>(squatKg.value);
    }
    if (deadliftKg.present) {
      map['deadlift_kg'] = Variable<double>(deadliftKg.value);
    }
    if (updatedAt.present) {
      map['updated_at'] = Variable<DateTime>(updatedAt.value);
    }
    return map;
  }

  @override
  String toString() {
    return (StringBuffer('AnchorsCompanion(')
          ..write('id: $id, ')
          ..write('benchPressKg: $benchPressKg, ')
          ..write('squatKg: $squatKg, ')
          ..write('deadliftKg: $deadliftKg, ')
          ..write('updatedAt: $updatedAt')
          ..write(')'))
        .toString();
  }
}

class $SessionsTable extends Sessions with TableInfo<$SessionsTable, Session> {
  @override
  final GeneratedDatabase attachedDatabase;
  final String? _alias;
  $SessionsTable(this.attachedDatabase, [this._alias]);
  static const VerificationMeta _idMeta = const VerificationMeta('id');
  @override
  late final GeneratedColumn<int> id = GeneratedColumn<int>(
      'id', aliasedName, false,
      hasAutoIncrement: true,
      type: DriftSqlType.int,
      requiredDuringInsert: false,
      defaultConstraints:
          GeneratedColumn.constraintIsAlways('PRIMARY KEY AUTOINCREMENT'));
  static const VerificationMeta _startedAtMeta =
      const VerificationMeta('startedAt');
  @override
  late final GeneratedColumn<DateTime> startedAt = GeneratedColumn<DateTime>(
      'started_at', aliasedName, false,
      type: DriftSqlType.dateTime, requiredDuringInsert: true);
  static const VerificationMeta _endedAtMeta =
      const VerificationMeta('endedAt');
  @override
  late final GeneratedColumn<DateTime> endedAt = GeneratedColumn<DateTime>(
      'ended_at', aliasedName, true,
      type: DriftSqlType.dateTime, requiredDuringInsert: false);
  static const VerificationMeta _timeBudgetSecMeta =
      const VerificationMeta('timeBudgetSec');
  @override
  late final GeneratedColumn<int> timeBudgetSec = GeneratedColumn<int>(
      'time_budget_sec', aliasedName, true,
      type: DriftSqlType.int, requiredDuringInsert: false);
  static const VerificationMeta _targetRirMeta =
      const VerificationMeta('targetRir');
  @override
  late final GeneratedColumn<int> targetRir = GeneratedColumn<int>(
      'target_rir', aliasedName, true,
      type: DriftSqlType.int, requiredDuringInsert: false);
  @override
  List<GeneratedColumn> get $columns =>
      [id, startedAt, endedAt, timeBudgetSec, targetRir];
  @override
  String get aliasedName => _alias ?? actualTableName;
  @override
  String get actualTableName => $name;
  static const String $name = 'sessions';
  @override
  VerificationContext validateIntegrity(Insertable<Session> instance,
      {bool isInserting = false}) {
    final context = VerificationContext();
    final data = instance.toColumns(true);
    if (data.containsKey('id')) {
      context.handle(_idMeta, id.isAcceptableOrUnknown(data['id']!, _idMeta));
    }
    if (data.containsKey('started_at')) {
      context.handle(_startedAtMeta,
          startedAt.isAcceptableOrUnknown(data['started_at']!, _startedAtMeta));
    } else if (isInserting) {
      context.missing(_startedAtMeta);
    }
    if (data.containsKey('ended_at')) {
      context.handle(_endedAtMeta,
          endedAt.isAcceptableOrUnknown(data['ended_at']!, _endedAtMeta));
    }
    if (data.containsKey('time_budget_sec')) {
      context.handle(
          _timeBudgetSecMeta,
          timeBudgetSec.isAcceptableOrUnknown(
              data['time_budget_sec']!, _timeBudgetSecMeta));
    }
    if (data.containsKey('target_rir')) {
      context.handle(_targetRirMeta,
          targetRir.isAcceptableOrUnknown(data['target_rir']!, _targetRirMeta));
    }
    return context;
  }

  @override
  Set<GeneratedColumn> get $primaryKey => {id};
  @override
  Session map(Map<String, dynamic> data, {String? tablePrefix}) {
    final effectivePrefix = tablePrefix != null ? '$tablePrefix.' : '';
    return Session(
      id: attachedDatabase.typeMapping
          .read(DriftSqlType.int, data['${effectivePrefix}id'])!,
      startedAt: attachedDatabase.typeMapping
          .read(DriftSqlType.dateTime, data['${effectivePrefix}started_at'])!,
      endedAt: attachedDatabase.typeMapping
          .read(DriftSqlType.dateTime, data['${effectivePrefix}ended_at']),
      timeBudgetSec: attachedDatabase.typeMapping
          .read(DriftSqlType.int, data['${effectivePrefix}time_budget_sec']),
      targetRir: attachedDatabase.typeMapping
          .read(DriftSqlType.int, data['${effectivePrefix}target_rir']),
    );
  }

  @override
  $SessionsTable createAlias(String alias) {
    return $SessionsTable(attachedDatabase, alias);
  }
}

class Session extends DataClass implements Insertable<Session> {
  final int id;
  final DateTime startedAt;
  final DateTime? endedAt;
  final int? timeBudgetSec;
  final int? targetRir;
  const Session(
      {required this.id,
      required this.startedAt,
      this.endedAt,
      this.timeBudgetSec,
      this.targetRir});
  @override
  Map<String, Expression> toColumns(bool nullToAbsent) {
    final map = <String, Expression>{};
    map['id'] = Variable<int>(id);
    map['started_at'] = Variable<DateTime>(startedAt);
    if (!nullToAbsent || endedAt != null) {
      map['ended_at'] = Variable<DateTime>(endedAt);
    }
    if (!nullToAbsent || timeBudgetSec != null) {
      map['time_budget_sec'] = Variable<int>(timeBudgetSec);
    }
    if (!nullToAbsent || targetRir != null) {
      map['target_rir'] = Variable<int>(targetRir);
    }
    return map;
  }

  SessionsCompanion toCompanion(bool nullToAbsent) {
    return SessionsCompanion(
      id: Value(id),
      startedAt: Value(startedAt),
      endedAt: endedAt == null && nullToAbsent
          ? const Value.absent()
          : Value(endedAt),
      timeBudgetSec: timeBudgetSec == null && nullToAbsent
          ? const Value.absent()
          : Value(timeBudgetSec),
      targetRir: targetRir == null && nullToAbsent
          ? const Value.absent()
          : Value(targetRir),
    );
  }

  factory Session.fromJson(Map<String, dynamic> json,
      {ValueSerializer? serializer}) {
    serializer ??= driftRuntimeOptions.defaultSerializer;
    return Session(
      id: serializer.fromJson<int>(json['id']),
      startedAt: serializer.fromJson<DateTime>(json['startedAt']),
      endedAt: serializer.fromJson<DateTime?>(json['endedAt']),
      timeBudgetSec: serializer.fromJson<int?>(json['timeBudgetSec']),
      targetRir: serializer.fromJson<int?>(json['targetRir']),
    );
  }
  @override
  Map<String, dynamic> toJson({ValueSerializer? serializer}) {
    serializer ??= driftRuntimeOptions.defaultSerializer;
    return <String, dynamic>{
      'id': serializer.toJson<int>(id),
      'startedAt': serializer.toJson<DateTime>(startedAt),
      'endedAt': serializer.toJson<DateTime?>(endedAt),
      'timeBudgetSec': serializer.toJson<int?>(timeBudgetSec),
      'targetRir': serializer.toJson<int?>(targetRir),
    };
  }

  Session copyWith(
          {int? id,
          DateTime? startedAt,
          Value<DateTime?> endedAt = const Value.absent(),
          Value<int?> timeBudgetSec = const Value.absent(),
          Value<int?> targetRir = const Value.absent()}) =>
      Session(
        id: id ?? this.id,
        startedAt: startedAt ?? this.startedAt,
        endedAt: endedAt.present ? endedAt.value : this.endedAt,
        timeBudgetSec:
            timeBudgetSec.present ? timeBudgetSec.value : this.timeBudgetSec,
        targetRir: targetRir.present ? targetRir.value : this.targetRir,
      );
  Session copyWithCompanion(SessionsCompanion data) {
    return Session(
      id: data.id.present ? data.id.value : this.id,
      startedAt: data.startedAt.present ? data.startedAt.value : this.startedAt,
      endedAt: data.endedAt.present ? data.endedAt.value : this.endedAt,
      timeBudgetSec: data.timeBudgetSec.present
          ? data.timeBudgetSec.value
          : this.timeBudgetSec,
      targetRir: data.targetRir.present ? data.targetRir.value : this.targetRir,
    );
  }

  @override
  String toString() {
    return (StringBuffer('Session(')
          ..write('id: $id, ')
          ..write('startedAt: $startedAt, ')
          ..write('endedAt: $endedAt, ')
          ..write('timeBudgetSec: $timeBudgetSec, ')
          ..write('targetRir: $targetRir')
          ..write(')'))
        .toString();
  }

  @override
  int get hashCode =>
      Object.hash(id, startedAt, endedAt, timeBudgetSec, targetRir);
  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      (other is Session &&
          other.id == this.id &&
          other.startedAt == this.startedAt &&
          other.endedAt == this.endedAt &&
          other.timeBudgetSec == this.timeBudgetSec &&
          other.targetRir == this.targetRir);
}

class SessionsCompanion extends UpdateCompanion<Session> {
  final Value<int> id;
  final Value<DateTime> startedAt;
  final Value<DateTime?> endedAt;
  final Value<int?> timeBudgetSec;
  final Value<int?> targetRir;
  const SessionsCompanion({
    this.id = const Value.absent(),
    this.startedAt = const Value.absent(),
    this.endedAt = const Value.absent(),
    this.timeBudgetSec = const Value.absent(),
    this.targetRir = const Value.absent(),
  });
  SessionsCompanion.insert({
    this.id = const Value.absent(),
    required DateTime startedAt,
    this.endedAt = const Value.absent(),
    this.timeBudgetSec = const Value.absent(),
    this.targetRir = const Value.absent(),
  }) : startedAt = Value(startedAt);
  static Insertable<Session> custom({
    Expression<int>? id,
    Expression<DateTime>? startedAt,
    Expression<DateTime>? endedAt,
    Expression<int>? timeBudgetSec,
    Expression<int>? targetRir,
  }) {
    return RawValuesInsertable({
      if (id != null) 'id': id,
      if (startedAt != null) 'started_at': startedAt,
      if (endedAt != null) 'ended_at': endedAt,
      if (timeBudgetSec != null) 'time_budget_sec': timeBudgetSec,
      if (targetRir != null) 'target_rir': targetRir,
    });
  }

  SessionsCompanion copyWith(
      {Value<int>? id,
      Value<DateTime>? startedAt,
      Value<DateTime?>? endedAt,
      Value<int?>? timeBudgetSec,
      Value<int?>? targetRir}) {
    return SessionsCompanion(
      id: id ?? this.id,
      startedAt: startedAt ?? this.startedAt,
      endedAt: endedAt ?? this.endedAt,
      timeBudgetSec: timeBudgetSec ?? this.timeBudgetSec,
      targetRir: targetRir ?? this.targetRir,
    );
  }

  @override
  Map<String, Expression> toColumns(bool nullToAbsent) {
    final map = <String, Expression>{};
    if (id.present) {
      map['id'] = Variable<int>(id.value);
    }
    if (startedAt.present) {
      map['started_at'] = Variable<DateTime>(startedAt.value);
    }
    if (endedAt.present) {
      map['ended_at'] = Variable<DateTime>(endedAt.value);
    }
    if (timeBudgetSec.present) {
      map['time_budget_sec'] = Variable<int>(timeBudgetSec.value);
    }
    if (targetRir.present) {
      map['target_rir'] = Variable<int>(targetRir.value);
    }
    return map;
  }

  @override
  String toString() {
    return (StringBuffer('SessionsCompanion(')
          ..write('id: $id, ')
          ..write('startedAt: $startedAt, ')
          ..write('endedAt: $endedAt, ')
          ..write('timeBudgetSec: $timeBudgetSec, ')
          ..write('targetRir: $targetRir')
          ..write(')'))
        .toString();
  }
}

class $CompletedSetsTable extends CompletedSets
    with TableInfo<$CompletedSetsTable, CompletedSet> {
  @override
  final GeneratedDatabase attachedDatabase;
  final String? _alias;
  $CompletedSetsTable(this.attachedDatabase, [this._alias]);
  static const VerificationMeta _idMeta = const VerificationMeta('id');
  @override
  late final GeneratedColumn<int> id = GeneratedColumn<int>(
      'id', aliasedName, false,
      hasAutoIncrement: true,
      type: DriftSqlType.int,
      requiredDuringInsert: false,
      defaultConstraints:
          GeneratedColumn.constraintIsAlways('PRIMARY KEY AUTOINCREMENT'));
  static const VerificationMeta _sessionIdMeta =
      const VerificationMeta('sessionId');
  @override
  late final GeneratedColumn<int> sessionId = GeneratedColumn<int>(
      'session_id', aliasedName, true,
      type: DriftSqlType.int,
      requiredDuringInsert: false,
      defaultConstraints: GeneratedColumn.constraintIsAlways(
          'REFERENCES sessions (id) ON DELETE SET NULL'));
  static const VerificationMeta _exerciseIdMeta =
      const VerificationMeta('exerciseId');
  @override
  late final GeneratedColumn<String> exerciseId = GeneratedColumn<String>(
      'exercise_id', aliasedName, false,
      type: DriftSqlType.string, requiredDuringInsert: true);
  static const VerificationMeta _weightKgMeta =
      const VerificationMeta('weightKg');
  @override
  late final GeneratedColumn<double> weightKg = GeneratedColumn<double>(
      'weight_kg', aliasedName, false,
      type: DriftSqlType.double, requiredDuringInsert: true);
  static const VerificationMeta _repsMeta = const VerificationMeta('reps');
  @override
  late final GeneratedColumn<int> reps = GeneratedColumn<int>(
      'reps', aliasedName, false,
      type: DriftSqlType.int, requiredDuringInsert: true);
  static const VerificationMeta _rirMeta = const VerificationMeta('rir');
  @override
  late final GeneratedColumn<double> rir = GeneratedColumn<double>(
      'rir', aliasedName, false,
      type: DriftSqlType.double, requiredDuringInsert: true);
  static const VerificationMeta _timestampMeta =
      const VerificationMeta('timestamp');
  @override
  late final GeneratedColumn<DateTime> timestamp = GeneratedColumn<DateTime>(
      'timestamp', aliasedName, false,
      type: DriftSqlType.dateTime, requiredDuringInsert: true);
  static const VerificationMeta _anchorsBenchKgMeta =
      const VerificationMeta('anchorsBenchKg');
  @override
  late final GeneratedColumn<double> anchorsBenchKg = GeneratedColumn<double>(
      'anchors_bench_kg', aliasedName, false,
      type: DriftSqlType.double,
      requiredDuringInsert: false,
      defaultValue: const Constant(100.0));
  static const VerificationMeta _anchorsSquatKgMeta =
      const VerificationMeta('anchorsSquatKg');
  @override
  late final GeneratedColumn<double> anchorsSquatKg = GeneratedColumn<double>(
      'anchors_squat_kg', aliasedName, false,
      type: DriftSqlType.double,
      requiredDuringInsert: false,
      defaultValue: const Constant(140.0));
  static const VerificationMeta _anchorsDeadliftKgMeta =
      const VerificationMeta('anchorsDeadliftKg');
  @override
  late final GeneratedColumn<double> anchorsDeadliftKg =
      GeneratedColumn<double>('anchors_deadlift_kg', aliasedName, false,
          type: DriftSqlType.double,
          requiredDuringInsert: false,
          defaultValue: const Constant(180.0));
  @override
  List<GeneratedColumn> get $columns => [
        id,
        sessionId,
        exerciseId,
        weightKg,
        reps,
        rir,
        timestamp,
        anchorsBenchKg,
        anchorsSquatKg,
        anchorsDeadliftKg
      ];
  @override
  String get aliasedName => _alias ?? actualTableName;
  @override
  String get actualTableName => $name;
  static const String $name = 'completed_sets';
  @override
  VerificationContext validateIntegrity(Insertable<CompletedSet> instance,
      {bool isInserting = false}) {
    final context = VerificationContext();
    final data = instance.toColumns(true);
    if (data.containsKey('id')) {
      context.handle(_idMeta, id.isAcceptableOrUnknown(data['id']!, _idMeta));
    }
    if (data.containsKey('session_id')) {
      context.handle(_sessionIdMeta,
          sessionId.isAcceptableOrUnknown(data['session_id']!, _sessionIdMeta));
    }
    if (data.containsKey('exercise_id')) {
      context.handle(
          _exerciseIdMeta,
          exerciseId.isAcceptableOrUnknown(
              data['exercise_id']!, _exerciseIdMeta));
    } else if (isInserting) {
      context.missing(_exerciseIdMeta);
    }
    if (data.containsKey('weight_kg')) {
      context.handle(_weightKgMeta,
          weightKg.isAcceptableOrUnknown(data['weight_kg']!, _weightKgMeta));
    } else if (isInserting) {
      context.missing(_weightKgMeta);
    }
    if (data.containsKey('reps')) {
      context.handle(
          _repsMeta, reps.isAcceptableOrUnknown(data['reps']!, _repsMeta));
    } else if (isInserting) {
      context.missing(_repsMeta);
    }
    if (data.containsKey('rir')) {
      context.handle(
          _rirMeta, rir.isAcceptableOrUnknown(data['rir']!, _rirMeta));
    } else if (isInserting) {
      context.missing(_rirMeta);
    }
    if (data.containsKey('timestamp')) {
      context.handle(_timestampMeta,
          timestamp.isAcceptableOrUnknown(data['timestamp']!, _timestampMeta));
    } else if (isInserting) {
      context.missing(_timestampMeta);
    }
    if (data.containsKey('anchors_bench_kg')) {
      context.handle(
          _anchorsBenchKgMeta,
          anchorsBenchKg.isAcceptableOrUnknown(
              data['anchors_bench_kg']!, _anchorsBenchKgMeta));
    }
    if (data.containsKey('anchors_squat_kg')) {
      context.handle(
          _anchorsSquatKgMeta,
          anchorsSquatKg.isAcceptableOrUnknown(
              data['anchors_squat_kg']!, _anchorsSquatKgMeta));
    }
    if (data.containsKey('anchors_deadlift_kg')) {
      context.handle(
          _anchorsDeadliftKgMeta,
          anchorsDeadliftKg.isAcceptableOrUnknown(
              data['anchors_deadlift_kg']!, _anchorsDeadliftKgMeta));
    }
    return context;
  }

  @override
  Set<GeneratedColumn> get $primaryKey => {id};
  @override
  CompletedSet map(Map<String, dynamic> data, {String? tablePrefix}) {
    final effectivePrefix = tablePrefix != null ? '$tablePrefix.' : '';
    return CompletedSet(
      id: attachedDatabase.typeMapping
          .read(DriftSqlType.int, data['${effectivePrefix}id'])!,
      sessionId: attachedDatabase.typeMapping
          .read(DriftSqlType.int, data['${effectivePrefix}session_id']),
      exerciseId: attachedDatabase.typeMapping
          .read(DriftSqlType.string, data['${effectivePrefix}exercise_id'])!,
      weightKg: attachedDatabase.typeMapping
          .read(DriftSqlType.double, data['${effectivePrefix}weight_kg'])!,
      reps: attachedDatabase.typeMapping
          .read(DriftSqlType.int, data['${effectivePrefix}reps'])!,
      rir: attachedDatabase.typeMapping
          .read(DriftSqlType.double, data['${effectivePrefix}rir'])!,
      timestamp: attachedDatabase.typeMapping
          .read(DriftSqlType.dateTime, data['${effectivePrefix}timestamp'])!,
      anchorsBenchKg: attachedDatabase.typeMapping.read(
          DriftSqlType.double, data['${effectivePrefix}anchors_bench_kg'])!,
      anchorsSquatKg: attachedDatabase.typeMapping.read(
          DriftSqlType.double, data['${effectivePrefix}anchors_squat_kg'])!,
      anchorsDeadliftKg: attachedDatabase.typeMapping.read(
          DriftSqlType.double, data['${effectivePrefix}anchors_deadlift_kg'])!,
    );
  }

  @override
  $CompletedSetsTable createAlias(String alias) {
    return $CompletedSetsTable(attachedDatabase, alias);
  }
}

class CompletedSet extends DataClass implements Insertable<CompletedSet> {
  final int id;
  final int? sessionId;
  final String exerciseId;
  final double weightKg;
  final int reps;
  final double rir;
  final DateTime timestamp;
  final double anchorsBenchKg;
  final double anchorsSquatKg;
  final double anchorsDeadliftKg;
  const CompletedSet(
      {required this.id,
      this.sessionId,
      required this.exerciseId,
      required this.weightKg,
      required this.reps,
      required this.rir,
      required this.timestamp,
      required this.anchorsBenchKg,
      required this.anchorsSquatKg,
      required this.anchorsDeadliftKg});
  @override
  Map<String, Expression> toColumns(bool nullToAbsent) {
    final map = <String, Expression>{};
    map['id'] = Variable<int>(id);
    if (!nullToAbsent || sessionId != null) {
      map['session_id'] = Variable<int>(sessionId);
    }
    map['exercise_id'] = Variable<String>(exerciseId);
    map['weight_kg'] = Variable<double>(weightKg);
    map['reps'] = Variable<int>(reps);
    map['rir'] = Variable<double>(rir);
    map['timestamp'] = Variable<DateTime>(timestamp);
    map['anchors_bench_kg'] = Variable<double>(anchorsBenchKg);
    map['anchors_squat_kg'] = Variable<double>(anchorsSquatKg);
    map['anchors_deadlift_kg'] = Variable<double>(anchorsDeadliftKg);
    return map;
  }

  CompletedSetsCompanion toCompanion(bool nullToAbsent) {
    return CompletedSetsCompanion(
      id: Value(id),
      sessionId: sessionId == null && nullToAbsent
          ? const Value.absent()
          : Value(sessionId),
      exerciseId: Value(exerciseId),
      weightKg: Value(weightKg),
      reps: Value(reps),
      rir: Value(rir),
      timestamp: Value(timestamp),
      anchorsBenchKg: Value(anchorsBenchKg),
      anchorsSquatKg: Value(anchorsSquatKg),
      anchorsDeadliftKg: Value(anchorsDeadliftKg),
    );
  }

  factory CompletedSet.fromJson(Map<String, dynamic> json,
      {ValueSerializer? serializer}) {
    serializer ??= driftRuntimeOptions.defaultSerializer;
    return CompletedSet(
      id: serializer.fromJson<int>(json['id']),
      sessionId: serializer.fromJson<int?>(json['sessionId']),
      exerciseId: serializer.fromJson<String>(json['exerciseId']),
      weightKg: serializer.fromJson<double>(json['weightKg']),
      reps: serializer.fromJson<int>(json['reps']),
      rir: serializer.fromJson<double>(json['rir']),
      timestamp: serializer.fromJson<DateTime>(json['timestamp']),
      anchorsBenchKg: serializer.fromJson<double>(json['anchorsBenchKg']),
      anchorsSquatKg: serializer.fromJson<double>(json['anchorsSquatKg']),
      anchorsDeadliftKg: serializer.fromJson<double>(json['anchorsDeadliftKg']),
    );
  }
  @override
  Map<String, dynamic> toJson({ValueSerializer? serializer}) {
    serializer ??= driftRuntimeOptions.defaultSerializer;
    return <String, dynamic>{
      'id': serializer.toJson<int>(id),
      'sessionId': serializer.toJson<int?>(sessionId),
      'exerciseId': serializer.toJson<String>(exerciseId),
      'weightKg': serializer.toJson<double>(weightKg),
      'reps': serializer.toJson<int>(reps),
      'rir': serializer.toJson<double>(rir),
      'timestamp': serializer.toJson<DateTime>(timestamp),
      'anchorsBenchKg': serializer.toJson<double>(anchorsBenchKg),
      'anchorsSquatKg': serializer.toJson<double>(anchorsSquatKg),
      'anchorsDeadliftKg': serializer.toJson<double>(anchorsDeadliftKg),
    };
  }

  CompletedSet copyWith(
          {int? id,
          Value<int?> sessionId = const Value.absent(),
          String? exerciseId,
          double? weightKg,
          int? reps,
          double? rir,
          DateTime? timestamp,
          double? anchorsBenchKg,
          double? anchorsSquatKg,
          double? anchorsDeadliftKg}) =>
      CompletedSet(
        id: id ?? this.id,
        sessionId: sessionId.present ? sessionId.value : this.sessionId,
        exerciseId: exerciseId ?? this.exerciseId,
        weightKg: weightKg ?? this.weightKg,
        reps: reps ?? this.reps,
        rir: rir ?? this.rir,
        timestamp: timestamp ?? this.timestamp,
        anchorsBenchKg: anchorsBenchKg ?? this.anchorsBenchKg,
        anchorsSquatKg: anchorsSquatKg ?? this.anchorsSquatKg,
        anchorsDeadliftKg: anchorsDeadliftKg ?? this.anchorsDeadliftKg,
      );
  CompletedSet copyWithCompanion(CompletedSetsCompanion data) {
    return CompletedSet(
      id: data.id.present ? data.id.value : this.id,
      sessionId: data.sessionId.present ? data.sessionId.value : this.sessionId,
      exerciseId:
          data.exerciseId.present ? data.exerciseId.value : this.exerciseId,
      weightKg: data.weightKg.present ? data.weightKg.value : this.weightKg,
      reps: data.reps.present ? data.reps.value : this.reps,
      rir: data.rir.present ? data.rir.value : this.rir,
      timestamp: data.timestamp.present ? data.timestamp.value : this.timestamp,
      anchorsBenchKg: data.anchorsBenchKg.present
          ? data.anchorsBenchKg.value
          : this.anchorsBenchKg,
      anchorsSquatKg: data.anchorsSquatKg.present
          ? data.anchorsSquatKg.value
          : this.anchorsSquatKg,
      anchorsDeadliftKg: data.anchorsDeadliftKg.present
          ? data.anchorsDeadliftKg.value
          : this.anchorsDeadliftKg,
    );
  }

  @override
  String toString() {
    return (StringBuffer('CompletedSet(')
          ..write('id: $id, ')
          ..write('sessionId: $sessionId, ')
          ..write('exerciseId: $exerciseId, ')
          ..write('weightKg: $weightKg, ')
          ..write('reps: $reps, ')
          ..write('rir: $rir, ')
          ..write('timestamp: $timestamp, ')
          ..write('anchorsBenchKg: $anchorsBenchKg, ')
          ..write('anchorsSquatKg: $anchorsSquatKg, ')
          ..write('anchorsDeadliftKg: $anchorsDeadliftKg')
          ..write(')'))
        .toString();
  }

  @override
  int get hashCode => Object.hash(id, sessionId, exerciseId, weightKg, reps,
      rir, timestamp, anchorsBenchKg, anchorsSquatKg, anchorsDeadliftKg);
  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      (other is CompletedSet &&
          other.id == this.id &&
          other.sessionId == this.sessionId &&
          other.exerciseId == this.exerciseId &&
          other.weightKg == this.weightKg &&
          other.reps == this.reps &&
          other.rir == this.rir &&
          other.timestamp == this.timestamp &&
          other.anchorsBenchKg == this.anchorsBenchKg &&
          other.anchorsSquatKg == this.anchorsSquatKg &&
          other.anchorsDeadliftKg == this.anchorsDeadliftKg);
}

class CompletedSetsCompanion extends UpdateCompanion<CompletedSet> {
  final Value<int> id;
  final Value<int?> sessionId;
  final Value<String> exerciseId;
  final Value<double> weightKg;
  final Value<int> reps;
  final Value<double> rir;
  final Value<DateTime> timestamp;
  final Value<double> anchorsBenchKg;
  final Value<double> anchorsSquatKg;
  final Value<double> anchorsDeadliftKg;
  const CompletedSetsCompanion({
    this.id = const Value.absent(),
    this.sessionId = const Value.absent(),
    this.exerciseId = const Value.absent(),
    this.weightKg = const Value.absent(),
    this.reps = const Value.absent(),
    this.rir = const Value.absent(),
    this.timestamp = const Value.absent(),
    this.anchorsBenchKg = const Value.absent(),
    this.anchorsSquatKg = const Value.absent(),
    this.anchorsDeadliftKg = const Value.absent(),
  });
  CompletedSetsCompanion.insert({
    this.id = const Value.absent(),
    this.sessionId = const Value.absent(),
    required String exerciseId,
    required double weightKg,
    required int reps,
    required double rir,
    required DateTime timestamp,
    this.anchorsBenchKg = const Value.absent(),
    this.anchorsSquatKg = const Value.absent(),
    this.anchorsDeadliftKg = const Value.absent(),
  })  : exerciseId = Value(exerciseId),
        weightKg = Value(weightKg),
        reps = Value(reps),
        rir = Value(rir),
        timestamp = Value(timestamp);
  static Insertable<CompletedSet> custom({
    Expression<int>? id,
    Expression<int>? sessionId,
    Expression<String>? exerciseId,
    Expression<double>? weightKg,
    Expression<int>? reps,
    Expression<double>? rir,
    Expression<DateTime>? timestamp,
    Expression<double>? anchorsBenchKg,
    Expression<double>? anchorsSquatKg,
    Expression<double>? anchorsDeadliftKg,
  }) {
    return RawValuesInsertable({
      if (id != null) 'id': id,
      if (sessionId != null) 'session_id': sessionId,
      if (exerciseId != null) 'exercise_id': exerciseId,
      if (weightKg != null) 'weight_kg': weightKg,
      if (reps != null) 'reps': reps,
      if (rir != null) 'rir': rir,
      if (timestamp != null) 'timestamp': timestamp,
      if (anchorsBenchKg != null) 'anchors_bench_kg': anchorsBenchKg,
      if (anchorsSquatKg != null) 'anchors_squat_kg': anchorsSquatKg,
      if (anchorsDeadliftKg != null) 'anchors_deadlift_kg': anchorsDeadliftKg,
    });
  }

  CompletedSetsCompanion copyWith(
      {Value<int>? id,
      Value<int?>? sessionId,
      Value<String>? exerciseId,
      Value<double>? weightKg,
      Value<int>? reps,
      Value<double>? rir,
      Value<DateTime>? timestamp,
      Value<double>? anchorsBenchKg,
      Value<double>? anchorsSquatKg,
      Value<double>? anchorsDeadliftKg}) {
    return CompletedSetsCompanion(
      id: id ?? this.id,
      sessionId: sessionId ?? this.sessionId,
      exerciseId: exerciseId ?? this.exerciseId,
      weightKg: weightKg ?? this.weightKg,
      reps: reps ?? this.reps,
      rir: rir ?? this.rir,
      timestamp: timestamp ?? this.timestamp,
      anchorsBenchKg: anchorsBenchKg ?? this.anchorsBenchKg,
      anchorsSquatKg: anchorsSquatKg ?? this.anchorsSquatKg,
      anchorsDeadliftKg: anchorsDeadliftKg ?? this.anchorsDeadliftKg,
    );
  }

  @override
  Map<String, Expression> toColumns(bool nullToAbsent) {
    final map = <String, Expression>{};
    if (id.present) {
      map['id'] = Variable<int>(id.value);
    }
    if (sessionId.present) {
      map['session_id'] = Variable<int>(sessionId.value);
    }
    if (exerciseId.present) {
      map['exercise_id'] = Variable<String>(exerciseId.value);
    }
    if (weightKg.present) {
      map['weight_kg'] = Variable<double>(weightKg.value);
    }
    if (reps.present) {
      map['reps'] = Variable<int>(reps.value);
    }
    if (rir.present) {
      map['rir'] = Variable<double>(rir.value);
    }
    if (timestamp.present) {
      map['timestamp'] = Variable<DateTime>(timestamp.value);
    }
    if (anchorsBenchKg.present) {
      map['anchors_bench_kg'] = Variable<double>(anchorsBenchKg.value);
    }
    if (anchorsSquatKg.present) {
      map['anchors_squat_kg'] = Variable<double>(anchorsSquatKg.value);
    }
    if (anchorsDeadliftKg.present) {
      map['anchors_deadlift_kg'] = Variable<double>(anchorsDeadliftKg.value);
    }
    return map;
  }

  @override
  String toString() {
    return (StringBuffer('CompletedSetsCompanion(')
          ..write('id: $id, ')
          ..write('sessionId: $sessionId, ')
          ..write('exerciseId: $exerciseId, ')
          ..write('weightKg: $weightKg, ')
          ..write('reps: $reps, ')
          ..write('rir: $rir, ')
          ..write('timestamp: $timestamp, ')
          ..write('anchorsBenchKg: $anchorsBenchKg, ')
          ..write('anchorsSquatKg: $anchorsSquatKg, ')
          ..write('anchorsDeadliftKg: $anchorsDeadliftKg')
          ..write(')'))
        .toString();
  }
}

class $DismissedLogTable extends DismissedLog
    with TableInfo<$DismissedLogTable, DismissedLogData> {
  @override
  final GeneratedDatabase attachedDatabase;
  final String? _alias;
  $DismissedLogTable(this.attachedDatabase, [this._alias]);
  static const VerificationMeta _idMeta = const VerificationMeta('id');
  @override
  late final GeneratedColumn<int> id = GeneratedColumn<int>(
      'id', aliasedName, false,
      hasAutoIncrement: true,
      type: DriftSqlType.int,
      requiredDuringInsert: false,
      defaultConstraints:
          GeneratedColumn.constraintIsAlways('PRIMARY KEY AUTOINCREMENT'));
  static const VerificationMeta _sessionIdMeta =
      const VerificationMeta('sessionId');
  @override
  late final GeneratedColumn<int> sessionId = GeneratedColumn<int>(
      'session_id', aliasedName, false,
      type: DriftSqlType.int,
      requiredDuringInsert: true,
      defaultConstraints: GeneratedColumn.constraintIsAlways(
          'REFERENCES sessions (id) ON DELETE CASCADE'));
  static const VerificationMeta _exerciseIdMeta =
      const VerificationMeta('exerciseId');
  @override
  late final GeneratedColumn<String> exerciseId = GeneratedColumn<String>(
      'exercise_id', aliasedName, false,
      type: DriftSqlType.string, requiredDuringInsert: true);
  static const VerificationMeta _timestampMeta =
      const VerificationMeta('timestamp');
  @override
  late final GeneratedColumn<DateTime> timestamp = GeneratedColumn<DateTime>(
      'timestamp', aliasedName, false,
      type: DriftSqlType.dateTime, requiredDuringInsert: true);
  @override
  List<GeneratedColumn> get $columns => [id, sessionId, exerciseId, timestamp];
  @override
  String get aliasedName => _alias ?? actualTableName;
  @override
  String get actualTableName => $name;
  static const String $name = 'dismissed_log';
  @override
  VerificationContext validateIntegrity(Insertable<DismissedLogData> instance,
      {bool isInserting = false}) {
    final context = VerificationContext();
    final data = instance.toColumns(true);
    if (data.containsKey('id')) {
      context.handle(_idMeta, id.isAcceptableOrUnknown(data['id']!, _idMeta));
    }
    if (data.containsKey('session_id')) {
      context.handle(_sessionIdMeta,
          sessionId.isAcceptableOrUnknown(data['session_id']!, _sessionIdMeta));
    } else if (isInserting) {
      context.missing(_sessionIdMeta);
    }
    if (data.containsKey('exercise_id')) {
      context.handle(
          _exerciseIdMeta,
          exerciseId.isAcceptableOrUnknown(
              data['exercise_id']!, _exerciseIdMeta));
    } else if (isInserting) {
      context.missing(_exerciseIdMeta);
    }
    if (data.containsKey('timestamp')) {
      context.handle(_timestampMeta,
          timestamp.isAcceptableOrUnknown(data['timestamp']!, _timestampMeta));
    } else if (isInserting) {
      context.missing(_timestampMeta);
    }
    return context;
  }

  @override
  Set<GeneratedColumn> get $primaryKey => {id};
  @override
  DismissedLogData map(Map<String, dynamic> data, {String? tablePrefix}) {
    final effectivePrefix = tablePrefix != null ? '$tablePrefix.' : '';
    return DismissedLogData(
      id: attachedDatabase.typeMapping
          .read(DriftSqlType.int, data['${effectivePrefix}id'])!,
      sessionId: attachedDatabase.typeMapping
          .read(DriftSqlType.int, data['${effectivePrefix}session_id'])!,
      exerciseId: attachedDatabase.typeMapping
          .read(DriftSqlType.string, data['${effectivePrefix}exercise_id'])!,
      timestamp: attachedDatabase.typeMapping
          .read(DriftSqlType.dateTime, data['${effectivePrefix}timestamp'])!,
    );
  }

  @override
  $DismissedLogTable createAlias(String alias) {
    return $DismissedLogTable(attachedDatabase, alias);
  }
}

class DismissedLogData extends DataClass
    implements Insertable<DismissedLogData> {
  final int id;
  final int sessionId;
  final String exerciseId;
  final DateTime timestamp;
  const DismissedLogData(
      {required this.id,
      required this.sessionId,
      required this.exerciseId,
      required this.timestamp});
  @override
  Map<String, Expression> toColumns(bool nullToAbsent) {
    final map = <String, Expression>{};
    map['id'] = Variable<int>(id);
    map['session_id'] = Variable<int>(sessionId);
    map['exercise_id'] = Variable<String>(exerciseId);
    map['timestamp'] = Variable<DateTime>(timestamp);
    return map;
  }

  DismissedLogCompanion toCompanion(bool nullToAbsent) {
    return DismissedLogCompanion(
      id: Value(id),
      sessionId: Value(sessionId),
      exerciseId: Value(exerciseId),
      timestamp: Value(timestamp),
    );
  }

  factory DismissedLogData.fromJson(Map<String, dynamic> json,
      {ValueSerializer? serializer}) {
    serializer ??= driftRuntimeOptions.defaultSerializer;
    return DismissedLogData(
      id: serializer.fromJson<int>(json['id']),
      sessionId: serializer.fromJson<int>(json['sessionId']),
      exerciseId: serializer.fromJson<String>(json['exerciseId']),
      timestamp: serializer.fromJson<DateTime>(json['timestamp']),
    );
  }
  @override
  Map<String, dynamic> toJson({ValueSerializer? serializer}) {
    serializer ??= driftRuntimeOptions.defaultSerializer;
    return <String, dynamic>{
      'id': serializer.toJson<int>(id),
      'sessionId': serializer.toJson<int>(sessionId),
      'exerciseId': serializer.toJson<String>(exerciseId),
      'timestamp': serializer.toJson<DateTime>(timestamp),
    };
  }

  DismissedLogData copyWith(
          {int? id, int? sessionId, String? exerciseId, DateTime? timestamp}) =>
      DismissedLogData(
        id: id ?? this.id,
        sessionId: sessionId ?? this.sessionId,
        exerciseId: exerciseId ?? this.exerciseId,
        timestamp: timestamp ?? this.timestamp,
      );
  DismissedLogData copyWithCompanion(DismissedLogCompanion data) {
    return DismissedLogData(
      id: data.id.present ? data.id.value : this.id,
      sessionId: data.sessionId.present ? data.sessionId.value : this.sessionId,
      exerciseId:
          data.exerciseId.present ? data.exerciseId.value : this.exerciseId,
      timestamp: data.timestamp.present ? data.timestamp.value : this.timestamp,
    );
  }

  @override
  String toString() {
    return (StringBuffer('DismissedLogData(')
          ..write('id: $id, ')
          ..write('sessionId: $sessionId, ')
          ..write('exerciseId: $exerciseId, ')
          ..write('timestamp: $timestamp')
          ..write(')'))
        .toString();
  }

  @override
  int get hashCode => Object.hash(id, sessionId, exerciseId, timestamp);
  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      (other is DismissedLogData &&
          other.id == this.id &&
          other.sessionId == this.sessionId &&
          other.exerciseId == this.exerciseId &&
          other.timestamp == this.timestamp);
}

class DismissedLogCompanion extends UpdateCompanion<DismissedLogData> {
  final Value<int> id;
  final Value<int> sessionId;
  final Value<String> exerciseId;
  final Value<DateTime> timestamp;
  const DismissedLogCompanion({
    this.id = const Value.absent(),
    this.sessionId = const Value.absent(),
    this.exerciseId = const Value.absent(),
    this.timestamp = const Value.absent(),
  });
  DismissedLogCompanion.insert({
    this.id = const Value.absent(),
    required int sessionId,
    required String exerciseId,
    required DateTime timestamp,
  })  : sessionId = Value(sessionId),
        exerciseId = Value(exerciseId),
        timestamp = Value(timestamp);
  static Insertable<DismissedLogData> custom({
    Expression<int>? id,
    Expression<int>? sessionId,
    Expression<String>? exerciseId,
    Expression<DateTime>? timestamp,
  }) {
    return RawValuesInsertable({
      if (id != null) 'id': id,
      if (sessionId != null) 'session_id': sessionId,
      if (exerciseId != null) 'exercise_id': exerciseId,
      if (timestamp != null) 'timestamp': timestamp,
    });
  }

  DismissedLogCompanion copyWith(
      {Value<int>? id,
      Value<int>? sessionId,
      Value<String>? exerciseId,
      Value<DateTime>? timestamp}) {
    return DismissedLogCompanion(
      id: id ?? this.id,
      sessionId: sessionId ?? this.sessionId,
      exerciseId: exerciseId ?? this.exerciseId,
      timestamp: timestamp ?? this.timestamp,
    );
  }

  @override
  Map<String, Expression> toColumns(bool nullToAbsent) {
    final map = <String, Expression>{};
    if (id.present) {
      map['id'] = Variable<int>(id.value);
    }
    if (sessionId.present) {
      map['session_id'] = Variable<int>(sessionId.value);
    }
    if (exerciseId.present) {
      map['exercise_id'] = Variable<String>(exerciseId.value);
    }
    if (timestamp.present) {
      map['timestamp'] = Variable<DateTime>(timestamp.value);
    }
    return map;
  }

  @override
  String toString() {
    return (StringBuffer('DismissedLogCompanion(')
          ..write('id: $id, ')
          ..write('sessionId: $sessionId, ')
          ..write('exerciseId: $exerciseId, ')
          ..write('timestamp: $timestamp')
          ..write(')'))
        .toString();
  }
}

class $AppSettingsTable extends AppSettings
    with TableInfo<$AppSettingsTable, AppSetting> {
  @override
  final GeneratedDatabase attachedDatabase;
  final String? _alias;
  $AppSettingsTable(this.attachedDatabase, [this._alias]);
  static const VerificationMeta _idMeta = const VerificationMeta('id');
  @override
  late final GeneratedColumn<int> id = GeneratedColumn<int>(
      'id', aliasedName, false,
      type: DriftSqlType.int,
      requiredDuringInsert: false,
      defaultValue: const Constant(0));
  static const VerificationMeta _restBetweenSetsSecMeta =
      const VerificationMeta('restBetweenSetsSec');
  @override
  late final GeneratedColumn<int> restBetweenSetsSec = GeneratedColumn<int>(
      'rest_between_sets_sec', aliasedName, false,
      type: DriftSqlType.int,
      requiredDuringInsert: false,
      defaultValue: const Constant(180));
  static const VerificationMeta _defaultTargetRirMeta =
      const VerificationMeta('defaultTargetRir');
  @override
  late final GeneratedColumn<int> defaultTargetRir = GeneratedColumn<int>(
      'default_target_rir', aliasedName, false,
      type: DriftSqlType.int,
      requiredDuringInsert: false,
      defaultValue: const Constant(2));
  static const VerificationMeta _themeModeMeta =
      const VerificationMeta('themeMode');
  @override
  late final GeneratedColumn<String> themeMode = GeneratedColumn<String>(
      'theme_mode', aliasedName, false,
      type: DriftSqlType.string,
      requiredDuringInsert: false,
      defaultValue: const Constant('system'));
  static const VerificationMeta _userIdMeta = const VerificationMeta('userId');
  @override
  late final GeneratedColumn<String> userId = GeneratedColumn<String>(
      'user_id', aliasedName, false,
      type: DriftSqlType.string,
      requiredDuringInsert: false,
      defaultValue: const Constant(''));
  @override
  List<GeneratedColumn> get $columns =>
      [id, restBetweenSetsSec, defaultTargetRir, themeMode, userId];
  @override
  String get aliasedName => _alias ?? actualTableName;
  @override
  String get actualTableName => $name;
  static const String $name = 'app_settings';
  @override
  VerificationContext validateIntegrity(Insertable<AppSetting> instance,
      {bool isInserting = false}) {
    final context = VerificationContext();
    final data = instance.toColumns(true);
    if (data.containsKey('id')) {
      context.handle(_idMeta, id.isAcceptableOrUnknown(data['id']!, _idMeta));
    }
    if (data.containsKey('rest_between_sets_sec')) {
      context.handle(
          _restBetweenSetsSecMeta,
          restBetweenSetsSec.isAcceptableOrUnknown(
              data['rest_between_sets_sec']!, _restBetweenSetsSecMeta));
    }
    if (data.containsKey('default_target_rir')) {
      context.handle(
          _defaultTargetRirMeta,
          defaultTargetRir.isAcceptableOrUnknown(
              data['default_target_rir']!, _defaultTargetRirMeta));
    }
    if (data.containsKey('theme_mode')) {
      context.handle(_themeModeMeta,
          themeMode.isAcceptableOrUnknown(data['theme_mode']!, _themeModeMeta));
    }
    if (data.containsKey('user_id')) {
      context.handle(_userIdMeta,
          userId.isAcceptableOrUnknown(data['user_id']!, _userIdMeta));
    }
    return context;
  }

  @override
  Set<GeneratedColumn> get $primaryKey => {id};
  @override
  AppSetting map(Map<String, dynamic> data, {String? tablePrefix}) {
    final effectivePrefix = tablePrefix != null ? '$tablePrefix.' : '';
    return AppSetting(
      id: attachedDatabase.typeMapping
          .read(DriftSqlType.int, data['${effectivePrefix}id'])!,
      restBetweenSetsSec: attachedDatabase.typeMapping.read(
          DriftSqlType.int, data['${effectivePrefix}rest_between_sets_sec'])!,
      defaultTargetRir: attachedDatabase.typeMapping.read(
          DriftSqlType.int, data['${effectivePrefix}default_target_rir'])!,
      themeMode: attachedDatabase.typeMapping
          .read(DriftSqlType.string, data['${effectivePrefix}theme_mode'])!,
      userId: attachedDatabase.typeMapping
          .read(DriftSqlType.string, data['${effectivePrefix}user_id'])!,
    );
  }

  @override
  $AppSettingsTable createAlias(String alias) {
    return $AppSettingsTable(attachedDatabase, alias);
  }
}

class AppSetting extends DataClass implements Insertable<AppSetting> {
  final int id;
  final int restBetweenSetsSec;
  final int defaultTargetRir;
  final String themeMode;
  final String userId;
  const AppSetting(
      {required this.id,
      required this.restBetweenSetsSec,
      required this.defaultTargetRir,
      required this.themeMode,
      required this.userId});
  @override
  Map<String, Expression> toColumns(bool nullToAbsent) {
    final map = <String, Expression>{};
    map['id'] = Variable<int>(id);
    map['rest_between_sets_sec'] = Variable<int>(restBetweenSetsSec);
    map['default_target_rir'] = Variable<int>(defaultTargetRir);
    map['theme_mode'] = Variable<String>(themeMode);
    map['user_id'] = Variable<String>(userId);
    return map;
  }

  AppSettingsCompanion toCompanion(bool nullToAbsent) {
    return AppSettingsCompanion(
      id: Value(id),
      restBetweenSetsSec: Value(restBetweenSetsSec),
      defaultTargetRir: Value(defaultTargetRir),
      themeMode: Value(themeMode),
      userId: Value(userId),
    );
  }

  factory AppSetting.fromJson(Map<String, dynamic> json,
      {ValueSerializer? serializer}) {
    serializer ??= driftRuntimeOptions.defaultSerializer;
    return AppSetting(
      id: serializer.fromJson<int>(json['id']),
      restBetweenSetsSec: serializer.fromJson<int>(json['restBetweenSetsSec']),
      defaultTargetRir: serializer.fromJson<int>(json['defaultTargetRir']),
      themeMode: serializer.fromJson<String>(json['themeMode']),
      userId: serializer.fromJson<String>(json['userId']),
    );
  }
  @override
  Map<String, dynamic> toJson({ValueSerializer? serializer}) {
    serializer ??= driftRuntimeOptions.defaultSerializer;
    return <String, dynamic>{
      'id': serializer.toJson<int>(id),
      'restBetweenSetsSec': serializer.toJson<int>(restBetweenSetsSec),
      'defaultTargetRir': serializer.toJson<int>(defaultTargetRir),
      'themeMode': serializer.toJson<String>(themeMode),
      'userId': serializer.toJson<String>(userId),
    };
  }

  AppSetting copyWith(
          {int? id,
          int? restBetweenSetsSec,
          int? defaultTargetRir,
          String? themeMode,
          String? userId}) =>
      AppSetting(
        id: id ?? this.id,
        restBetweenSetsSec: restBetweenSetsSec ?? this.restBetweenSetsSec,
        defaultTargetRir: defaultTargetRir ?? this.defaultTargetRir,
        themeMode: themeMode ?? this.themeMode,
        userId: userId ?? this.userId,
      );
  AppSetting copyWithCompanion(AppSettingsCompanion data) {
    return AppSetting(
      id: data.id.present ? data.id.value : this.id,
      restBetweenSetsSec: data.restBetweenSetsSec.present
          ? data.restBetweenSetsSec.value
          : this.restBetweenSetsSec,
      defaultTargetRir: data.defaultTargetRir.present
          ? data.defaultTargetRir.value
          : this.defaultTargetRir,
      themeMode: data.themeMode.present ? data.themeMode.value : this.themeMode,
      userId: data.userId.present ? data.userId.value : this.userId,
    );
  }

  @override
  String toString() {
    return (StringBuffer('AppSetting(')
          ..write('id: $id, ')
          ..write('restBetweenSetsSec: $restBetweenSetsSec, ')
          ..write('defaultTargetRir: $defaultTargetRir, ')
          ..write('themeMode: $themeMode, ')
          ..write('userId: $userId')
          ..write(')'))
        .toString();
  }

  @override
  int get hashCode =>
      Object.hash(id, restBetweenSetsSec, defaultTargetRir, themeMode, userId);
  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      (other is AppSetting &&
          other.id == this.id &&
          other.restBetweenSetsSec == this.restBetweenSetsSec &&
          other.defaultTargetRir == this.defaultTargetRir &&
          other.themeMode == this.themeMode &&
          other.userId == this.userId);
}

class AppSettingsCompanion extends UpdateCompanion<AppSetting> {
  final Value<int> id;
  final Value<int> restBetweenSetsSec;
  final Value<int> defaultTargetRir;
  final Value<String> themeMode;
  final Value<String> userId;
  const AppSettingsCompanion({
    this.id = const Value.absent(),
    this.restBetweenSetsSec = const Value.absent(),
    this.defaultTargetRir = const Value.absent(),
    this.themeMode = const Value.absent(),
    this.userId = const Value.absent(),
  });
  AppSettingsCompanion.insert({
    this.id = const Value.absent(),
    this.restBetweenSetsSec = const Value.absent(),
    this.defaultTargetRir = const Value.absent(),
    this.themeMode = const Value.absent(),
    this.userId = const Value.absent(),
  });
  static Insertable<AppSetting> custom({
    Expression<int>? id,
    Expression<int>? restBetweenSetsSec,
    Expression<int>? defaultTargetRir,
    Expression<String>? themeMode,
    Expression<String>? userId,
  }) {
    return RawValuesInsertable({
      if (id != null) 'id': id,
      if (restBetweenSetsSec != null)
        'rest_between_sets_sec': restBetweenSetsSec,
      if (defaultTargetRir != null) 'default_target_rir': defaultTargetRir,
      if (themeMode != null) 'theme_mode': themeMode,
      if (userId != null) 'user_id': userId,
    });
  }

  AppSettingsCompanion copyWith(
      {Value<int>? id,
      Value<int>? restBetweenSetsSec,
      Value<int>? defaultTargetRir,
      Value<String>? themeMode,
      Value<String>? userId}) {
    return AppSettingsCompanion(
      id: id ?? this.id,
      restBetweenSetsSec: restBetweenSetsSec ?? this.restBetweenSetsSec,
      defaultTargetRir: defaultTargetRir ?? this.defaultTargetRir,
      themeMode: themeMode ?? this.themeMode,
      userId: userId ?? this.userId,
    );
  }

  @override
  Map<String, Expression> toColumns(bool nullToAbsent) {
    final map = <String, Expression>{};
    if (id.present) {
      map['id'] = Variable<int>(id.value);
    }
    if (restBetweenSetsSec.present) {
      map['rest_between_sets_sec'] = Variable<int>(restBetweenSetsSec.value);
    }
    if (defaultTargetRir.present) {
      map['default_target_rir'] = Variable<int>(defaultTargetRir.value);
    }
    if (themeMode.present) {
      map['theme_mode'] = Variable<String>(themeMode.value);
    }
    if (userId.present) {
      map['user_id'] = Variable<String>(userId.value);
    }
    return map;
  }

  @override
  String toString() {
    return (StringBuffer('AppSettingsCompanion(')
          ..write('id: $id, ')
          ..write('restBetweenSetsSec: $restBetweenSetsSec, ')
          ..write('defaultTargetRir: $defaultTargetRir, ')
          ..write('themeMode: $themeMode, ')
          ..write('userId: $userId')
          ..write(')'))
        .toString();
  }
}

abstract class _$AppDatabase extends GeneratedDatabase {
  _$AppDatabase(QueryExecutor e) : super(e);
  $AppDatabaseManager get managers => $AppDatabaseManager(this);
  late final $AnchorsTable anchors = $AnchorsTable(this);
  late final $SessionsTable sessions = $SessionsTable(this);
  late final $CompletedSetsTable completedSets = $CompletedSetsTable(this);
  late final $DismissedLogTable dismissedLog = $DismissedLogTable(this);
  late final $AppSettingsTable appSettings = $AppSettingsTable(this);
  @override
  Iterable<TableInfo<Table, Object?>> get allTables =>
      allSchemaEntities.whereType<TableInfo<Table, Object?>>();
  @override
  List<DatabaseSchemaEntity> get allSchemaEntities =>
      [anchors, sessions, completedSets, dismissedLog, appSettings];
  @override
  StreamQueryUpdateRules get streamUpdateRules => const StreamQueryUpdateRules(
        [
          WritePropagation(
            on: TableUpdateQuery.onTableName('sessions',
                limitUpdateKind: UpdateKind.delete),
            result: [
              TableUpdate('completed_sets', kind: UpdateKind.update),
            ],
          ),
          WritePropagation(
            on: TableUpdateQuery.onTableName('sessions',
                limitUpdateKind: UpdateKind.delete),
            result: [
              TableUpdate('dismissed_log', kind: UpdateKind.delete),
            ],
          ),
        ],
      );
}

typedef $$AnchorsTableCreateCompanionBuilder = AnchorsCompanion Function({
  Value<int> id,
  required double benchPressKg,
  required double squatKg,
  required double deadliftKg,
  required DateTime updatedAt,
});
typedef $$AnchorsTableUpdateCompanionBuilder = AnchorsCompanion Function({
  Value<int> id,
  Value<double> benchPressKg,
  Value<double> squatKg,
  Value<double> deadliftKg,
  Value<DateTime> updatedAt,
});

class $$AnchorsTableFilterComposer
    extends FilterComposer<_$AppDatabase, $AnchorsTable> {
  $$AnchorsTableFilterComposer(super.$state);
  ColumnFilters<int> get id => $state.composableBuilder(
      column: $state.table.id,
      builder: (column, joinBuilders) =>
          ColumnFilters(column, joinBuilders: joinBuilders));

  ColumnFilters<double> get benchPressKg => $state.composableBuilder(
      column: $state.table.benchPressKg,
      builder: (column, joinBuilders) =>
          ColumnFilters(column, joinBuilders: joinBuilders));

  ColumnFilters<double> get squatKg => $state.composableBuilder(
      column: $state.table.squatKg,
      builder: (column, joinBuilders) =>
          ColumnFilters(column, joinBuilders: joinBuilders));

  ColumnFilters<double> get deadliftKg => $state.composableBuilder(
      column: $state.table.deadliftKg,
      builder: (column, joinBuilders) =>
          ColumnFilters(column, joinBuilders: joinBuilders));

  ColumnFilters<DateTime> get updatedAt => $state.composableBuilder(
      column: $state.table.updatedAt,
      builder: (column, joinBuilders) =>
          ColumnFilters(column, joinBuilders: joinBuilders));
}

class $$AnchorsTableOrderingComposer
    extends OrderingComposer<_$AppDatabase, $AnchorsTable> {
  $$AnchorsTableOrderingComposer(super.$state);
  ColumnOrderings<int> get id => $state.composableBuilder(
      column: $state.table.id,
      builder: (column, joinBuilders) =>
          ColumnOrderings(column, joinBuilders: joinBuilders));

  ColumnOrderings<double> get benchPressKg => $state.composableBuilder(
      column: $state.table.benchPressKg,
      builder: (column, joinBuilders) =>
          ColumnOrderings(column, joinBuilders: joinBuilders));

  ColumnOrderings<double> get squatKg => $state.composableBuilder(
      column: $state.table.squatKg,
      builder: (column, joinBuilders) =>
          ColumnOrderings(column, joinBuilders: joinBuilders));

  ColumnOrderings<double> get deadliftKg => $state.composableBuilder(
      column: $state.table.deadliftKg,
      builder: (column, joinBuilders) =>
          ColumnOrderings(column, joinBuilders: joinBuilders));

  ColumnOrderings<DateTime> get updatedAt => $state.composableBuilder(
      column: $state.table.updatedAt,
      builder: (column, joinBuilders) =>
          ColumnOrderings(column, joinBuilders: joinBuilders));
}

class $$AnchorsTableTableManager extends RootTableManager<
    _$AppDatabase,
    $AnchorsTable,
    Anchor,
    $$AnchorsTableFilterComposer,
    $$AnchorsTableOrderingComposer,
    $$AnchorsTableCreateCompanionBuilder,
    $$AnchorsTableUpdateCompanionBuilder,
    (Anchor, BaseReferences<_$AppDatabase, $AnchorsTable, Anchor>),
    Anchor,
    PrefetchHooks Function()> {
  $$AnchorsTableTableManager(_$AppDatabase db, $AnchorsTable table)
      : super(TableManagerState(
          db: db,
          table: table,
          filteringComposer:
              $$AnchorsTableFilterComposer(ComposerState(db, table)),
          orderingComposer:
              $$AnchorsTableOrderingComposer(ComposerState(db, table)),
          updateCompanionCallback: ({
            Value<int> id = const Value.absent(),
            Value<double> benchPressKg = const Value.absent(),
            Value<double> squatKg = const Value.absent(),
            Value<double> deadliftKg = const Value.absent(),
            Value<DateTime> updatedAt = const Value.absent(),
          }) =>
              AnchorsCompanion(
            id: id,
            benchPressKg: benchPressKg,
            squatKg: squatKg,
            deadliftKg: deadliftKg,
            updatedAt: updatedAt,
          ),
          createCompanionCallback: ({
            Value<int> id = const Value.absent(),
            required double benchPressKg,
            required double squatKg,
            required double deadliftKg,
            required DateTime updatedAt,
          }) =>
              AnchorsCompanion.insert(
            id: id,
            benchPressKg: benchPressKg,
            squatKg: squatKg,
            deadliftKg: deadliftKg,
            updatedAt: updatedAt,
          ),
          withReferenceMapper: (p0) => p0
              .map((e) => (e.readTable(table), BaseReferences(db, table, e)))
              .toList(),
          prefetchHooksCallback: null,
        ));
}

typedef $$AnchorsTableProcessedTableManager = ProcessedTableManager<
    _$AppDatabase,
    $AnchorsTable,
    Anchor,
    $$AnchorsTableFilterComposer,
    $$AnchorsTableOrderingComposer,
    $$AnchorsTableCreateCompanionBuilder,
    $$AnchorsTableUpdateCompanionBuilder,
    (Anchor, BaseReferences<_$AppDatabase, $AnchorsTable, Anchor>),
    Anchor,
    PrefetchHooks Function()>;
typedef $$SessionsTableCreateCompanionBuilder = SessionsCompanion Function({
  Value<int> id,
  required DateTime startedAt,
  Value<DateTime?> endedAt,
  Value<int?> timeBudgetSec,
  Value<int?> targetRir,
});
typedef $$SessionsTableUpdateCompanionBuilder = SessionsCompanion Function({
  Value<int> id,
  Value<DateTime> startedAt,
  Value<DateTime?> endedAt,
  Value<int?> timeBudgetSec,
  Value<int?> targetRir,
});

final class $$SessionsTableReferences
    extends BaseReferences<_$AppDatabase, $SessionsTable, Session> {
  $$SessionsTableReferences(super.$_db, super.$_table, super.$_typedResult);

  static MultiTypedResultKey<$CompletedSetsTable, List<CompletedSet>>
      _completedSetsRefsTable(_$AppDatabase db) =>
          MultiTypedResultKey.fromTable(db.completedSets,
              aliasName: $_aliasNameGenerator(
                  db.sessions.id, db.completedSets.sessionId));

  $$CompletedSetsTableProcessedTableManager get completedSetsRefs {
    final manager = $$CompletedSetsTableTableManager($_db, $_db.completedSets)
        .filter((f) => f.sessionId.id($_item.id));

    final cache = $_typedResult.readTableOrNull(_completedSetsRefsTable($_db));
    return ProcessedTableManager(
        manager.$state.copyWith(prefetchedData: cache));
  }

  static MultiTypedResultKey<$DismissedLogTable, List<DismissedLogData>>
      _dismissedLogRefsTable(_$AppDatabase db) => MultiTypedResultKey.fromTable(
          db.dismissedLog,
          aliasName:
              $_aliasNameGenerator(db.sessions.id, db.dismissedLog.sessionId));

  $$DismissedLogTableProcessedTableManager get dismissedLogRefs {
    final manager = $$DismissedLogTableTableManager($_db, $_db.dismissedLog)
        .filter((f) => f.sessionId.id($_item.id));

    final cache = $_typedResult.readTableOrNull(_dismissedLogRefsTable($_db));
    return ProcessedTableManager(
        manager.$state.copyWith(prefetchedData: cache));
  }
}

class $$SessionsTableFilterComposer
    extends FilterComposer<_$AppDatabase, $SessionsTable> {
  $$SessionsTableFilterComposer(super.$state);
  ColumnFilters<int> get id => $state.composableBuilder(
      column: $state.table.id,
      builder: (column, joinBuilders) =>
          ColumnFilters(column, joinBuilders: joinBuilders));

  ColumnFilters<DateTime> get startedAt => $state.composableBuilder(
      column: $state.table.startedAt,
      builder: (column, joinBuilders) =>
          ColumnFilters(column, joinBuilders: joinBuilders));

  ColumnFilters<DateTime> get endedAt => $state.composableBuilder(
      column: $state.table.endedAt,
      builder: (column, joinBuilders) =>
          ColumnFilters(column, joinBuilders: joinBuilders));

  ColumnFilters<int> get timeBudgetSec => $state.composableBuilder(
      column: $state.table.timeBudgetSec,
      builder: (column, joinBuilders) =>
          ColumnFilters(column, joinBuilders: joinBuilders));

  ColumnFilters<int> get targetRir => $state.composableBuilder(
      column: $state.table.targetRir,
      builder: (column, joinBuilders) =>
          ColumnFilters(column, joinBuilders: joinBuilders));

  ComposableFilter completedSetsRefs(
      ComposableFilter Function($$CompletedSetsTableFilterComposer f) f) {
    final $$CompletedSetsTableFilterComposer composer = $state.composerBuilder(
        composer: this,
        getCurrentColumn: (t) => t.id,
        referencedTable: $state.db.completedSets,
        getReferencedColumn: (t) => t.sessionId,
        builder: (joinBuilder, parentComposers) =>
            $$CompletedSetsTableFilterComposer(ComposerState($state.db,
                $state.db.completedSets, joinBuilder, parentComposers)));
    return f(composer);
  }

  ComposableFilter dismissedLogRefs(
      ComposableFilter Function($$DismissedLogTableFilterComposer f) f) {
    final $$DismissedLogTableFilterComposer composer = $state.composerBuilder(
        composer: this,
        getCurrentColumn: (t) => t.id,
        referencedTable: $state.db.dismissedLog,
        getReferencedColumn: (t) => t.sessionId,
        builder: (joinBuilder, parentComposers) =>
            $$DismissedLogTableFilterComposer(ComposerState($state.db,
                $state.db.dismissedLog, joinBuilder, parentComposers)));
    return f(composer);
  }
}

class $$SessionsTableOrderingComposer
    extends OrderingComposer<_$AppDatabase, $SessionsTable> {
  $$SessionsTableOrderingComposer(super.$state);
  ColumnOrderings<int> get id => $state.composableBuilder(
      column: $state.table.id,
      builder: (column, joinBuilders) =>
          ColumnOrderings(column, joinBuilders: joinBuilders));

  ColumnOrderings<DateTime> get startedAt => $state.composableBuilder(
      column: $state.table.startedAt,
      builder: (column, joinBuilders) =>
          ColumnOrderings(column, joinBuilders: joinBuilders));

  ColumnOrderings<DateTime> get endedAt => $state.composableBuilder(
      column: $state.table.endedAt,
      builder: (column, joinBuilders) =>
          ColumnOrderings(column, joinBuilders: joinBuilders));

  ColumnOrderings<int> get timeBudgetSec => $state.composableBuilder(
      column: $state.table.timeBudgetSec,
      builder: (column, joinBuilders) =>
          ColumnOrderings(column, joinBuilders: joinBuilders));

  ColumnOrderings<int> get targetRir => $state.composableBuilder(
      column: $state.table.targetRir,
      builder: (column, joinBuilders) =>
          ColumnOrderings(column, joinBuilders: joinBuilders));
}

class $$SessionsTableTableManager extends RootTableManager<
    _$AppDatabase,
    $SessionsTable,
    Session,
    $$SessionsTableFilterComposer,
    $$SessionsTableOrderingComposer,
    $$SessionsTableCreateCompanionBuilder,
    $$SessionsTableUpdateCompanionBuilder,
    (Session, $$SessionsTableReferences),
    Session,
    PrefetchHooks Function({bool completedSetsRefs, bool dismissedLogRefs})> {
  $$SessionsTableTableManager(_$AppDatabase db, $SessionsTable table)
      : super(TableManagerState(
          db: db,
          table: table,
          filteringComposer:
              $$SessionsTableFilterComposer(ComposerState(db, table)),
          orderingComposer:
              $$SessionsTableOrderingComposer(ComposerState(db, table)),
          updateCompanionCallback: ({
            Value<int> id = const Value.absent(),
            Value<DateTime> startedAt = const Value.absent(),
            Value<DateTime?> endedAt = const Value.absent(),
            Value<int?> timeBudgetSec = const Value.absent(),
            Value<int?> targetRir = const Value.absent(),
          }) =>
              SessionsCompanion(
            id: id,
            startedAt: startedAt,
            endedAt: endedAt,
            timeBudgetSec: timeBudgetSec,
            targetRir: targetRir,
          ),
          createCompanionCallback: ({
            Value<int> id = const Value.absent(),
            required DateTime startedAt,
            Value<DateTime?> endedAt = const Value.absent(),
            Value<int?> timeBudgetSec = const Value.absent(),
            Value<int?> targetRir = const Value.absent(),
          }) =>
              SessionsCompanion.insert(
            id: id,
            startedAt: startedAt,
            endedAt: endedAt,
            timeBudgetSec: timeBudgetSec,
            targetRir: targetRir,
          ),
          withReferenceMapper: (p0) => p0
              .map((e) =>
                  (e.readTable(table), $$SessionsTableReferences(db, table, e)))
              .toList(),
          prefetchHooksCallback: (
              {completedSetsRefs = false, dismissedLogRefs = false}) {
            return PrefetchHooks(
              db: db,
              explicitlyWatchedTables: [
                if (completedSetsRefs) db.completedSets,
                if (dismissedLogRefs) db.dismissedLog
              ],
              addJoins: null,
              getPrefetchedDataCallback: (items) async {
                return [
                  if (completedSetsRefs)
                    await $_getPrefetchedData(
                        currentTable: table,
                        referencedTable: $$SessionsTableReferences
                            ._completedSetsRefsTable(db),
                        managerFromTypedResult: (p0) =>
                            $$SessionsTableReferences(db, table, p0)
                                .completedSetsRefs,
                        referencedItemsForCurrentItem:
                            (item, referencedItems) => referencedItems
                                .where((e) => e.sessionId == item.id),
                        typedResults: items),
                  if (dismissedLogRefs)
                    await $_getPrefetchedData(
                        currentTable: table,
                        referencedTable: $$SessionsTableReferences
                            ._dismissedLogRefsTable(db),
                        managerFromTypedResult: (p0) =>
                            $$SessionsTableReferences(db, table, p0)
                                .dismissedLogRefs,
                        referencedItemsForCurrentItem:
                            (item, referencedItems) => referencedItems
                                .where((e) => e.sessionId == item.id),
                        typedResults: items)
                ];
              },
            );
          },
        ));
}

typedef $$SessionsTableProcessedTableManager = ProcessedTableManager<
    _$AppDatabase,
    $SessionsTable,
    Session,
    $$SessionsTableFilterComposer,
    $$SessionsTableOrderingComposer,
    $$SessionsTableCreateCompanionBuilder,
    $$SessionsTableUpdateCompanionBuilder,
    (Session, $$SessionsTableReferences),
    Session,
    PrefetchHooks Function({bool completedSetsRefs, bool dismissedLogRefs})>;
typedef $$CompletedSetsTableCreateCompanionBuilder = CompletedSetsCompanion
    Function({
  Value<int> id,
  Value<int?> sessionId,
  required String exerciseId,
  required double weightKg,
  required int reps,
  required double rir,
  required DateTime timestamp,
  Value<double> anchorsBenchKg,
  Value<double> anchorsSquatKg,
  Value<double> anchorsDeadliftKg,
});
typedef $$CompletedSetsTableUpdateCompanionBuilder = CompletedSetsCompanion
    Function({
  Value<int> id,
  Value<int?> sessionId,
  Value<String> exerciseId,
  Value<double> weightKg,
  Value<int> reps,
  Value<double> rir,
  Value<DateTime> timestamp,
  Value<double> anchorsBenchKg,
  Value<double> anchorsSquatKg,
  Value<double> anchorsDeadliftKg,
});

final class $$CompletedSetsTableReferences
    extends BaseReferences<_$AppDatabase, $CompletedSetsTable, CompletedSet> {
  $$CompletedSetsTableReferences(
      super.$_db, super.$_table, super.$_typedResult);

  static $SessionsTable _sessionIdTable(_$AppDatabase db) =>
      db.sessions.createAlias(
          $_aliasNameGenerator(db.completedSets.sessionId, db.sessions.id));

  $$SessionsTableProcessedTableManager? get sessionId {
    if ($_item.sessionId == null) return null;
    final manager = $$SessionsTableTableManager($_db, $_db.sessions)
        .filter((f) => f.id($_item.sessionId!));
    final item = $_typedResult.readTableOrNull(_sessionIdTable($_db));
    if (item == null) return manager;
    return ProcessedTableManager(
        manager.$state.copyWith(prefetchedData: [item]));
  }
}

class $$CompletedSetsTableFilterComposer
    extends FilterComposer<_$AppDatabase, $CompletedSetsTable> {
  $$CompletedSetsTableFilterComposer(super.$state);
  ColumnFilters<int> get id => $state.composableBuilder(
      column: $state.table.id,
      builder: (column, joinBuilders) =>
          ColumnFilters(column, joinBuilders: joinBuilders));

  ColumnFilters<String> get exerciseId => $state.composableBuilder(
      column: $state.table.exerciseId,
      builder: (column, joinBuilders) =>
          ColumnFilters(column, joinBuilders: joinBuilders));

  ColumnFilters<double> get weightKg => $state.composableBuilder(
      column: $state.table.weightKg,
      builder: (column, joinBuilders) =>
          ColumnFilters(column, joinBuilders: joinBuilders));

  ColumnFilters<int> get reps => $state.composableBuilder(
      column: $state.table.reps,
      builder: (column, joinBuilders) =>
          ColumnFilters(column, joinBuilders: joinBuilders));

  ColumnFilters<double> get rir => $state.composableBuilder(
      column: $state.table.rir,
      builder: (column, joinBuilders) =>
          ColumnFilters(column, joinBuilders: joinBuilders));

  ColumnFilters<DateTime> get timestamp => $state.composableBuilder(
      column: $state.table.timestamp,
      builder: (column, joinBuilders) =>
          ColumnFilters(column, joinBuilders: joinBuilders));

  ColumnFilters<double> get anchorsBenchKg => $state.composableBuilder(
      column: $state.table.anchorsBenchKg,
      builder: (column, joinBuilders) =>
          ColumnFilters(column, joinBuilders: joinBuilders));

  ColumnFilters<double> get anchorsSquatKg => $state.composableBuilder(
      column: $state.table.anchorsSquatKg,
      builder: (column, joinBuilders) =>
          ColumnFilters(column, joinBuilders: joinBuilders));

  ColumnFilters<double> get anchorsDeadliftKg => $state.composableBuilder(
      column: $state.table.anchorsDeadliftKg,
      builder: (column, joinBuilders) =>
          ColumnFilters(column, joinBuilders: joinBuilders));

  $$SessionsTableFilterComposer get sessionId {
    final $$SessionsTableFilterComposer composer = $state.composerBuilder(
        composer: this,
        getCurrentColumn: (t) => t.sessionId,
        referencedTable: $state.db.sessions,
        getReferencedColumn: (t) => t.id,
        builder: (joinBuilder, parentComposers) =>
            $$SessionsTableFilterComposer(ComposerState(
                $state.db, $state.db.sessions, joinBuilder, parentComposers)));
    return composer;
  }
}

class $$CompletedSetsTableOrderingComposer
    extends OrderingComposer<_$AppDatabase, $CompletedSetsTable> {
  $$CompletedSetsTableOrderingComposer(super.$state);
  ColumnOrderings<int> get id => $state.composableBuilder(
      column: $state.table.id,
      builder: (column, joinBuilders) =>
          ColumnOrderings(column, joinBuilders: joinBuilders));

  ColumnOrderings<String> get exerciseId => $state.composableBuilder(
      column: $state.table.exerciseId,
      builder: (column, joinBuilders) =>
          ColumnOrderings(column, joinBuilders: joinBuilders));

  ColumnOrderings<double> get weightKg => $state.composableBuilder(
      column: $state.table.weightKg,
      builder: (column, joinBuilders) =>
          ColumnOrderings(column, joinBuilders: joinBuilders));

  ColumnOrderings<int> get reps => $state.composableBuilder(
      column: $state.table.reps,
      builder: (column, joinBuilders) =>
          ColumnOrderings(column, joinBuilders: joinBuilders));

  ColumnOrderings<double> get rir => $state.composableBuilder(
      column: $state.table.rir,
      builder: (column, joinBuilders) =>
          ColumnOrderings(column, joinBuilders: joinBuilders));

  ColumnOrderings<DateTime> get timestamp => $state.composableBuilder(
      column: $state.table.timestamp,
      builder: (column, joinBuilders) =>
          ColumnOrderings(column, joinBuilders: joinBuilders));

  ColumnOrderings<double> get anchorsBenchKg => $state.composableBuilder(
      column: $state.table.anchorsBenchKg,
      builder: (column, joinBuilders) =>
          ColumnOrderings(column, joinBuilders: joinBuilders));

  ColumnOrderings<double> get anchorsSquatKg => $state.composableBuilder(
      column: $state.table.anchorsSquatKg,
      builder: (column, joinBuilders) =>
          ColumnOrderings(column, joinBuilders: joinBuilders));

  ColumnOrderings<double> get anchorsDeadliftKg => $state.composableBuilder(
      column: $state.table.anchorsDeadliftKg,
      builder: (column, joinBuilders) =>
          ColumnOrderings(column, joinBuilders: joinBuilders));

  $$SessionsTableOrderingComposer get sessionId {
    final $$SessionsTableOrderingComposer composer = $state.composerBuilder(
        composer: this,
        getCurrentColumn: (t) => t.sessionId,
        referencedTable: $state.db.sessions,
        getReferencedColumn: (t) => t.id,
        builder: (joinBuilder, parentComposers) =>
            $$SessionsTableOrderingComposer(ComposerState(
                $state.db, $state.db.sessions, joinBuilder, parentComposers)));
    return composer;
  }
}

class $$CompletedSetsTableTableManager extends RootTableManager<
    _$AppDatabase,
    $CompletedSetsTable,
    CompletedSet,
    $$CompletedSetsTableFilterComposer,
    $$CompletedSetsTableOrderingComposer,
    $$CompletedSetsTableCreateCompanionBuilder,
    $$CompletedSetsTableUpdateCompanionBuilder,
    (CompletedSet, $$CompletedSetsTableReferences),
    CompletedSet,
    PrefetchHooks Function({bool sessionId})> {
  $$CompletedSetsTableTableManager(_$AppDatabase db, $CompletedSetsTable table)
      : super(TableManagerState(
          db: db,
          table: table,
          filteringComposer:
              $$CompletedSetsTableFilterComposer(ComposerState(db, table)),
          orderingComposer:
              $$CompletedSetsTableOrderingComposer(ComposerState(db, table)),
          updateCompanionCallback: ({
            Value<int> id = const Value.absent(),
            Value<int?> sessionId = const Value.absent(),
            Value<String> exerciseId = const Value.absent(),
            Value<double> weightKg = const Value.absent(),
            Value<int> reps = const Value.absent(),
            Value<double> rir = const Value.absent(),
            Value<DateTime> timestamp = const Value.absent(),
            Value<double> anchorsBenchKg = const Value.absent(),
            Value<double> anchorsSquatKg = const Value.absent(),
            Value<double> anchorsDeadliftKg = const Value.absent(),
          }) =>
              CompletedSetsCompanion(
            id: id,
            sessionId: sessionId,
            exerciseId: exerciseId,
            weightKg: weightKg,
            reps: reps,
            rir: rir,
            timestamp: timestamp,
            anchorsBenchKg: anchorsBenchKg,
            anchorsSquatKg: anchorsSquatKg,
            anchorsDeadliftKg: anchorsDeadliftKg,
          ),
          createCompanionCallback: ({
            Value<int> id = const Value.absent(),
            Value<int?> sessionId = const Value.absent(),
            required String exerciseId,
            required double weightKg,
            required int reps,
            required double rir,
            required DateTime timestamp,
            Value<double> anchorsBenchKg = const Value.absent(),
            Value<double> anchorsSquatKg = const Value.absent(),
            Value<double> anchorsDeadliftKg = const Value.absent(),
          }) =>
              CompletedSetsCompanion.insert(
            id: id,
            sessionId: sessionId,
            exerciseId: exerciseId,
            weightKg: weightKg,
            reps: reps,
            rir: rir,
            timestamp: timestamp,
            anchorsBenchKg: anchorsBenchKg,
            anchorsSquatKg: anchorsSquatKg,
            anchorsDeadliftKg: anchorsDeadliftKg,
          ),
          withReferenceMapper: (p0) => p0
              .map((e) => (
                    e.readTable(table),
                    $$CompletedSetsTableReferences(db, table, e)
                  ))
              .toList(),
          prefetchHooksCallback: ({sessionId = false}) {
            return PrefetchHooks(
              db: db,
              explicitlyWatchedTables: [],
              addJoins: <
                  T extends TableManagerState<
                      dynamic,
                      dynamic,
                      dynamic,
                      dynamic,
                      dynamic,
                      dynamic,
                      dynamic,
                      dynamic,
                      dynamic,
                      dynamic>>(state) {
                if (sessionId) {
                  state = state.withJoin(
                    currentTable: table,
                    currentColumn: table.sessionId,
                    referencedTable:
                        $$CompletedSetsTableReferences._sessionIdTable(db),
                    referencedColumn:
                        $$CompletedSetsTableReferences._sessionIdTable(db).id,
                  ) as T;
                }

                return state;
              },
              getPrefetchedDataCallback: (items) async {
                return [];
              },
            );
          },
        ));
}

typedef $$CompletedSetsTableProcessedTableManager = ProcessedTableManager<
    _$AppDatabase,
    $CompletedSetsTable,
    CompletedSet,
    $$CompletedSetsTableFilterComposer,
    $$CompletedSetsTableOrderingComposer,
    $$CompletedSetsTableCreateCompanionBuilder,
    $$CompletedSetsTableUpdateCompanionBuilder,
    (CompletedSet, $$CompletedSetsTableReferences),
    CompletedSet,
    PrefetchHooks Function({bool sessionId})>;
typedef $$DismissedLogTableCreateCompanionBuilder = DismissedLogCompanion
    Function({
  Value<int> id,
  required int sessionId,
  required String exerciseId,
  required DateTime timestamp,
});
typedef $$DismissedLogTableUpdateCompanionBuilder = DismissedLogCompanion
    Function({
  Value<int> id,
  Value<int> sessionId,
  Value<String> exerciseId,
  Value<DateTime> timestamp,
});

final class $$DismissedLogTableReferences extends BaseReferences<_$AppDatabase,
    $DismissedLogTable, DismissedLogData> {
  $$DismissedLogTableReferences(super.$_db, super.$_table, super.$_typedResult);

  static $SessionsTable _sessionIdTable(_$AppDatabase db) =>
      db.sessions.createAlias(
          $_aliasNameGenerator(db.dismissedLog.sessionId, db.sessions.id));

  $$SessionsTableProcessedTableManager? get sessionId {
    if ($_item.sessionId == null) return null;
    final manager = $$SessionsTableTableManager($_db, $_db.sessions)
        .filter((f) => f.id($_item.sessionId!));
    final item = $_typedResult.readTableOrNull(_sessionIdTable($_db));
    if (item == null) return manager;
    return ProcessedTableManager(
        manager.$state.copyWith(prefetchedData: [item]));
  }
}

class $$DismissedLogTableFilterComposer
    extends FilterComposer<_$AppDatabase, $DismissedLogTable> {
  $$DismissedLogTableFilterComposer(super.$state);
  ColumnFilters<int> get id => $state.composableBuilder(
      column: $state.table.id,
      builder: (column, joinBuilders) =>
          ColumnFilters(column, joinBuilders: joinBuilders));

  ColumnFilters<String> get exerciseId => $state.composableBuilder(
      column: $state.table.exerciseId,
      builder: (column, joinBuilders) =>
          ColumnFilters(column, joinBuilders: joinBuilders));

  ColumnFilters<DateTime> get timestamp => $state.composableBuilder(
      column: $state.table.timestamp,
      builder: (column, joinBuilders) =>
          ColumnFilters(column, joinBuilders: joinBuilders));

  $$SessionsTableFilterComposer get sessionId {
    final $$SessionsTableFilterComposer composer = $state.composerBuilder(
        composer: this,
        getCurrentColumn: (t) => t.sessionId,
        referencedTable: $state.db.sessions,
        getReferencedColumn: (t) => t.id,
        builder: (joinBuilder, parentComposers) =>
            $$SessionsTableFilterComposer(ComposerState(
                $state.db, $state.db.sessions, joinBuilder, parentComposers)));
    return composer;
  }
}

class $$DismissedLogTableOrderingComposer
    extends OrderingComposer<_$AppDatabase, $DismissedLogTable> {
  $$DismissedLogTableOrderingComposer(super.$state);
  ColumnOrderings<int> get id => $state.composableBuilder(
      column: $state.table.id,
      builder: (column, joinBuilders) =>
          ColumnOrderings(column, joinBuilders: joinBuilders));

  ColumnOrderings<String> get exerciseId => $state.composableBuilder(
      column: $state.table.exerciseId,
      builder: (column, joinBuilders) =>
          ColumnOrderings(column, joinBuilders: joinBuilders));

  ColumnOrderings<DateTime> get timestamp => $state.composableBuilder(
      column: $state.table.timestamp,
      builder: (column, joinBuilders) =>
          ColumnOrderings(column, joinBuilders: joinBuilders));

  $$SessionsTableOrderingComposer get sessionId {
    final $$SessionsTableOrderingComposer composer = $state.composerBuilder(
        composer: this,
        getCurrentColumn: (t) => t.sessionId,
        referencedTable: $state.db.sessions,
        getReferencedColumn: (t) => t.id,
        builder: (joinBuilder, parentComposers) =>
            $$SessionsTableOrderingComposer(ComposerState(
                $state.db, $state.db.sessions, joinBuilder, parentComposers)));
    return composer;
  }
}

class $$DismissedLogTableTableManager extends RootTableManager<
    _$AppDatabase,
    $DismissedLogTable,
    DismissedLogData,
    $$DismissedLogTableFilterComposer,
    $$DismissedLogTableOrderingComposer,
    $$DismissedLogTableCreateCompanionBuilder,
    $$DismissedLogTableUpdateCompanionBuilder,
    (DismissedLogData, $$DismissedLogTableReferences),
    DismissedLogData,
    PrefetchHooks Function({bool sessionId})> {
  $$DismissedLogTableTableManager(_$AppDatabase db, $DismissedLogTable table)
      : super(TableManagerState(
          db: db,
          table: table,
          filteringComposer:
              $$DismissedLogTableFilterComposer(ComposerState(db, table)),
          orderingComposer:
              $$DismissedLogTableOrderingComposer(ComposerState(db, table)),
          updateCompanionCallback: ({
            Value<int> id = const Value.absent(),
            Value<int> sessionId = const Value.absent(),
            Value<String> exerciseId = const Value.absent(),
            Value<DateTime> timestamp = const Value.absent(),
          }) =>
              DismissedLogCompanion(
            id: id,
            sessionId: sessionId,
            exerciseId: exerciseId,
            timestamp: timestamp,
          ),
          createCompanionCallback: ({
            Value<int> id = const Value.absent(),
            required int sessionId,
            required String exerciseId,
            required DateTime timestamp,
          }) =>
              DismissedLogCompanion.insert(
            id: id,
            sessionId: sessionId,
            exerciseId: exerciseId,
            timestamp: timestamp,
          ),
          withReferenceMapper: (p0) => p0
              .map((e) => (
                    e.readTable(table),
                    $$DismissedLogTableReferences(db, table, e)
                  ))
              .toList(),
          prefetchHooksCallback: ({sessionId = false}) {
            return PrefetchHooks(
              db: db,
              explicitlyWatchedTables: [],
              addJoins: <
                  T extends TableManagerState<
                      dynamic,
                      dynamic,
                      dynamic,
                      dynamic,
                      dynamic,
                      dynamic,
                      dynamic,
                      dynamic,
                      dynamic,
                      dynamic>>(state) {
                if (sessionId) {
                  state = state.withJoin(
                    currentTable: table,
                    currentColumn: table.sessionId,
                    referencedTable:
                        $$DismissedLogTableReferences._sessionIdTable(db),
                    referencedColumn:
                        $$DismissedLogTableReferences._sessionIdTable(db).id,
                  ) as T;
                }

                return state;
              },
              getPrefetchedDataCallback: (items) async {
                return [];
              },
            );
          },
        ));
}

typedef $$DismissedLogTableProcessedTableManager = ProcessedTableManager<
    _$AppDatabase,
    $DismissedLogTable,
    DismissedLogData,
    $$DismissedLogTableFilterComposer,
    $$DismissedLogTableOrderingComposer,
    $$DismissedLogTableCreateCompanionBuilder,
    $$DismissedLogTableUpdateCompanionBuilder,
    (DismissedLogData, $$DismissedLogTableReferences),
    DismissedLogData,
    PrefetchHooks Function({bool sessionId})>;
typedef $$AppSettingsTableCreateCompanionBuilder = AppSettingsCompanion
    Function({
  Value<int> id,
  Value<int> restBetweenSetsSec,
  Value<int> defaultTargetRir,
  Value<String> themeMode,
  Value<String> userId,
});
typedef $$AppSettingsTableUpdateCompanionBuilder = AppSettingsCompanion
    Function({
  Value<int> id,
  Value<int> restBetweenSetsSec,
  Value<int> defaultTargetRir,
  Value<String> themeMode,
  Value<String> userId,
});

class $$AppSettingsTableFilterComposer
    extends FilterComposer<_$AppDatabase, $AppSettingsTable> {
  $$AppSettingsTableFilterComposer(super.$state);
  ColumnFilters<int> get id => $state.composableBuilder(
      column: $state.table.id,
      builder: (column, joinBuilders) =>
          ColumnFilters(column, joinBuilders: joinBuilders));

  ColumnFilters<int> get restBetweenSetsSec => $state.composableBuilder(
      column: $state.table.restBetweenSetsSec,
      builder: (column, joinBuilders) =>
          ColumnFilters(column, joinBuilders: joinBuilders));

  ColumnFilters<int> get defaultTargetRir => $state.composableBuilder(
      column: $state.table.defaultTargetRir,
      builder: (column, joinBuilders) =>
          ColumnFilters(column, joinBuilders: joinBuilders));

  ColumnFilters<String> get themeMode => $state.composableBuilder(
      column: $state.table.themeMode,
      builder: (column, joinBuilders) =>
          ColumnFilters(column, joinBuilders: joinBuilders));

  ColumnFilters<String> get userId => $state.composableBuilder(
      column: $state.table.userId,
      builder: (column, joinBuilders) =>
          ColumnFilters(column, joinBuilders: joinBuilders));
}

class $$AppSettingsTableOrderingComposer
    extends OrderingComposer<_$AppDatabase, $AppSettingsTable> {
  $$AppSettingsTableOrderingComposer(super.$state);
  ColumnOrderings<int> get id => $state.composableBuilder(
      column: $state.table.id,
      builder: (column, joinBuilders) =>
          ColumnOrderings(column, joinBuilders: joinBuilders));

  ColumnOrderings<int> get restBetweenSetsSec => $state.composableBuilder(
      column: $state.table.restBetweenSetsSec,
      builder: (column, joinBuilders) =>
          ColumnOrderings(column, joinBuilders: joinBuilders));

  ColumnOrderings<int> get defaultTargetRir => $state.composableBuilder(
      column: $state.table.defaultTargetRir,
      builder: (column, joinBuilders) =>
          ColumnOrderings(column, joinBuilders: joinBuilders));

  ColumnOrderings<String> get themeMode => $state.composableBuilder(
      column: $state.table.themeMode,
      builder: (column, joinBuilders) =>
          ColumnOrderings(column, joinBuilders: joinBuilders));

  ColumnOrderings<String> get userId => $state.composableBuilder(
      column: $state.table.userId,
      builder: (column, joinBuilders) =>
          ColumnOrderings(column, joinBuilders: joinBuilders));
}

class $$AppSettingsTableTableManager extends RootTableManager<
    _$AppDatabase,
    $AppSettingsTable,
    AppSetting,
    $$AppSettingsTableFilterComposer,
    $$AppSettingsTableOrderingComposer,
    $$AppSettingsTableCreateCompanionBuilder,
    $$AppSettingsTableUpdateCompanionBuilder,
    (AppSetting, BaseReferences<_$AppDatabase, $AppSettingsTable, AppSetting>),
    AppSetting,
    PrefetchHooks Function()> {
  $$AppSettingsTableTableManager(_$AppDatabase db, $AppSettingsTable table)
      : super(TableManagerState(
          db: db,
          table: table,
          filteringComposer:
              $$AppSettingsTableFilterComposer(ComposerState(db, table)),
          orderingComposer:
              $$AppSettingsTableOrderingComposer(ComposerState(db, table)),
          updateCompanionCallback: ({
            Value<int> id = const Value.absent(),
            Value<int> restBetweenSetsSec = const Value.absent(),
            Value<int> defaultTargetRir = const Value.absent(),
            Value<String> themeMode = const Value.absent(),
            Value<String> userId = const Value.absent(),
          }) =>
              AppSettingsCompanion(
            id: id,
            restBetweenSetsSec: restBetweenSetsSec,
            defaultTargetRir: defaultTargetRir,
            themeMode: themeMode,
            userId: userId,
          ),
          createCompanionCallback: ({
            Value<int> id = const Value.absent(),
            Value<int> restBetweenSetsSec = const Value.absent(),
            Value<int> defaultTargetRir = const Value.absent(),
            Value<String> themeMode = const Value.absent(),
            Value<String> userId = const Value.absent(),
          }) =>
              AppSettingsCompanion.insert(
            id: id,
            restBetweenSetsSec: restBetweenSetsSec,
            defaultTargetRir: defaultTargetRir,
            themeMode: themeMode,
            userId: userId,
          ),
          withReferenceMapper: (p0) => p0
              .map((e) => (e.readTable(table), BaseReferences(db, table, e)))
              .toList(),
          prefetchHooksCallback: null,
        ));
}

typedef $$AppSettingsTableProcessedTableManager = ProcessedTableManager<
    _$AppDatabase,
    $AppSettingsTable,
    AppSetting,
    $$AppSettingsTableFilterComposer,
    $$AppSettingsTableOrderingComposer,
    $$AppSettingsTableCreateCompanionBuilder,
    $$AppSettingsTableUpdateCompanionBuilder,
    (AppSetting, BaseReferences<_$AppDatabase, $AppSettingsTable, AppSetting>),
    AppSetting,
    PrefetchHooks Function()>;

class $AppDatabaseManager {
  final _$AppDatabase _db;
  $AppDatabaseManager(this._db);
  $$AnchorsTableTableManager get anchors =>
      $$AnchorsTableTableManager(_db, _db.anchors);
  $$SessionsTableTableManager get sessions =>
      $$SessionsTableTableManager(_db, _db.sessions);
  $$CompletedSetsTableTableManager get completedSets =>
      $$CompletedSetsTableTableManager(_db, _db.completedSets);
  $$DismissedLogTableTableManager get dismissedLog =>
      $$DismissedLogTableTableManager(_db, _db.dismissedLog);
  $$AppSettingsTableTableManager get appSettings =>
      $$AppSettingsTableTableManager(_db, _db.appSettings);
}
