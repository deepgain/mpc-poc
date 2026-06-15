/// Onboarding — for each of bench / squat / deadlift the user dials a recent
/// working set (weight + reps 1-5) on two scroll wheels; we estimate the 1RM
/// with the Epley formula and write those to [Anchors]. Most people know their
/// working weights, not their true max, so we never ask for a 1RM directly.
/// "I don't know" keeps the old defaults (100 / 140 / 180 kg).
///
/// Designed for <60s to first plan generation.
library;

import 'package:deepgain_app/state/providers.dart';
import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

/// Epley 1RM estimate. At reps == 1 this returns the weight unchanged.
/// Rounded to the nearest 0.5 kg so anchors stay tidy.
double estimate1rm(double weightKg, int reps) {
  final raw = weightKg * (1 + reps / 30.0);
  return (raw / 0.5).round() * 0.5;
}

class OnboardingScreen extends ConsumerStatefulWidget {
  const OnboardingScreen({super.key});

  @override
  ConsumerState<OnboardingScreen> createState() => _OnboardingScreenState();
}

class _OnboardingScreenState extends ConsumerState<OnboardingScreen> {
  // Defaults used when "I don't know" is on — same numbers as before.
  static const _defaultBench = 100.0;
  static const _defaultSquat = 140.0;
  static const _defaultDeadlift = 180.0;

  // Starting working sets, chosen so the estimated 1RMs land near the old
  // defaults (~99 / 140 / 181 kg at 5 reps).
  double _bench1rm = estimate1rm(85, 5);
  double _squat1rm = estimate1rm(120, 5);
  double _deadlift1rm = estimate1rm(155, 5);

  bool _dontKnow = false;
  bool _saving = false;

  Future<void> _start() async {
    setState(() => _saving = true);
    final db = ref.read(databaseProvider);
    final bench = _dontKnow ? _defaultBench : _bench1rm;
    final squat = _dontKnow ? _defaultSquat : _squat1rm;
    final dead = _dontKnow ? _defaultDeadlift : _deadlift1rm;
    try {
      await db.setAnchors(benchKg: bench, squatKg: squat, deadliftKg: dead);

      // Verify the write actually committed — without this it's hard to tell
      // whether a stuck gate is "drift didn't write" vs "drift didn't notify".
      final readback = await db.getAnchors();
      if (readback == null) {
        throw StateError('setAnchors completed but row is missing');
      }

      // Refresh the gate. anchorsProvider is a FutureProvider so .future
      // returns the next computed value; awaiting guarantees the gate has
      // data BEFORE we clear the spinner, so the user never sees a brief
      // stale onboarding screen.
      ref.invalidate(anchorsProvider);
      await ref.read(anchorsProvider.future);
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text('Could not save: $e')));
    } finally {
      if (mounted) setState(() => _saving = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final scheme = theme.colorScheme;
    return Scaffold(
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.fromLTRB(24, 24, 24, 24),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              const SizedBox(height: 12),
              Text(
                'Welcome to DeepGain',
                style: theme.textTheme.headlineSmall?.copyWith(
                  fontWeight: FontWeight.w700,
                  color: scheme.primary,
                ),
              ),
              const SizedBox(height: 6),
              Text(
                'Dial in a recent set for each lift',
                style: theme.textTheme.bodyMedium?.copyWith(
                  color: scheme.onSurfaceVariant,
                ),
              ),
              const SizedBox(height: 22),
              Opacity(
                opacity: _dontKnow ? 0.4 : 1.0,
                child: AbsorbPointer(
                  absorbing: _dontKnow,
                  child: Column(
                    children: [
                      _LiftEstimatorRow(
                        label: 'Bench press',
                        initialWeight: 85,
                        initialReps: 5,
                        onChanged: (v) => setState(() => _bench1rm = v),
                      ),
                      const SizedBox(height: 12),
                      _LiftEstimatorRow(
                        label: 'Squat',
                        initialWeight: 120,
                        initialReps: 5,
                        onChanged: (v) => setState(() => _squat1rm = v),
                      ),
                      const SizedBox(height: 12),
                      _LiftEstimatorRow(
                        label: 'Deadlift',
                        initialWeight: 155,
                        initialReps: 5,
                        onChanged: (v) => setState(() => _deadlift1rm = v),
                      ),
                    ],
                  ),
                ),
              ),
              const SizedBox(height: 16),
              SwitchListTile.adaptive(
                contentPadding: EdgeInsets.zero,
                value: _dontKnow,
                onChanged: (v) => setState(() => _dontKnow = v),
                title: Text(
                  "I don't know my numbers",
                  style: theme.textTheme.bodyLarge,
                ),
                subtitle: Text(
                  'Use defaults, you can update later',
                  style: theme.textTheme.bodySmall,
                ),
              ),
              const Spacer(),
              const SizedBox(height: 20),
              FilledButton(
                onPressed: _saving ? null : _start,
                style: FilledButton.styleFrom(
                  minimumSize: const Size.fromHeight(56),
                  textStyle: theme.textTheme.titleMedium,
                ),
                child: _saving
                    ? const SizedBox(
                        height: 24,
                        width: 24,
                        child: CircularProgressIndicator(strokeWidth: 2),
                      )
                    : const Text('Start'),
              ),
              const SizedBox(height: 12),
              Text(
                'Your data stays on this device. No account needed.',
                textAlign: TextAlign.center,
                style: theme.textTheme.bodySmall?.copyWith(
                  color: scheme.onSurfaceVariant,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

/// One lift: weight wheel (20-300 kg, 2.5 kg steps) + reps wheel (1-5) with a
/// live estimated-1RM readout. Owns its own scroll state so the parent can
/// rebuild on every change without snapping the wheels back.
class _LiftEstimatorRow extends StatefulWidget {
  final String label;
  final double initialWeight;
  final int initialReps;
  final ValueChanged<double> onChanged;

  const _LiftEstimatorRow({
    required this.label,
    required this.initialWeight,
    required this.initialReps,
    required this.onChanged,
  });

  @override
  State<_LiftEstimatorRow> createState() => _LiftEstimatorRowState();
}

class _LiftEstimatorRowState extends State<_LiftEstimatorRow> {
  static const double _wMin = 20;
  static const double _wMax = 300;
  static const double _wStep = 2.5;
  static const _reps = <int>[1, 2, 3, 4, 5];

  static final List<double> _weights = [
    for (double w = _wMin; w <= _wMax; w += _wStep) w,
  ];

  late int _wIndex;
  late int _rIndex;
  late final FixedExtentScrollController _wCtl;
  late final FixedExtentScrollController _rCtl;

  @override
  void initState() {
    super.initState();
    _wIndex = ((widget.initialWeight - _wMin) / _wStep).round().clamp(
      0,
      _weights.length - 1,
    );
    _rIndex = _reps.indexOf(widget.initialReps).clamp(0, _reps.length - 1);
    _wCtl = FixedExtentScrollController(initialItem: _wIndex);
    _rCtl = FixedExtentScrollController(initialItem: _rIndex);
  }

  @override
  void dispose() {
    _wCtl.dispose();
    _rCtl.dispose();
    super.dispose();
  }

  double get _weight => _weights[_wIndex];
  int get _repsValue => _reps[_rIndex];
  double get _oneRm => estimate1rm(_weight, _repsValue);

  void _notify() => widget.onChanged(_oneRm);

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final scheme = theme.colorScheme;
    return Container(
      padding: const EdgeInsets.fromLTRB(16, 8, 16, 8),
      decoration: BoxDecoration(
        color: scheme.surfaceContainerHighest.withValues(alpha: 0.4),
        borderRadius: BorderRadius.circular(16),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(widget.label, style: theme.textTheme.titleSmall),
              Text(
                '≈ ${_oneRm.toStringAsFixed(0)} kg 1RM',
                style: theme.textTheme.titleSmall?.copyWith(
                  color: scheme.primary,
                  fontWeight: FontWeight.w700,
                ),
              ),
            ],
          ),
          const SizedBox(height: 2),
          Row(
            crossAxisAlignment: CrossAxisAlignment.end,
            children: [
              Expanded(
                flex: 3,
                child: _WheelColumn(
                  caption: 'Weight (kg)',
                  controller: _wCtl,
                  labels: [
                    for (final w in _weights)
                      w == w.roundToDouble()
                          ? w.toStringAsFixed(0)
                          : w.toStringAsFixed(1),
                  ],
                  onSelected: (i) {
                    setState(() => _wIndex = i);
                    _notify();
                  },
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                flex: 2,
                child: _WheelColumn(
                  caption: 'Reps',
                  controller: _rCtl,
                  labels: [for (final r in _reps) '$r'],
                  onSelected: (i) {
                    setState(() => _rIndex = i);
                    _notify();
                  },
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

/// A captioned vertical scroll wheel (góra-dół). Bounded height so the
/// CupertinoPicker has room to lay out its items.
class _WheelColumn extends StatelessWidget {
  final String caption;
  final List<String> labels;
  final FixedExtentScrollController controller;
  final ValueChanged<int> onSelected;

  const _WheelColumn({
    required this.caption,
    required this.labels,
    required this.controller,
    required this.onSelected,
  });

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Column(
      children: [
        Text(
          caption,
          style: theme.textTheme.labelSmall?.copyWith(
            color: theme.colorScheme.onSurfaceVariant,
          ),
        ),
        const SizedBox(height: 2),
        SizedBox(
          height: 74,
          child: CupertinoPicker(
            scrollController: controller,
            itemExtent: 28,
            magnification: 1.1,
            squeeze: 1.1,
            useMagnifier: true,
            selectionOverlay: CupertinoPickerDefaultSelectionOverlay(
              background: theme.colorScheme.primary.withValues(alpha: 0.08),
            ),
            onSelectedItemChanged: onSelected,
            children: [
              for (final l in labels)
                Center(
                  child: Text(
                    l,
                    style: theme.textTheme.titleMedium?.copyWith(
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                ),
            ],
          ),
        ),
      ],
    );
  }
}
