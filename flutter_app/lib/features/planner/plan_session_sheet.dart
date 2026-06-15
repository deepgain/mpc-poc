/// Plan-a-session bottom sheet: time / intensity / muscle-group exclusion
/// chips, then "Start workout" runs KnapsackPlanner.plan and goes straight
/// into the live training screen — the user never sees the full plan upfront.
library;

import 'dart:async';

import 'package:deepgain_app/features/training/live_training_screen.dart';
import 'package:deepgain_app/state/providers.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

const _timeOptions = <int>[15, 30, 45, 60, 90]; // minutes
const _rirOptions = <int>[1, 2, 3, 4, 5];

/// Preset muscle-group exclusion chips. Each maps to a list of exercise IDs
/// from `models/exercise_muscle_order.yaml` / `planner_meta.json`.
const _muscleGroupExclusions = <String, List<String>>{
  'No legs': [
    'squat', 'low_bar_squat', 'high_bar_squat',
    'leg_press', 'bulgarian_split_squat',
    'leg_curl', 'leg_extension', 'rdl',
  ],
  'No push': [
    'bench_press', 'incline_bench', 'incline_bench_45',
    'ohp', 'close_grip_bench', 'spoto_press', 'decline_bench',
    'dips', 'chest_press_machine', 'dumbbell_flyes',
  ],
  'No pull': [
    'pendlay_row', 'seal_row', 'lat_pulldown', 'pull_up',
    'reverse_fly',
  ],
  'No deadlift': ['deadlift', 'sumo_deadlift', 'rdl'],
};

class PlanSessionSheet extends ConsumerStatefulWidget {
  const PlanSessionSheet({super.key});

  @override
  ConsumerState<PlanSessionSheet> createState() => _PlanSessionSheetState();
}

class _PlanSessionSheetState extends ConsumerState<PlanSessionSheet> {
  int _timeMin = 60;
  int _targetRir = 2;
  final Set<String> _activeGroups = {};

  bool _generating = false;

  Set<String> _resolvedExclusions() {
    return {
      for (final g in _activeGroups) ..._muscleGroupExclusions[g]!,
    };
  }

  Future<void> _start() async {
    setState(() => _generating = true);
    try {
      final planner = await ref.read(plannerProvider.future);
      final history = ref.read(historyProvider).value ?? const [];
      final plan = planner.plan(
        userHistory: history,
        timeBudgetSec: _timeMin * 60,
        targetRir: _targetRir,
        exclusions: _resolvedExclusions(),
      );
      if (!mounted) return;
      Navigator.of(context).pop(); // close sheet
      // Skip the plan-review screen — go straight into the live workout.
      unawaited(Navigator.of(context).push(
        MaterialPageRoute<void>(
          builder: (_) => LiveTrainingScreen(
            initialPlan: plan,
            timeBudgetSec: _timeMin * 60,
            targetRir: _targetRir,
          ),
        ),
      ));
    } finally {
      if (mounted) setState(() => _generating = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final mq = MediaQuery.of(context);
    // Bottom padding sums:
    //   - viewInsets.bottom (keyboard)
    //   - padding.bottom (Android system gesture nav, when not consumed by SafeArea)
    //   - 16 visual gap
    // viewPadding (raw) wouldn't work with `useSafeArea: true` on the modal —
    // SafeArea would consume it but our padding would double-add.
    final bottomPad = mq.viewInsets.bottom + mq.padding.bottom + 16;
    return SingleChildScrollView(
      // Always scrollable so the Generate button is reachable on small
      // screens / phones with chunky gesture navs even when the chip rows
      // wrap to multiple lines.
      padding: EdgeInsets.only(left: 16, right: 16, top: 8, bottom: bottomPad),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Text('Plan a session', style: theme.textTheme.headlineSmall),
          const SizedBox(height: 24),
          _ChipGroup<int>(
            label: 'Time',
            value: _timeMin,
            options: _timeOptions,
            labelOf: (v) => '$v min',
            onChanged: (v) => setState(() => _timeMin = v),
          ),
          const SizedBox(height: 16),
          _ChipGroup<int>(
            label: 'Intensity (target RIR)',
            value: _targetRir,
            options: _rirOptions,
            labelOf: _rirLabel,
            onChanged: (v) => setState(() => _targetRir = v),
          ),
          const SizedBox(height: 16),
          Text('Skip muscle groups',
              style: theme.textTheme.titleSmall?.copyWith(
                color: theme.colorScheme.onSurfaceVariant,
              )),
          const SizedBox(height: 8),
          Wrap(
            spacing: 8,
            runSpacing: 8,
            children: [
              for (final g in _muscleGroupExclusions.keys)
                FilterChip(
                  label: Text(g),
                  selected: _activeGroups.contains(g),
                  onSelected: (sel) => setState(() {
                    if (sel) {
                      _activeGroups.add(g);
                    } else {
                      _activeGroups.remove(g);
                    }
                  }),
                ),
            ],
          ),
          const SizedBox(height: 24),
          FilledButton(
            onPressed: _generating ? null : _start,
            style: FilledButton.styleFrom(
              minimumSize: const Size.fromHeight(56),
              textStyle: theme.textTheme.titleMedium,
            ),
            child: _generating
                ? const SizedBox(
                    height: 24, width: 24,
                    child: CircularProgressIndicator(strokeWidth: 2),
                  )
                : const Text('Start workout'),
          ),
        ],
      ),
    );
  }

  static String _rirLabel(int rir) {
    return switch (rir) {
      1 => '1 (very hard)',
      2 => '2 (hard)',
      3 => '3 (normal)',
      4 => '4 (easy)',
      5 => '5 (very easy)',
      _ => '$rir',
    };
  }
}

class _ChipGroup<T> extends StatelessWidget {
  final String label;
  final T value;
  final List<T> options;
  final String Function(T) labelOf;
  final ValueChanged<T> onChanged;

  const _ChipGroup({
    required this.label,
    required this.value,
    required this.options,
    required this.labelOf,
    required this.onChanged,
  });

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(label,
            style: theme.textTheme.titleSmall?.copyWith(
              color: theme.colorScheme.onSurfaceVariant,
            )),
        const SizedBox(height: 8),
        Wrap(
          spacing: 8,
          runSpacing: 8,
          children: [
            for (final o in options)
              ChoiceChip(
                label: Text(labelOf(o)),
                selected: o == value,
                onSelected: (_) => onChanged(o),
              ),
          ],
        ),
      ],
    );
  }
}

