/// Live training screen — the killer flow.
///
/// State machine per block:
///   - Set 1 of N → user taps Done/Adjust → set written to db, RIR confirmed.
///   - Rest timer (3 min default) starts. UI shows countdown.
///   - When timer hits 0 (or user taps Skip rest), advance to next set.
///   - After last set of a block, advance to first set of next block.
///
/// Dismiss flow: drops the active block AND any further uses of that exercise,
/// then re-runs KnapsackPlanner.plan with:
///   - userHistory = original session-history + everything completed so far
///   - exclusions  = original exclusions + dismissed exercise ids
///   - timeBudgetSec = remaining time
/// The remaining-blocks list is replaced atomically.
///
/// On end-session: writes session row, runs updateStrengthAnchors with the
/// completed sets, persists new anchors.
library;

import 'dart:async';

import 'package:deepgain_app/inference/types.dart';
import 'package:deepgain_app/planner/exercise_block.dart';
import 'package:deepgain_app/state/providers.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

class LiveTrainingScreen extends ConsumerStatefulWidget {
  final KnapsackPlan initialPlan;
  final int timeBudgetSec;
  final int targetRir;

  const LiveTrainingScreen({
    super.key,
    required this.initialPlan,
    required this.timeBudgetSec,
    required this.targetRir,
  });

  @override
  ConsumerState<LiveTrainingScreen> createState() => _LiveTrainingScreenState();
}

class _LiveTrainingScreenState extends ConsumerState<LiveTrainingScreen> {
  /// Remaining blocks (current at index 0). Mutated when sets complete or
  /// after a dismiss/replan.
  late List<ExerciseBlock> _blocks;

  /// 1-based set index within the current block (1..setsCount).
  int _setInBlock = 1;

  /// Blocks fully completed so far this session — used for the "Block X of Y"
  /// header. Survives dismiss/replan: after a replan, current = _blocksCompleted+1
  /// and total = _blocksCompleted + _blocks.length.
  int _blocksCompleted = 0;

  /// Sets completed so far this session (for db + anchor update + replan history).
  final List<WorkoutSet> _completedSets = [];

  /// Exercise ids the user has dismissed during this session — passed as
  /// exclusions on every replan so the planner doesn't re-suggest them.
  final Set<String> _dismissed = {};

  /// Wall-clock at session start, used to compute remaining time budget.
  late final DateTime _sessionStartedAt;

  /// Timer for rest period after a completed set. Null when not resting.
  Timer? _restTimer;
  int _restSecondsLeft = 0;

  /// Db session id, created on first set completion (so an empty session
  /// doesn't pollute history).
  int? _sessionId;

  bool _busyAdvancing = false;

  @override
  void initState() {
    super.initState();
    _blocks = List.of(widget.initialPlan.blocks);
    _sessionStartedAt = DateTime.now();
  }

  @override
  void dispose() {
    _restTimer?.cancel();
    super.dispose();
  }

  // ── Computed ────────────────────────────────────────────────────────────

  ExerciseBlock? get _currentBlock => _blocks.isEmpty ? null : _blocks.first;
  bool get _isResting => _restTimer != null && _restSecondsLeft > 0;

  int get _remainingTimeSec {
    final elapsed = DateTime.now().difference(_sessionStartedAt).inSeconds;
    return (widget.timeBudgetSec - elapsed).clamp(0, widget.timeBudgetSec);
  }

  // ── Set completion ──────────────────────────────────────────────────────

  Future<void> _markCurrentSetDone({
    double? weightKgOverride,
    int? repsOverride,
    double? rirOverride,
  }) async {
    final block = _currentBlock;
    if (block == null) return;
    final db = ref.read(databaseProvider);

    // Lazy-create the db session row on first completed set.
    _sessionId ??= await db.startSession(
      timeBudgetSec: widget.timeBudgetSec,
      targetRir: widget.targetRir,
    );

    final ws = WorkoutSet(
      exercise: block.exerciseId,
      weightKg: weightKgOverride ?? block.weightKg,
      reps: repsOverride ?? block.reps,
      rir: rirOverride ?? block.predictedRir.clamp(0.0, 5.0),
      timestamp: DateTime.now(),
    );
    _completedSets.add(ws);
    await db.addCompletedSet(
      sessionId: _sessionId,
      exerciseId: ws.exercise,
      weightKg: ws.weightKg,
      reps: ws.reps,
      rir: ws.rir,
      timestamp: ws.timestamp,
    );

    if (!mounted) return;
    setState(() {
      // Advance set counter or block.
      if (_setInBlock < block.setsCount) {
        _setInBlock += 1;
      } else {
        _blocks.removeAt(0);
        _setInBlock = 1;
        _blocksCompleted += 1;
      }
    });

    // Start rest timer if there's another set/block to do.
    if (_blocks.isNotEmpty) {
      final settings = ref.read(settingsProvider).value;
      _startRest(settings?.restBetweenSetsSec ?? 180);
    }
  }

  // ── Rest timer ──────────────────────────────────────────────────────────

  void _startRest(int seconds) {
    _restTimer?.cancel();
    setState(() => _restSecondsLeft = seconds);
    _restTimer = Timer.periodic(const Duration(seconds: 1), (t) {
      if (!mounted) {
        t.cancel();
        return;
      }
      setState(() => _restSecondsLeft -= 1);
      if (_restSecondsLeft <= 0) {
        t.cancel();
        _restTimer = null;
      }
    });
  }

  void _skipRest() {
    _restTimer?.cancel();
    _restTimer = null;
    setState(() => _restSecondsLeft = 0);
  }

  // ── Adjust dialog ───────────────────────────────────────────────────────

  Future<void> _openAdjust() async {
    final block = _currentBlock;
    if (block == null) return;
    final result = await showDialog<({double weight, int reps, double rir})>(
      context: context,
      builder: (_) => _AdjustDialog(
        initialWeight: block.weightKg,
        initialReps: block.reps,
        initialRir: block.predictedRir,
      ),
    );
    if (result == null || !mounted) return;
    // Only update the current set's prescribed values — do NOT mark it done.
    // The user completes the set explicitly with the "Done" button.
    setState(() {
      _blocks[0] = block.copyWith(
        weightKg: result.weight,
        reps: result.reps,
        predictedRir: result.rir,
      );
    });
  }

  // ── Change exercise (replan) ─────────────────────────────────────────────

  Future<void> _changeExercise() async {
    final block = _currentBlock;
    if (block == null || _busyAdvancing) return;

    // Snapshot so the swap can be undone from the SnackBar.
    final prevBlocks = List.of(_blocks);
    final prevSetInBlock = _setInBlock;

    setState(() => _busyAdvancing = true);
    try {
      _dismissed.add(block.exerciseId);
      // Log the swap (analytics).
      if (_sessionId != null) {
        await ref.read(databaseProvider).logDismissed(_sessionId!, block.exerciseId);
      }

      // Replace ONLY the current exercise — keep the rest of the plan intact so
      // the total exercise count stays stable (a swap, not a full re-plan).
      // Exclude the remaining blocks too so the replacement isn't a duplicate
      // of something still to come.
      final keepRest = _blocks.sublist(1);
      final planner = await ref.read(plannerProvider.future);
      final pastHistory = ref.read(historyProvider).value ?? const [];
      final fullHistory = [...pastHistory, ..._completedSets];
      final newPlan = planner.plan(
        userHistory: fullHistory,
        timeBudgetSec: _remainingTimeSec,
        targetRir: widget.targetRir,
        exclusions: {..._dismissed, for (final b in keepRest) b.exerciseId},
      );

      if (!mounted) return;

      // Prefer a like-for-like replacement (same exercise type), else the
      // planner's top pick.
      final candidates = newPlan.blocks;
      final replacement = candidates.isEmpty
          ? null
          : candidates.firstWhere(
              (b) => b.exType == block.exType,
              orElse: () => candidates.first,
            );

      if (replacement == null) {
        // No alternative available — revert the dismissal and tell the user.
        _dismissed.remove(block.exerciseId);
        ScaffoldMessenger.of(context)
          ..clearSnackBars()
          ..showSnackBar(const SnackBar(
            behavior: SnackBarBehavior.floating,
            content: Text('No alternative exercise available.'),
          ));
        return;
      }

      setState(() {
        _blocks = [replacement, ...keepRest];
        _setInBlock = 1;
      });
      _showChangeFeedback(
        from: block.exerciseId,
        to: replacement.exerciseId,
        prevBlocks: prevBlocks,
        prevSetInBlock: prevSetInBlock,
        dismissedId: block.exerciseId,
      );
    } finally {
      if (mounted) setState(() => _busyAdvancing = false);
    }
  }

  void _showChangeFeedback({
    required String from,
    required String to,
    required List<ExerciseBlock> prevBlocks,
    required int prevSetInBlock,
    required String dismissedId,
  }) {
    final messenger = ScaffoldMessenger.of(context);
    messenger.clearSnackBars();
    messenger.showSnackBar(
      SnackBar(
        behavior: SnackBarBehavior.floating,
        content: Text('${_pretty(from)}  →  ${_pretty(to)}'),
        action: SnackBarAction(
          label: 'Undo',
          onPressed: () {
            if (!mounted) return;
            setState(() {
              _blocks = prevBlocks;
              _setInBlock = prevSetInBlock;
              _dismissed.remove(dismissedId);
            });
          },
        ),
      ),
    );
  }

  // ── End session ─────────────────────────────────────────────────────────

  Future<void> _endSession() async {
    if (_sessionId != null) {
      final db = ref.read(databaseProvider);
      await db.endSession(_sessionId!);

      // Update strength anchors from this session's quality sets.
      final priors = await ref.read(strengthPriorsProvider.future);
      final currentRow = await db.getAnchors();
      if (currentRow != null) {
        final newAnchors = priors.updateStrengthAnchors(
          [currentRow.benchPressKg, currentRow.squatKg, currentRow.deadliftKg],
          [for (final s in _completedSets) s.toJson()],
        );
        await db.setAnchors(
          benchKg: newAnchors[0],
          squatKg: newAnchors[1],
          deadliftKg: newAnchors[2],
        );
        // anchorsProvider is a FutureProvider — invalidate so muscle
        // dashboard / planner / charts pick up the new 1RMs immediately.
        ref.invalidate(anchorsProvider);
      }
      // Also force fresh recomputes of MPC + sessions list before the home
      // screen rebuilds, so the dashboard reflects this session's sets.
      ref.invalidate(mpcProvider);
      ref.invalidate(sessionsProvider);
    }
    if (!mounted) return;
    Navigator.of(context).pop();
  }

  // ── UI ──────────────────────────────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final block = _currentBlock;

    return PopScope(
      canPop: false,
      onPopInvokedWithResult: (didPop, _) async {
        if (didPop) return;
        final ok = await showDialog<bool>(
          context: context,
          builder: (_) => AlertDialog(
            title: const Text('End session?'),
            content: const Text('Your progress will be saved.'),
            actions: [
              TextButton(
                onPressed: () => Navigator.of(context).pop(false),
                child: const Text('Keep going'),
              ),
              FilledButton(
                onPressed: () => Navigator.of(context).pop(true),
                child: const Text('End'),
              ),
            ],
          ),
        );
        if (ok == true) await _endSession();
      },
      child: Scaffold(
        appBar: AppBar(
          title: Text(block == null
              ? 'Session complete'
              : 'Exercise ${_blocksCompleted + 1} of '
                  '${_blocksCompleted + _blocks.length}'),
          actions: [
            TextButton(
              onPressed: _endSession,
              child: const Text('End'),
            ),
          ],
        ),
        // SafeArea bottom inset prevents the action buttons / Skip-rest
        // from sitting under the Android system gesture bar.
        body: SafeArea(
          top: false,
          child: Stack(
          children: [
            if (block == null)
              _SessionDoneView(onEnd: _endSession)
            else
              _ActiveBlockView(
                block: block,
                setInBlock: _setInBlock,
                upcoming: _blocks.length > 1 ? _blocks[1] : null,
                isResting: _isResting,
                restSecondsLeft: _restSecondsLeft,
                onDone: () => _markCurrentSetDone(),
                onAdjust: _openAdjust,
                onChange: _changeExercise,
                onSkipRest: _skipRest,
              ),
            if (_busyAdvancing)
              Positioned.fill(
                child: ColoredBox(
                  color: theme.colorScheme.surface.withValues(alpha: 0.7),
                  child: const Center(child: CircularProgressIndicator()),
                ),
              ),
          ],
          ),
        ),
      ),
    );
  }

}

// ─────────────────────────────────────────────────────────────────────────────
// Sub-widgets
// ─────────────────────────────────────────────────────────────────────────────

class _ActiveBlockView extends StatelessWidget {
  final ExerciseBlock block;
  final int setInBlock;
  final ExerciseBlock? upcoming;
  final bool isResting;
  final int restSecondsLeft;
  final VoidCallback onDone;
  final VoidCallback onAdjust;
  final VoidCallback onChange;
  final VoidCallback onSkipRest;

  const _ActiveBlockView({
    required this.block,
    required this.setInBlock,
    required this.upcoming,
    required this.isResting,
    required this.restSecondsLeft,
    required this.onDone,
    required this.onAdjust,
    required this.onChange,
    required this.onSkipRest,
  });

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final scheme = theme.colorScheme;

    return Padding(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Card(
            elevation: 0,
            color: scheme.surfaceContainerHigh,
            child: Padding(
              padding: const EdgeInsets.all(20),
              child: AnimatedSwitcher(
                duration: const Duration(milliseconds: 320),
                switchInCurve: Curves.easeOutCubic,
                switchOutCurve: Curves.easeInCubic,
                transitionBuilder: (child, animation) => FadeTransition(
                  opacity: animation,
                  child: SlideTransition(
                    position: Tween<Offset>(
                      begin: const Offset(0, 0.10),
                      end: Offset.zero,
                    ).animate(animation),
                    child: child,
                  ),
                ),
                child: SizedBox(
                  // Key drives the animation: changes on every set advance
                  // (Done) and every exercise swap (Change exercise).
                  key: ValueKey('${block.exerciseId}_$setInBlock'),
                  width: double.infinity,
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Text(
                        _pretty(block.exerciseId),
                        textAlign: TextAlign.center,
                        style: theme.textTheme.headlineSmall?.copyWith(
                          fontWeight: FontWeight.w700,
                        ),
                      ),
                      const SizedBox(height: 8),
                      Text(
                        'Set $setInBlock of ${block.setsCount}',
                        style: theme.textTheme.titleMedium?.copyWith(
                          color: scheme.onSurfaceVariant,
                        ),
                      ),
                      const SizedBox(height: 24),
                      Text(
                        '${block.weightKg.toStringAsFixed(1)} kg',
                        style: theme.textTheme.displayMedium?.copyWith(
                          fontWeight: FontWeight.w800,
                          color: scheme.primary,
                        ),
                      ),
                      Text(
                        '× ${block.reps} reps',
                        style: theme.textTheme.headlineSmall?.copyWith(
                          color: scheme.onSurfaceVariant,
                        ),
                      ),
                      const SizedBox(height: 8),
                      Text(
                        'Predicted RIR ${block.predictedRir.toStringAsFixed(1)}',
                        style: theme.textTheme.bodyMedium?.copyWith(
                          color: scheme.onSurfaceVariant,
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ),
          ),
          const Spacer(),
          if (isResting)
            _RestPanel(secondsLeft: restSecondsLeft, onSkip: onSkipRest)
          else
            Column(
              children: [
                Row(
                  children: [
                    Expanded(
                      child: OutlinedButton(
                        onPressed: onAdjust,
                        style: OutlinedButton.styleFrom(
                          minimumSize: const Size.fromHeight(56),
                        ),
                        child: const Text('Adjust'),
                      ),
                    ),
                    const SizedBox(width: 12),
                    Expanded(
                      flex: 2,
                      child: FilledButton(
                        onPressed: onDone,
                        style: FilledButton.styleFrom(
                          minimumSize: const Size.fromHeight(56),
                          textStyle: theme.textTheme.titleMedium,
                        ),
                        child: const Text('Done'),
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 12),
                TextButton.icon(
                  onPressed: onChange,
                  icon: const Icon(Icons.swap_horiz),
                  label: const Text('Change exercise'),
                  style: TextButton.styleFrom(
                    minimumSize: const Size.fromHeight(48),
                    foregroundColor: scheme.onSurfaceVariant,
                  ),
                ),
              ],
            ),
          if (upcoming != null) ...[
            const SizedBox(height: 16),
            Text(
              'Up next: ${_pretty(upcoming!.exerciseId)} '
              '${upcoming!.setsCount}×${upcoming!.reps} '
              '@ ${upcoming!.weightKg.toStringAsFixed(1)} kg',
              textAlign: TextAlign.center,
              style: theme.textTheme.bodySmall?.copyWith(
                color: scheme.onSurfaceVariant,
              ),
            ),
          ],
        ],
      ),
    );
  }
}

class _RestPanel extends StatelessWidget {
  final int secondsLeft;
  final VoidCallback onSkip;

  const _RestPanel({required this.secondsLeft, required this.onSkip});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final m = secondsLeft ~/ 60;
    final s = secondsLeft % 60;
    return Column(
      children: [
        Text(
          'Rest',
          style: theme.textTheme.titleMedium?.copyWith(
            color: theme.colorScheme.onSurfaceVariant,
          ),
        ),
        const SizedBox(height: 8),
        Text(
          '$m:${s.toString().padLeft(2, '0')}',
          style: theme.textTheme.displayLarge?.copyWith(
            fontFeatures: const [FontFeature.tabularFigures()],
            fontWeight: FontWeight.w800,
            color: theme.colorScheme.primary,
          ),
        ),
        const SizedBox(height: 16),
        OutlinedButton(
          onPressed: onSkip,
          style: OutlinedButton.styleFrom(
            minimumSize: const Size.fromHeight(56),
          ),
          child: const Text('Skip rest'),
        ),
      ],
    );
  }
}

class _SessionDoneView extends StatelessWidget {
  final VoidCallback onEnd;
  const _SessionDoneView({required this.onEnd});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Padding(
      padding: const EdgeInsets.all(24),
      child: Column(
        children: [
          const Spacer(),
          Icon(Icons.check_circle, size: 96, color: theme.colorScheme.primary),
          const SizedBox(height: 16),
          Text('Nice work.',
              style: theme.textTheme.headlineMedium?.copyWith(
                fontWeight: FontWeight.w700,
              )),
          const SizedBox(height: 8),
          Text('All sets complete.',
              style: theme.textTheme.bodyLarge?.copyWith(
                color: theme.colorScheme.onSurfaceVariant,
              )),
          const Spacer(),
          FilledButton(
            onPressed: onEnd,
            style: FilledButton.styleFrom(
              minimumSize: const Size.fromHeight(56),
            ),
            child: const Text('Finish session'),
          ),
        ],
      ),
    );
  }
}

class _AdjustDialog extends StatefulWidget {
  final double initialWeight;
  final int initialReps;
  final double initialRir;

  const _AdjustDialog({
    required this.initialWeight,
    required this.initialReps,
    required this.initialRir,
  });

  @override
  State<_AdjustDialog> createState() => _AdjustDialogState();
}

class _AdjustDialogState extends State<_AdjustDialog> {
  late final TextEditingController _w =
      TextEditingController(text: widget.initialWeight.toStringAsFixed(1));
  late final TextEditingController _r =
      TextEditingController(text: widget.initialReps.toString());
  late double _rir = widget.initialRir.clamp(0.0, 5.0);

  @override
  void dispose() {
    _w.dispose();
    _r.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AlertDialog(
      title: const Text('Adjust set'),
      content: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          TextField(
            controller: _w,
            keyboardType: const TextInputType.numberWithOptions(decimal: true),
            decoration: const InputDecoration(labelText: 'Weight (kg)'),
          ),
          const SizedBox(height: 12),
          TextField(
            controller: _r,
            keyboardType: TextInputType.number,
            decoration: const InputDecoration(labelText: 'Reps'),
          ),
          const SizedBox(height: 16),
          Text('RIR: ${_rir.toStringAsFixed(0)}'),
          Slider(
            min: 0, max: 5, divisions: 5,
            value: _rir.clamp(0.0, 5.0),
            label: _rir.toStringAsFixed(0),
            onChanged: (v) => setState(() => _rir = v),
          ),
        ],
      ),
      actions: [
        TextButton(
          onPressed: () => Navigator.of(context).pop(),
          child: const Text('Cancel'),
        ),
        FilledButton(
          onPressed: () {
            final weight = double.tryParse(_w.text.replaceAll(',', '.'));
            final reps = int.tryParse(_r.text);
            if (weight == null || reps == null || weight <= 0 || reps <= 0) {
              return;
            }
            Navigator.of(context).pop((
              weight: weight,
              reps: reps,
              rir: _rir,
            ));
          },
          child: const Text('Save'),
        ),
      ],
    );
  }
}

String _pretty(String id) => id
    .split('_')
    .map((p) => p.isEmpty ? p : '${p[0].toUpperCase()}${p.substring(1)}')
    .join(' ');
