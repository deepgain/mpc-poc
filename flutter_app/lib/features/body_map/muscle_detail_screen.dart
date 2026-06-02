/// Muscle detail — shows current MPC + a 7-day trajectory line chart
/// recomputed by replaying [DeepGain.predictMpc] at hourly checkpoints.
///
/// Cost: ~168 predictMpc calls per chart open. Each call is ~10ms on host
/// for typical history sizes, so a chart loads in ~1.5–2s. If this gets
/// too slow we'll downsample (every 4h or only at session boundaries).
library;

import 'package:deepgain_app/inference/deepgain.dart';
import 'package:deepgain_app/inference/strength.dart';
import 'package:deepgain_app/inference/types.dart';
import 'package:deepgain_app/state/providers.dart';
import 'package:fl_chart/fl_chart.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:intl/intl.dart';

const int _trajectoryDays = 7;
const int _hoursPerSample = 1;

class MuscleDetailScreen extends ConsumerWidget {
  final String muscle;
  const MuscleDetailScreen({super.key, required this.muscle});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final theme = Theme.of(context);
    final mpc = ref.watch(mpcProvider).valueOrNull;
    final history = ref.watch(historyProvider).valueOrNull ?? const [];
    final anchors = ref.watch(anchorsProvider).valueOrNull;
    final dgAsync = ref.watch(deepGainProvider);
    final priorsAsync = ref.watch(strengthPriorsProvider);

    return Scaffold(
      appBar: AppBar(title: Text(_pretty(muscle))),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          _CurrentMpcCard(value: mpc?[muscle] ?? 1.0),
          const SizedBox(height: 24),
          Text(
            'Last $_trajectoryDays days',
            style: theme.textTheme.titleMedium,
          ),
          const SizedBox(height: 8),
          if (dgAsync.isLoading || priorsAsync.isLoading)
            const SizedBox(height: 200, child: Center(child: CircularProgressIndicator()))
          else
            FutureBuilder<List<_TrajectoryPoint>>(
              future: _computeTrajectory(
                muscle: muscle,
                history: history,
                anchors: anchors,
                dg: dgAsync.value!,
                priors: priorsAsync.value!,
              ),
              builder: (context, snap) {
                if (!snap.hasData) {
                  return const SizedBox(
                    height: 200,
                    child: Center(child: CircularProgressIndicator()),
                  );
                }
                return SizedBox(
                  height: 220,
                  child: _MpcTrajectoryChart(
                    points: snap.data!,
                  ),
                );
              },
            ),
        ],
      ),
    );
  }
}

class _CurrentMpcCard extends StatelessWidget {
  final double value;
  const _CurrentMpcCard({required this.value});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final scheme = theme.colorScheme;
    final color = value < 0.55
        ? scheme.error
        : value < 0.85
            ? Colors.orange
            : Colors.green;
    return Card(
      elevation: 0,
      color: scheme.surfaceContainerLow,
      child: Padding(
        padding: const EdgeInsets.all(20),
        child: Row(
          children: [
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text('Current MPC',
                      style: theme.textTheme.titleSmall?.copyWith(
                        color: scheme.onSurfaceVariant,
                      )),
                  const SizedBox(height: 8),
                  Text(value.toStringAsFixed(2),
                      style: theme.textTheme.displaySmall?.copyWith(
                        fontWeight: FontWeight.w800,
                        color: color,
                      )),
                ],
              ),
            ),
            Container(
              width: 16, height: 80,
              decoration: BoxDecoration(
                color: color,
                borderRadius: BorderRadius.circular(4),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _TrajectoryPoint {
  final DateTime t;
  final double mpc;
  const _TrajectoryPoint(this.t, this.mpc);
}

Future<List<_TrajectoryPoint>> _computeTrajectory({
  required String muscle,
  required List<WorkoutSet> history,
  required AnchorsKg? anchors,
  required DeepGain dg,
  required StrengthPriors priors,
}) async {
  final now = DateTime.now();
  final start = now.subtract(const Duration(days: _trajectoryDays));
  final samples = (_trajectoryDays * 24) ~/ _hoursPerSample;
  final out = <_TrajectoryPoint>[];
  for (var i = 0; i <= samples; i++) {
    final t = start.add(Duration(hours: i * _hoursPerSample));
    final mpcMap = dg.predictMpc(
      history: history,
      timestamp: t,
      anchorsKg: anchors,
      strengthPriors: priors,
    );
    out.add(_TrajectoryPoint(t, mpcMap[muscle] ?? 1.0));
  }
  return out;
}

class _MpcTrajectoryChart extends StatelessWidget {
  final List<_TrajectoryPoint> points;
  const _MpcTrajectoryChart({required this.points});

  @override
  Widget build(BuildContext context) {
    if (points.isEmpty) return const SizedBox.shrink();
    final theme = Theme.of(context);
    final scheme = theme.colorScheme;
    final start = points.first.t;
    final spots = [
      for (final p in points)
        FlSpot(
          p.t.difference(start).inMinutes / 60.0, // hours since start
          p.mpc,
        ),
    ];
    final maxX = spots.last.x;

    return LineChart(
      LineChartData(
        minX: 0, maxX: maxX,
        minY: 0.0, maxY: 1.0,
        gridData: FlGridData(
          show: true,
          drawVerticalLine: false,
          horizontalInterval: 0.25,
          getDrawingHorizontalLine: (_) => FlLine(
            color: scheme.outlineVariant.withValues(alpha: 0.3),
            strokeWidth: 1,
          ),
        ),
        borderData: FlBorderData(show: false),
        titlesData: FlTitlesData(
          rightTitles: const AxisTitles(sideTitles: SideTitles(showTitles: false)),
          topTitles: const AxisTitles(sideTitles: SideTitles(showTitles: false)),
          leftTitles: AxisTitles(
            sideTitles: SideTitles(
              showTitles: true,
              reservedSize: 36,
              interval: 0.25,
              getTitlesWidget: (v, _) => Text(
                v.toStringAsFixed(2),
                style: theme.textTheme.bodySmall?.copyWith(
                  color: scheme.onSurfaceVariant,
                ),
              ),
            ),
          ),
          bottomTitles: AxisTitles(
            sideTitles: SideTitles(
              showTitles: true,
              reservedSize: 24,
              interval: 24,
              getTitlesWidget: (v, _) {
                final t = start.add(Duration(hours: v.round()));
                return Padding(
                  padding: const EdgeInsets.only(top: 4),
                  child: Text(
                    DateFormat('M/d').format(t),
                    style: theme.textTheme.bodySmall?.copyWith(
                      color: scheme.onSurfaceVariant,
                    ),
                  ),
                );
              },
            ),
          ),
        ),
        // Reference horizontal lines at 0.55 and 0.85 (the MPC zone bounds).
        extraLinesData: ExtraLinesData(
          horizontalLines: [
            HorizontalLine(
              y: 0.55,
              color: Colors.red.withValues(alpha: 0.4),
              strokeWidth: 1, dashArray: [4, 4],
            ),
            HorizontalLine(
              y: 0.85,
              color: Colors.green.withValues(alpha: 0.4),
              strokeWidth: 1, dashArray: [4, 4],
            ),
          ],
        ),
        lineBarsData: [
          LineChartBarData(
            spots: spots,
            isCurved: false,
            color: scheme.primary,
            barWidth: 2,
            dotData: const FlDotData(show: false),
            belowBarData: BarAreaData(
              show: true,
              color: scheme.primary.withValues(alpha: 0.1),
            ),
          ),
        ],
      ),
    );
  }
}

String _pretty(String id) => id
    .split('_')
    .map((p) => p.isEmpty ? p : '${p[0].toUpperCase()}${p.substring(1)}')
    .join(' ');
