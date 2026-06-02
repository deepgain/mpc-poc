/// History — past sessions, oldest first. Pull-to-refresh + auto-refresh
/// when the History tab becomes active (see HomeScreen).
library;

import 'package:deepgain_app/data/database.dart';
import 'package:deepgain_app/state/providers.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:intl/intl.dart';

class HistoryScreen extends ConsumerWidget {
  const HistoryScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final sessions = ref.watch(sessionsProvider);
    final theme = Theme.of(context);

    return RefreshIndicator(
      onRefresh: () async {
        ref.invalidate(sessionsProvider);
        await ref.read(sessionsProvider.future);
      },
      child: sessions.when(
        loading: () => const _Loading(),
        error: (e, _) => _Empty(
          text: 'Could not load history.\n$e',
          textColor: theme.colorScheme.error,
        ),
        data: (rows) {
          if (rows.isEmpty) {
            return const _Empty(
              text:
                  'No sessions yet.\n\nGenerate a plan from the home tab to start a workout.',
            );
          }
          return ListView.separated(
            // Always physics-scrollable so RefreshIndicator works on short lists.
            physics: const AlwaysScrollableScrollPhysics(),
            padding: const EdgeInsets.fromLTRB(16, 8, 16, 140),
            itemCount: rows.length,
            separatorBuilder: (_, _) => const SizedBox(height: 8),
            itemBuilder: (_, i) => _SessionTile(session: rows[i]),
          );
        },
      ),
    );
  }
}

class _Loading extends StatelessWidget {
  const _Loading();
  @override
  Widget build(BuildContext context) {
    // Wrap in a scrollable so the RefreshIndicator can still be triggered
    // while the initial future is in flight.
    return ListView(
      physics: const AlwaysScrollableScrollPhysics(),
      children: const [
        SizedBox(height: 200),
        Center(child: CircularProgressIndicator()),
      ],
    );
  }
}

class _Empty extends StatelessWidget {
  final String text;
  final Color? textColor;
  const _Empty({required this.text, this.textColor});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return ListView(
      physics: const AlwaysScrollableScrollPhysics(),
      padding: const EdgeInsets.symmetric(horizontal: 32),
      children: [
        const SizedBox(height: 120),
        Text(
          text,
          textAlign: TextAlign.center,
          style: theme.textTheme.bodyLarge?.copyWith(
            color: textColor ?? theme.colorScheme.onSurfaceVariant,
          ),
        ),
      ],
    );
  }
}

class _SessionTile extends ConsumerWidget {
  final Session session;
  const _SessionTile({required this.session});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final db = ref.watch(databaseProvider);
    final theme = Theme.of(context);
    final scheme = theme.colorScheme;
    return FutureBuilder<List<CompletedSet>>(
      future: db.setsForSession(session.id),
      builder: (context, snap) {
        final sets = snap.data ?? const [];
        final exerciseCount = sets.map((s) => s.exerciseId).toSet().length;
        final dateLabel = DateFormat('EEE, MMM d · HH:mm').format(session.startedAt);
        final durLabel = session.endedAt != null
            ? '${session.endedAt!.difference(session.startedAt).inMinutes} min'
            : 'in progress';
        return Card(
          elevation: 0,
          color: scheme.surfaceContainerLow,
          child: ListTile(
            contentPadding:
                const EdgeInsets.symmetric(horizontal: 16, vertical: 4),
            title: Text(dateLabel, style: theme.textTheme.titleMedium),
            subtitle: Padding(
              padding: const EdgeInsets.only(top: 4),
              child: Text(
                '$exerciseCount exercises · ${sets.length} sets · $durLabel',
                style: theme.textTheme.bodySmall?.copyWith(
                  color: scheme.onSurfaceVariant,
                ),
              ),
            ),
            trailing: const Icon(Icons.chevron_right),
            onTap: sets.isEmpty
                ? null
                : () {
                    Navigator.of(context).push(
                      MaterialPageRoute<void>(
                        builder: (_) => _SessionDetailScreen(
                          session: session,
                          sets: sets,
                        ),
                      ),
                    );
                  },
          ),
        );
      },
    );
  }
}

class _SessionDetailScreen extends StatelessWidget {
  final Session session;
  final List<CompletedSet> sets;

  const _SessionDetailScreen({required this.session, required this.sets});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final scheme = theme.colorScheme;
    final dateLabel = DateFormat('EEEE, MMMM d').format(session.startedAt);

    final groups = <List<CompletedSet>>[];
    for (final s in sets) {
      if (groups.isEmpty || groups.last.first.exerciseId != s.exerciseId) {
        groups.add([s]);
      } else {
        groups.last.add(s);
      }
    }

    return Scaffold(
      appBar: AppBar(title: Text(dateLabel)),
      body: ListView.separated(
        padding: const EdgeInsets.fromLTRB(16, 8, 16, 24),
        itemCount: groups.length,
        separatorBuilder: (_, _) => const SizedBox(height: 8),
        itemBuilder: (_, i) {
          final group = groups[i];
          final ex = group.first.exerciseId;
          return Card(
            elevation: 0,
            color: scheme.surfaceContainerLow,
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(_pretty(ex), style: theme.textTheme.titleMedium),
                  const SizedBox(height: 8),
                  for (var k = 0; k < group.length; k++)
                    Padding(
                      padding: const EdgeInsets.symmetric(vertical: 2),
                      child: Row(
                        children: [
                          SizedBox(
                            width: 32,
                            child: Text('${k + 1}',
                                style: theme.textTheme.bodySmall?.copyWith(
                                  color: scheme.onSurfaceVariant,
                                )),
                          ),
                          Expanded(
                            child: Text(
                              '${group[k].weightKg.toStringAsFixed(1)} kg × ${group[k].reps} '
                              '@ RIR ${group[k].rir.toStringAsFixed(0)}',
                              style: theme.textTheme.bodyMedium,
                            ),
                          ),
                          Text(
                            DateFormat('HH:mm').format(group[k].timestamp),
                            style: theme.textTheme.bodySmall?.copyWith(
                              color: scheme.onSurfaceVariant,
                            ),
                          ),
                        ],
                      ),
                    ),
                ],
              ),
            ),
          );
        },
      ),
    );
  }
}

String _pretty(String id) => id
    .split('_')
    .map((p) => p.isEmpty ? p : '${p[0].toUpperCase()}${p.substring(1)}')
    .join(' ');
