/// Home — three-tab scaffold (Muscles / Body / History) with the
/// "Plan a session" FAB always available, and a Settings entry-point in the
/// app bar.
library;

import 'dart:math' as math;

import 'package:deepgain_app/features/body_map/body_map_screen.dart';
import 'package:deepgain_app/features/body_map/muscle_detail_screen.dart';
import 'package:deepgain_app/features/history/history_screen.dart';
import 'package:deepgain_app/features/planner/plan_session_sheet.dart';
import 'package:deepgain_app/features/settings/settings_screen.dart';
import 'package:deepgain_app/state/providers.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

class HomeScreen extends ConsumerStatefulWidget {
  const HomeScreen({super.key});

  @override
  ConsumerState<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends ConsumerState<HomeScreen> {
  int _tab = 0;

  // Outlined icon when idle, filled when selected — the Material 3
  // NavigationBar convention. Icons chosen to read as what each tab is:
  // an exercising figure for muscle training/recovery, a full-body figure
  // for the body map, a clock for session history.
  static const _tabs = <_TabSpec>[
    _TabSpec('Muscles', Icons.bolt_outlined, Icons.bolt, _MusclesTab()),
    _TabSpec(
      'Body',
      Icons.accessibility_new_outlined,
      Icons.accessibility_new,
      BodyMapScreen(),
    ),
    _TabSpec('History', Icons.history_outlined, Icons.history, HistoryScreen()),
  ];

  /// Refresh the data backing each tab when it becomes visible. Without
  /// this the muscle dashboard would keep showing MPC computed at
  /// `predictMpc(then)` even after hours have passed (recovery curves
  /// would look frozen), and History wouldn't pick up sessions written
  /// since the screen was first opened.
  void _refreshFor(int tab) {
    switch (tab) {
      case 0: // Muscles
      case 1: // Body — both share mpcProvider
        ref.invalidate(mpcProvider);
      case 2: // History
        ref.invalidate(sessionsProvider);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      // Default false but spelled out: never let body content slip under
      // the NavigationBar / FAB on Android with edge-to-edge gesture nav.
      extendBody: false,
      extendBodyBehindAppBar: false,
      appBar: AppBar(
        title: Text(_tabs[_tab].label),
        scrolledUnderElevation: 0,
        actions: [
          IconButton(
            icon: const Icon(Icons.settings),
            onPressed: () {
              Navigator.of(context).push(
                MaterialPageRoute<void>(builder: (_) => const SettingsScreen()),
              );
            },
          ),
        ],
      ),
      body: IndexedStack(
        index: _tab,
        children: [for (final t in _tabs) t.body],
      ),
      bottomNavigationBar: _BottomBar(
        tabs: _tabs,
        selectedIndex: _tab,
        onSelected: (i) {
          setState(() => _tab = i);
          _refreshFor(i);
        },
      ),
      floatingActionButton: _StartWorkoutButton(
        onPressed: () => _openPlanSheet(context),
      ),
    );
  }

  void _openPlanSheet(BuildContext context) {
    showModalBottomSheet<void>(
      context: context,
      isScrollControlled: true,
      useSafeArea: true,
      showDragHandle: true,
      builder: (_) => const PlanSessionSheet(),
    );
  }
}

class _TabSpec {
  final String label;
  final IconData icon;
  final IconData selectedIcon;
  final Widget body;
  const _TabSpec(this.label, this.icon, this.selectedIcon, this.body);
}

/// Custom bottom navigation bar. Built by hand (rather than NavigationBar) so
/// every tab always shows icon + label, and the selected tab gets an animated
/// rounded pill behind a filled icon plus a primary-tinted label — visually
/// balanced and cohesive with the gradient "Start workout" button.
class _BottomBar extends StatelessWidget {
  final List<_TabSpec> tabs;
  final int selectedIndex;
  final ValueChanged<int> onSelected;

  const _BottomBar({
    required this.tabs,
    required this.selectedIndex,
    required this.onSelected,
  });

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return Material(
      color: scheme.surface,
      // Hairline separator so the bar reads as its own surface above content.
      shape: Border(top: BorderSide(color: scheme.outlineVariant, width: 1)),
      child: SafeArea(
        top: false,
        child: SizedBox(
          height: 64,
          child: Row(
            children: [
              for (var i = 0; i < tabs.length; i++)
                Expanded(
                  child: _BottomBarItem(
                    tab: tabs[i],
                    selected: i == selectedIndex,
                    onTap: () => onSelected(i),
                  ),
                ),
            ],
          ),
        ),
      ),
    );
  }
}

class _BottomBarItem extends StatelessWidget {
  final _TabSpec tab;
  final bool selected;
  final VoidCallback onTap;

  const _BottomBarItem({
    required this.tab,
    required this.selected,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final accent = selected ? scheme.primary : scheme.onSurfaceVariant;
    return InkWell(
      onTap: onTap,
      // Suppress the full-cell rectangular ripple — the animated pill below is
      // the only highlight, so two competing highlights don't stack.
      splashColor: Colors.transparent,
      highlightColor: Colors.transparent,
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          AnimatedContainer(
            duration: const Duration(milliseconds: 200),
            curve: Curves.easeOut,
            padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 4),
            decoration: BoxDecoration(
              color: selected ? scheme.primary.withValues(alpha: 0.14) : null,
              borderRadius: BorderRadius.circular(16),
            ),
            child: Icon(
              selected ? tab.selectedIcon : tab.icon,
              size: 24,
              color: accent,
            ),
          ),
          const SizedBox(height: 4),
          Text(
            tab.label,
            style: TextStyle(
              fontSize: 12,
              fontWeight: selected ? FontWeight.w700 : FontWeight.w500,
              color: accent,
            ),
          ),
        ],
      ),
    );
  }
}

/// Prominent gradient pill that launches the plan-a-workout sheet. Built as a
/// custom widget (rather than FloatingActionButton.extended) so we can give it
/// a branded gradient, a soft tinted shadow, and a bold confident label.
class _StartWorkoutButton extends StatelessWidget {
  final VoidCallback onPressed;
  const _StartWorkoutButton({required this.onPressed});

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    // Hugs its content (Row is mainAxisSize.min, no minWidth) so the pill
    // is only as wide as the icon + label need.
    return DecoratedBox(
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(28),
        gradient: LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [
            scheme.primary,
            Color.lerp(scheme.primary, Colors.black, 0.18)!,
          ],
        ),
        boxShadow: [
          BoxShadow(
            color: scheme.primary.withValues(alpha: 0.35),
            blurRadius: 16,
            offset: const Offset(0, 5),
          ),
        ],
      ),
      child: Material(
        color: Colors.transparent,
        child: InkWell(
          borderRadius: BorderRadius.circular(28),
          onTap: onPressed,
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 22, vertical: 15),
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                Icon(Icons.bolt_rounded, color: scheme.onPrimary, size: 22),
                const SizedBox(width: 8),
                Text(
                  'Start workout',
                  style: TextStyle(
                    color: scheme.onPrimary,
                    fontSize: 16,
                    fontWeight: FontWeight.w700,
                    letterSpacing: 0.3,
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Muscles tab — original muscle dashboard, refactored to be tappable
// (drills into per-muscle MPC chart).
// ─────────────────────────────────────────────────────────────────────────────

class _MusclesTab extends ConsumerWidget {
  const _MusclesTab();

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final mpc = ref.watch(mpcProvider);
    final assets = ref.watch(modelAssetsProvider);
    final theme = Theme.of(context);

    return RefreshIndicator(
      onRefresh: () async {
        ref.invalidate(mpcProvider);
        await ref.read(mpcProvider.future);
      },
      child: mpc.when(
        loading: () => ListView(
          physics: const AlwaysScrollableScrollPhysics(),
          children: const [
            SizedBox(height: 200),
            Center(child: CircularProgressIndicator()),
          ],
        ),
        error: (e, _) => ListView(
          physics: const AlwaysScrollableScrollPhysics(),
          children: [Center(child: Text('$e'))],
        ),
        data: (mpcMap) {
          final tau = assets.value!.tau;
          final muscles = assets.value!.muscles;
          final entries = List.generate(
            muscles.length,
            (i) => _MuscleRow(
              muscle: muscles[i],
              mpc: mpcMap[muscles[i]] ?? 1.0,
              tauHours: tau[i],
            ),
          )..sort((a, b) => a.mpc.compareTo(b.mpc));

          return ListView.separated(
            physics: const AlwaysScrollableScrollPhysics(),
            // Bottom padding clears the extended FAB (~64dp + margin)
            // sitting above the NavigationBar.
            padding: const EdgeInsets.fromLTRB(16, 8, 16, 140),
            itemCount: entries.length + 1,
            separatorBuilder: (_, _) => const SizedBox(height: 8),
            itemBuilder: (context, i) {
              if (i == 0) {
                return Padding(
                  padding: const EdgeInsets.only(bottom: 8, top: 4),
                  child: Text(
                    'Muscle performance capacities',
                    style: theme.textTheme.titleMedium?.copyWith(
                      color: theme.colorScheme.onSurfaceVariant,
                    ),
                  ),
                );
              }
              return entries[i - 1];
            },
          );
        },
      ),
    );
  }
}

class _MuscleRow extends StatelessWidget {
  final String muscle;
  final double mpc;
  final double tauHours;

  const _MuscleRow({
    required this.muscle,
    required this.mpc,
    required this.tauHours,
  });

  /// Solve `target = 1 - (1 - mpc) * exp(-dt/tau)` for `dt` in hours.
  /// Returns 0 if already at or above target.
  double _hoursToReach(double target) {
    if (mpc >= target) return 0.0;
    if (mpc <= 0.1) return tauHours * 5;
    final ratio = (1.0 - mpc) / (1.0 - target);
    if (ratio <= 1.0) return 0.0;
    return tauHours * math.log(ratio);
  }

  String _formatHours(double h) {
    if (h <= 0) return 'Ready now';
    if (h < 1) return 'Ready in ${(h * 60).round()} min';
    if (h < 24) return 'Ready in ${h.round()} h';
    final days = h / 24;
    return 'Ready in ${days.toStringAsFixed(days < 2 ? 1 : 0)} d';
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final scheme = theme.colorScheme;
    final color = mpc < 0.55
        ? scheme.error
        : mpc < 0.85
        ? Colors.orange
        : Colors.green;
    final hoursToReady = _hoursToReach(0.85);

    return Card(
      elevation: 0,
      color: scheme.surfaceContainerLow,
      child: InkWell(
        borderRadius: BorderRadius.circular(12),
        onTap: () {
          Navigator.of(context).push(
            MaterialPageRoute<void>(
              builder: (_) => MuscleDetailScreen(muscle: muscle),
            ),
          );
        },
        child: ListTile(
          contentPadding: const EdgeInsets.fromLTRB(16, 8, 16, 8),
          title: Row(
            children: [
              Expanded(
                child: Text(
                  _prettyMuscle(muscle),
                  style: theme.textTheme.titleMedium,
                ),
              ),
              Text(
                mpc.toStringAsFixed(2),
                style: theme.textTheme.titleMedium?.copyWith(
                  fontWeight: FontWeight.w700,
                  color: color,
                  fontFeatures: const [FontFeature.tabularFigures()],
                ),
              ),
            ],
          ),
          subtitle: Padding(
            padding: const EdgeInsets.only(top: 8),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                ClipRRect(
                  borderRadius: BorderRadius.circular(8),
                  child: LinearProgressIndicator(
                    value: ((mpc - 0.1) / 0.9).clamp(0.0, 1.0),
                    minHeight: 8,
                    color: color,
                    backgroundColor: scheme.surfaceContainerHigh,
                  ),
                ),
                const SizedBox(height: 6),
                Text(
                  _formatHours(hoursToReady),
                  style: theme.textTheme.bodySmall?.copyWith(
                    color: scheme.onSurfaceVariant,
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

String _prettyMuscle(String id) {
  return id
      .split('_')
      .map((p) => p.isEmpty ? p : '${p[0].toUpperCase()}${p.substring(1)}')
      .join(' ');
}
