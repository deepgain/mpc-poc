/// Settings — edit 1RM anchors, change rest time, export training data.
library;

import 'package:deepgain_app/data/training_export.dart';
import 'package:deepgain_app/state/providers.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:share_plus/share_plus.dart';

class SettingsScreen extends ConsumerWidget {
  const SettingsScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final db = ref.watch(databaseProvider);
    final anchors = ref.watch(anchorsProvider).valueOrNull;
    final settings = ref.watch(settingsProvider).valueOrNull;

    return Scaffold(
      appBar: AppBar(title: const Text('Settings')),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          _Section(
            title: '1RM anchors',
            child: Column(
              children: [
                _AnchorTile(
                  label: 'Bench press',
                  valueKg: anchors?[0],
                  onChanged: (v) async {
                    await db.setAnchors(
                      benchKg: v,
                      squatKg: anchors?[1] ?? 140,
                      deadliftKg: anchors?[2] ?? 180,
                    );
                    ref.invalidate(anchorsProvider);
                  },
                ),
                _AnchorTile(
                  label: 'Squat',
                  valueKg: anchors?[1],
                  onChanged: (v) async {
                    await db.setAnchors(
                      benchKg: anchors?[0] ?? 100,
                      squatKg: v,
                      deadliftKg: anchors?[2] ?? 180,
                    );
                    ref.invalidate(anchorsProvider);
                  },
                ),
                _AnchorTile(
                  label: 'Deadlift',
                  valueKg: anchors?[2],
                  onChanged: (v) async {
                    await db.setAnchors(
                      benchKg: anchors?[0] ?? 100,
                      squatKg: anchors?[1] ?? 140,
                      deadliftKg: v,
                    );
                    ref.invalidate(anchorsProvider);
                  },
                ),
              ],
            ),
          ),
          const SizedBox(height: 16),
          _Section(
            title: 'Training defaults',
            child: Column(
              children: [
                _NumericTile(
                  label: 'Rest between sets',
                  value: settings?.restBetweenSetsSec ?? 180,
                  unit: 'sec',
                  min: 30, max: 600, step: 30,
                  onChanged: (v) => db.updateSettings(restBetweenSetsSec: v),
                ),
                _NumericTile(
                  label: 'Default target RIR',
                  value: settings?.defaultTargetRir ?? 2,
                  unit: '',
                  min: 1, max: 5, step: 1,
                  onChanged: (v) => db.updateSettings(defaultTargetRir: v),
                ),
              ],
            ),
          ),
          const SizedBox(height: 16),
          _Section(
            title: 'Training data',
            child: const _ExportTile(),
          ),
          const SizedBox(height: 16),
          _Section(
            title: 'About',
            child: ListTile(
              title: const Text('DeepGain'),
              subtitle: Text(
                'On-device fatigue model + workout planner.\n'
                'No account, no cloud. Your data stays here.',
                style: Theme.of(context).textTheme.bodySmall,
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _Section extends StatelessWidget {
  final String title;
  final Widget child;
  const _Section({required this.title, required this.child});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Padding(
          padding: const EdgeInsets.fromLTRB(4, 0, 4, 8),
          child: Text(
            title,
            style: theme.textTheme.titleSmall?.copyWith(
              color: theme.colorScheme.onSurfaceVariant,
            ),
          ),
        ),
        Card(
          elevation: 0,
          color: theme.colorScheme.surfaceContainerLow,
          margin: EdgeInsets.zero,
          child: child,
        ),
      ],
    );
  }
}

class _AnchorTile extends StatefulWidget {
  final String label;
  final double? valueKg;
  final ValueChanged<double> onChanged;

  const _AnchorTile({
    required this.label,
    required this.valueKg,
    required this.onChanged,
  });

  @override
  State<_AnchorTile> createState() => _AnchorTileState();
}

class _AnchorTileState extends State<_AnchorTile> {
  Future<void> _edit() async {
    final ctl = TextEditingController(
      text: (widget.valueKg ?? 0).toStringAsFixed(0),
    );
    final result = await showDialog<double>(
      context: context,
      builder: (_) => AlertDialog(
        title: Text(widget.label),
        content: TextField(
          controller: ctl,
          autofocus: true,
          keyboardType: const TextInputType.numberWithOptions(decimal: true),
          inputFormatters: [
            FilteringTextInputFormatter.allow(RegExp(r'[0-9.,]')),
          ],
          decoration: const InputDecoration(suffixText: 'kg'),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(context).pop(),
            child: const Text('Cancel'),
          ),
          FilledButton(
            onPressed: () {
              final v = double.tryParse(ctl.text.replaceAll(',', '.'));
              if (v != null && v > 0 && v < 1000) Navigator.of(context).pop(v);
            },
            child: const Text('Save'),
          ),
        ],
      ),
    );
    if (result != null) widget.onChanged(result);
  }

  @override
  Widget build(BuildContext context) {
    return ListTile(
      title: Text(widget.label),
      trailing: Text(
        '${(widget.valueKg ?? 0).toStringAsFixed(0)} kg',
        style: Theme.of(context)
            .textTheme
            .titleMedium
            ?.copyWith(fontWeight: FontWeight.w600),
      ),
      onTap: _edit,
    );
  }
}

class _NumericTile extends StatelessWidget {
  final String label;
  final int value;
  final String unit;
  final int min, max, step;
  final ValueChanged<int> onChanged;

  const _NumericTile({
    required this.label,
    required this.value,
    required this.unit,
    required this.min,
    required this.max,
    required this.step,
    required this.onChanged,
  });

  @override
  Widget build(BuildContext context) {
    return ListTile(
      title: Text(label),
      subtitle: Text('$value${unit.isEmpty ? '' : ' $unit'}'),
      trailing: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          IconButton(
            icon: const Icon(Icons.remove),
            onPressed: value > min ? () => onChanged(value - step) : null,
          ),
          IconButton(
            icon: const Icon(Icons.add),
            onPressed: value < max ? () => onChanged(value + step) : null,
          ),
        ],
      ),
    );
  }
}

class _ExportTile extends ConsumerStatefulWidget {
  const _ExportTile();

  @override
  ConsumerState<_ExportTile> createState() => _ExportTileState();
}

class _ExportTileState extends ConsumerState<_ExportTile> {
  bool _exporting = false;

  Future<void> _export() async {
    setState(() => _exporting = true);
    try {
      final exporter = TrainingDataExporter(ref.read(databaseProvider));
      final count = await exporter.completedSetsCount();
      if (count == 0) {
        if (!mounted) return;
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('No completed sets yet.')),
        );
        return;
      }
      final file = await exporter.exportToFile();
      if (!mounted) return;
      await Share.shareXFiles(
        [XFile(file.path)],
        subject: 'DeepGain training data ($count sets)',
        text: 'DeepGain training data export (DeepGain Data Standard v2)',
      );
    } finally {
      if (mounted) setState(() => _exporting = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        ListTile(
          title: const Text('Export training data'),
          subtitle: const Text(
            'CSV in DeepGain Data Standard v2 format. '
            'Use to retrain or improve the model.',
          ),
          trailing: _exporting
              ? const SizedBox(
                  width: 20, height: 20,
                  child: CircularProgressIndicator(strokeWidth: 2),
                )
              : const Icon(Icons.ios_share),
          onTap: _exporting ? null : _export,
        ),
      ],
    );
  }
}
