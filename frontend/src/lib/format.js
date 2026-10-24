export const metricDefinitions = [
  {
    key: 'accuracy',
    label: 'Accuracy',
    help: 'Share of all patients classified correctly. Inflated by the large non-diabetic majority.',
  },
  {
    key: 'precision',
    label: 'Precision',
    help: 'Of patients the model flagged as diabetic, the share who actually were.',
  },
  {
    key: 'recall',
    label: 'Recall (sensitivity)',
    help: 'Of patients who actually had diabetes, the share the model flagged.',
  },
  {
    key: 'f1',
    label: 'F1-score',
    help: 'Harmonic mean of precision and recall for the diabetic class.',
  },
  {
    key: 'roc_auc',
    label: 'ROC-AUC',
    decimal: true,
    help: 'How well predicted probabilities rank diabetic above non-diabetic patients (0.5 = chance, 1 = perfect).',
  },
];

export const pct = (v, digits = 2) => `${(v * 100).toFixed(digits)}%`;
export const num = (v) => v.toLocaleString('en-US');
export const formatMetric = (def, v) => (def.decimal ? v.toFixed(4) : pct(v));
