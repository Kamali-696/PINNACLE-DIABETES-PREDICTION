import { formatMetric } from '../lib/format.js';

export default function MetricCard({ definition, value, emphasis = false }) {
  const missing = value === null || value === undefined;
  const pct = missing ? 0 : Math.max(0, Math.min(1, value)) * 100;

  return (
    <article className={`card metric-card${emphasis ? ' metric-emphasis' : ''}`}>
      <div className="metric-label">{definition.label}</div>
      <div className="metric-value">{missing ? <span className="muted" style={{ fontSize: '1rem', fontWeight: 500 }}>Not provided</span> : formatMetric(definition, value)}</div>
      <div className="meter" aria-hidden="true">
        <div className="meter-fill" style={{ width: `${pct}%` }} />
      </div>
      <p className="metric-help">{definition.help}</p>
    </article>
  );
}
