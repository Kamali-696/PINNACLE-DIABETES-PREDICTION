import { num, pct } from '../lib/format.js';

// 100%-stacked horizontal bars, one per split. Classes are direct-labelled and
// listed in a legend, so identity never depends on colour alone.
export default function ClassDistribution({ rows }) {
  return (
    <div className="dist">
      <div className="legend" aria-hidden="true">
        <span className="legend-item">
          <span className="swatch class-0" /> No diabetes (0)
        </span>
        <span className="legend-item">
          <span className="swatch class-1" /> Diabetes (1)
        </span>
      </div>

      {rows.map(({ label, counts }) => {
        const neg = counts['0'] ?? 0;
        const pos = counts['1'] ?? 0;
        const total = neg + pos;
        return (
          <div className="dist-row" key={label}>
            <div className="dist-label">
              {label}
              <span className="muted small"> · {num(total)}</span>
            </div>
            <div className="dist-bar" role="img" aria-label={`${label}: ${pct(pos / total, 1)} diabetic`}>
              <div className="dist-seg class-0" style={{ width: `${(neg / total) * 100}%` }}>
                <span className="tip">No diabetes: {num(neg)} ({pct(neg / total, 1)})</span>
              </div>
              <div className="dist-seg class-1" style={{ width: `${(pos / total) * 100}%` }}>
                <span className="tip">Diabetes: {num(pos)} ({pct(pos / total, 1)})</span>
              </div>
            </div>
            <div className="dist-value">{pct(pos / total, 1)} diabetic</div>
          </div>
        );
      })}

      <details className="table-toggle">
        <summary>Show as table</summary>
        <table className="data-table">
          <thead>
            <tr>
              <th scope="col">Set</th>
              <th scope="col">No diabetes</th>
              <th scope="col">Diabetes</th>
              <th scope="col">Total</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(({ label, counts }) => (
              <tr key={label}>
                <th scope="row">{label}</th>
                <td>{num(counts['0'] ?? 0)}</td>
                <td>{num(counts['1'] ?? 0)}</td>
                <td>{num((counts['0'] ?? 0) + (counts['1'] ?? 0))}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </details>
    </div>
  );
}
