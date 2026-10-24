import { num, pct } from '../lib/format.js';

// 2×2 confusion matrix. Rows = actual class, columns = predicted class.
export default function ConfusionMatrix({ matrix }) {
  const { tn, fp, fn, tp } = matrix;
  const actualNeg = tn + fp;
  const actualPos = fn + tp;

  const cell = (count, rowTotal, kind, title) => (
    <td className={`cm-cell cm-${kind}`} title={`${title}: ${num(count)} (${pct(count / rowTotal, 1)} of row)`}>
      <div className="cm-count">{num(count)}</div>
      <div className="cm-kind">{title}</div>
      <div className="cm-share">{pct(count / rowTotal, 1)} of actual</div>
    </td>
  );

  return (
    <div className="table-wrap">
      <table className="cm-table">
        <thead>
          <tr>
            <td />
            <th scope="col">Predicted: no diabetes</th>
            <th scope="col">Predicted: diabetes</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <th scope="row">
              Actual: no diabetes
              <div className="th-sub">{num(actualNeg)} patients</div>
            </th>
            {cell(tn, actualNeg, 'correct', 'True negatives')}
            {cell(fp, actualNeg, 'error', 'False positives')}
          </tr>
          <tr>
            <th scope="row">
              Actual: diabetes
              <div className="th-sub">{num(actualPos)} patients</div>
            </th>
            {cell(fn, actualPos, 'error', 'False negatives')}
            {cell(tp, actualPos, 'correct', 'True positives')}
          </tr>
        </tbody>
      </table>
    </div>
  );
}
