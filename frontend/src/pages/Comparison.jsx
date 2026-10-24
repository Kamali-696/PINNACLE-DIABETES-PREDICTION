import { getMetrics } from '../api/client.js';
import Notice, { Loading } from '../components/Notice.jsx';
import useApi from '../hooks/useApi.js';
import { formatMetric, metricDefinitions, num } from '../lib/format.js';

export default function Comparison() {
  const { data: report, error, loading } = useApi(getMetrics);

  return (
    <>
      <div className="page-header">
        <div>
          <h1>Model Comparison</h1>
          <p className="page-sub">Side-by-side evaluation of trained models on the same held-out test set.</p>
        </div>
      </div>

      <Notice tone="info" title="No comparison available">
        Only one model has been trained in this project. No other model results exist, so no comparison is
        shown. Train an additional model on the same split to compare it here.
      </Notice>

      {loading && <Loading />}
      {error && <Notice tone="danger">{error.message}</Notice>}
      {report && (
        <section className="card">
          <h2>Trained models</h2>
          <div className="table-wrap">
            <table className="data-table">
              <thead>
                <tr>
                  <th scope="col">Model</th>
                  <th scope="col">Test patients</th>
                  {metricDefinitions.map((d) => (
                    <th key={d.key} scope="col">
                      {d.label}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                <tr>
                  <th scope="row">Diabetes Prediction Model</th>
                  <td>{num(report.metrics.test.samples)}</td>
                  {metricDefinitions.map((d) => (
                    <td key={d.key}>{formatMetric(d, report.metrics.test[d.key])}</td>
                  ))}
                </tr>
              </tbody>
            </table>
          </div>
        </section>
      )}
    </>
  );
}
