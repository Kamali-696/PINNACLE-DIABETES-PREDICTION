import { Link } from 'react-router-dom';
import { getMetrics } from '../api/client.js';
import ClassDistribution from '../components/ClassDistribution.jsx';
import ConfusionMatrix from '../components/ConfusionMatrix.jsx';
import MetricCard from '../components/MetricCard.jsx';
import Notice, { ClinicalDisclaimer, Loading } from '../components/Notice.jsx';
import { confusionMatrix, reportedMetrics } from '../data/reportedMetrics.js';
import useApi from '../hooks/useApi.js';
import { metricDefinitions, num, pct } from '../lib/format.js';

export default function Dashboard() {
  const { data: report, error, loading } = useApi(getMetrics);

  return (
    <>
      <div className="page-header">
        <div>
          <h1>Model Evaluation</h1>
          <p className="page-sub">
            {report
              ? `Diabetes Prediction Model · ${report.dataset.file}`
              : 'Diabetes Prediction Model'}
          </p>
        </div>
        <Link to="/predict" className="btn btn-primary">
          New risk prediction
        </Link>
      </div>

      {loading && <Loading label="Loading evaluation report…" />}
      {error && (
        <Notice tone="danger" title="Evaluation report unavailable">
          {error.message}
        </Notice>
      )}
      {report && <Report report={report} />}

      <ClinicalDisclaimer />
    </>
  );
}

function Report({ report }) {
  const { dataset, split } = report;
  const cm = confusionMatrix;

  return (
    <>
      <section aria-labelledby="test-heading" className="section">
        <div className="section-head">
          <h2 id="test-heading">Evaluation results</h2>
          <span className="badge badge-test">Reported evaluation</span>
        </div>
        <div className="metric-grid">
          {metricDefinitions.map((def) => (
            <MetricCard key={def.key} definition={def} value={reportedMetrics[def.key]} emphasis={def.key === 'recall'} />
          ))}
        </div>
      </section>

      <div className="two-col">
        <section className="card">
          <h2>Confusion matrix</h2>
          {cm ? (
            <ConfusionMatrix matrix={cm} />
          ) : (
            <p className="muted">Not provided. Confusion matrix counts were not included in the reported results.</p>
          )}
        </section>

        <section className="card">
          <h2>Reading these results</h2>
          <p>
            Recall of <strong>{pct(reportedMetrics.recall)}</strong> and precision of{' '}
            <strong>{pct(reportedMetrics.precision)}</strong> were reported for this model.
          </p>
          <p className="muted">
            About 91% of patients in this dataset are non-diabetic, so accuracy alone can overstate performance on
            diabetic patients. A negative result should never be read as ruling out diabetes.
          </p>
        </section>
      </div>

      <div className="two-col">
        <section className="card">
          <h2>Class distribution</h2>
          <ClassDistribution
            rows={[
              { label: 'Raw dataset', counts: dataset.raw.class_counts },
              { label: 'After de-duplication', counts: dataset.class_counts_after_deduplication },
              { label: 'Training (fit)', counts: split.class_counts.training },
              { label: 'Validation', counts: split.class_counts.validation },
              { label: 'Test', counts: split.class_counts.test },
            ]}
          />
        </section>

        <section className="card">
          <h2>Data preparation</h2>
          <dl className="kv">
            <div className="kv-row">
              <dt>Raw rows</dt>
              <dd>{num(dataset.raw.rows)}</dd>
            </div>
            <div className="kv-row">
              <dt>Exact duplicates removed</dt>
              <dd>{num(dataset.duplicates_removed)}</dd>
            </div>
            <div className="kv-row">
              <dt>Missing values</dt>
              <dd>{Object.values(dataset.raw.missing_values).reduce((a, b) => a + b, 0) === 0 ? 'None' : 'Present'}</dd>
            </div>
            <div className="kv-row">
              <dt>Split</dt>
              <dd>
                Stratified {Math.round((1 - split.test_size) * 100)}/{Math.round(split.test_size * 100)}, seed{' '}
                {split.random_seed}
              </dd>
            </div>
            <div className="kv-row">
              <dt>Categorical encoding</dt>
              <dd>One-hot (gender, smoking_history)</dd>
            </div>
          </dl>
          <p className="muted small">
            "No Info" in smoking history is the dataset's own missing-value marker and is kept as a separate category.
          </p>
        </section>
      </div>
    </>
  );
}
