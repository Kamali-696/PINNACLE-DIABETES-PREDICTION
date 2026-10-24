import { Link, useLocation } from 'react-router-dom';
import { getMetrics } from '../api/client.js';
import Notice from '../components/Notice.jsx';
import { displayValue, featureSchema } from '../data/featureSchema.js';
import useApi from '../hooks/useApi.js';
import { pct } from '../lib/format.js';

// Displays only what POST /api/predict returned; nothing is computed client-side.
export default function Result() {
  const { state } = useLocation();
  const prediction = state?.prediction;

  if (!prediction) {
    return (
      <>
        <div className="page-header">
          <div>
            <h1>Prediction Result</h1>
            <p className="page-sub">Model-estimated diabetes risk for a submitted patient.</p>
          </div>
        </div>
        <div className="card empty-state">
          <h2>No prediction to display</h2>
          <p className="muted">Results appear here after the backend returns a prediction for a submitted form.</p>
          <Link to="/predict" className="btn btn-primary">
            Go to risk prediction
          </Link>
        </div>
      </>
    );
  }

  return <ResultView input={state.input} prediction={prediction} />;
}

function ResultView({ input, prediction }) {
  const { data: report } = useApi(getMetrics);
  const { probability, predicted_class: predictedClass, threshold, label, warnings } = prediction;
  const positive = predictedClass === 1;
  const testRecall = report?.metrics?.test?.recall;

  return (
    <>
      <div className="page-header">
        <div>
          <h1>Prediction Result</h1>
          <p className="page-sub">
            Returned by the Diabetes Prediction Model (trained {new Date(prediction.model_trained_at).toLocaleString()}).
          </p>
        </div>
        <Link to="/predict" className="btn btn-ghost">
          New prediction
        </Link>
      </div>

      <div className="result-grid">
        <section className={`card result-card ${positive ? 'result-positive' : 'result-negative'}`} aria-live="polite">
          <div className="metric-label">Model classification</div>
          <div className="result-label" data-testid="result-label">
            <span className={`result-icon ${positive ? 'icon-pos' : 'icon-neg'}`} aria-hidden="true">
              {positive ? '!' : '✓'}
            </span>
            {label}
          </div>

          <div className="metric-label" style={{ marginTop: 18 }}>
            Model-estimated probability of diabetes
          </div>
          <div className="result-value" data-testid="result-probability">
            {pct(probability, 1)}
          </div>
          <div className="meter meter-lg" aria-hidden="true">
            <div className="meter-fill" style={{ width: `${probability * 100}%` }} />
            <div className="meter-threshold" style={{ left: `${threshold * 100}%` }} />
          </div>
          <p className="muted small">
            Classified as diabetic when the probability is at least {pct(threshold, 0)} (marker on the bar).
          </p>
        </section>

        <section className="card">
          <h2>Submitted inputs</h2>
          <dl className="kv">
            {featureSchema.map((f) => (
              <div key={f.name} className="kv-row">
                <dt>{f.label}</dt>
                <dd>{displayValue(f, input?.[f.name])}</dd>
              </div>
            ))}
          </dl>
        </section>
      </div>

      {warnings?.length > 0 && (
        <Notice tone="warning" title="Inputs outside the training data">
          <ul className="plain-list">
            {warnings.map((w) => (
              <li key={w}>{w}</li>
            ))}
          </ul>
        </Notice>
      )}

      <Notice tone="warning" title="This is a machine-learning estimate, not a medical diagnosis">
        {prediction.disclaimer}
        {typeof testRecall === 'number' && (
          <>
            {' '}
            On the held-out test set the model identified {pct(testRecall)} of diabetic patients, so a "not indicated"
            result does not rule out diabetes.
          </>
        )}{' '}
        Diagnosis requires clinical assessment and laboratory testing by a qualified professional.
      </Notice>
    </>
  );
}
