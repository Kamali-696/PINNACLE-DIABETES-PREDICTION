import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { postPrediction } from '../api/client.js';
import Notice, { ClinicalDisclaimer } from '../components/Notice.jsx';
import { featureSchema, toPayload, validateForm } from '../data/featureSchema.js';

const emptyForm = Object.fromEntries(featureSchema.map((f) => [f.name, '']));

function Field({ field, value, error, onChange }) {
  const id = `f-${field.name}`;
  const errorId = `${id}-error`;
  const common = {
    id,
    name: field.name,
    'aria-invalid': Boolean(error),
    'aria-describedby': error ? errorId : undefined,
  };

  let control;
  if (field.type === 'select') {
    control = (
      <select {...common} value={value} onChange={(e) => onChange(e.target.value)}>
        <option value="" disabled>
          Select…
        </option>
        {field.options.map((o) => (
          <option key={o.value} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>
    );
  } else if (field.type === 'binary') {
    control = (
      <div className="segmented" role="radiogroup" aria-labelledby={`${id}-label`} aria-describedby={common['aria-describedby']}>
        {field.options.map((o) => (
          <label key={o.value} className={`segment${String(value) === String(o.value) ? ' selected' : ''}`}>
            <input
              type="radio"
              name={field.name}
              value={o.value}
              checked={String(value) === String(o.value)}
              onChange={(e) => onChange(e.target.value)}
            />
            {o.label}
          </label>
        ))}
      </div>
    );
  } else {
    control = (
      <input
        {...common}
        type="number"
        inputMode="decimal"
        min={field.min}
        max={field.max}
        step={field.step}
        value={value}
        onChange={(e) => onChange(e.target.value)}
      />
    );
  }

  return (
    <div className={`field${error ? ' has-error' : ''}`}>
      <label className="field-label" id={`${id}-label`} htmlFor={field.type === 'binary' ? undefined : id}>
        {field.label}
        {field.unit && <span className="field-unit"> ({field.unit})</span>}
      </label>
      {control}
      {field.type === 'number' && !error && (
        <span className="field-hint">
          {field.min}–{field.max}
        </span>
      )}
      {error && (
        <span className="field-error" id={errorId}>
          {error}
        </span>
      )}
    </div>
  );
}

export default function Predict() {
  const navigate = useNavigate();
  const [values, setValues] = useState(emptyForm);
  const [errors, setErrors] = useState({});
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState(null);

  const update = (name) => (value) => {
    setValues((v) => ({ ...v, [name]: value }));
    setErrors((e) => ({ ...e, [name]: undefined }));
  };

  async function handleSubmit(e) {
    e.preventDefault();
    setSubmitError(null);
    const found = validateForm(values);
    setErrors(found);
    if (Object.keys(found).length > 0) return;

    setSubmitting(true);
    try {
      const payload = toPayload(values);
      const prediction = await postPrediction(payload);
      navigate('/result', { state: { input: payload, prediction } });
    } catch (err) {
      setErrors(err.fieldErrors || {});
      setSubmitError(err.message);
    } finally {
      setSubmitting(false);
    }
  }

  const errorCount = Object.values(errors).filter(Boolean).length;

  return (
    <>
      <div className="page-header">
        <div>
          <h1>Diabetes Risk Prediction</h1>
          <p className="page-sub">Enter patient details to request a model-estimated risk from the Diabetes Prediction Model.</p>
        </div>
      </div>

      <form className="card form" onSubmit={handleSubmit} noValidate>
        <div className="form-grid">
          {featureSchema.map((field) => (
            <Field
              key={field.name}
              field={field}
              value={values[field.name]}
              error={errors[field.name]}
              onChange={update(field.name)}
            />
          ))}
        </div>

        {errorCount > 0 && !submitError && (
          <Notice tone="warning">Please correct {errorCount === 1 ? 'the highlighted field' : `${errorCount} highlighted fields`}.</Notice>
        )}
        {submitError && (
          <Notice tone="danger" title="No prediction was produced">
            {submitError}
          </Notice>
        )}

        <div className="form-actions">
          <button
            type="button"
            className="btn btn-ghost"
            onClick={() => {
              setValues(emptyForm);
              setErrors({});
              setSubmitError(null);
            }}
            disabled={submitting}
          >
            Clear
          </button>
          <button type="submit" className="btn btn-primary" disabled={submitting}>
            {submitting && <span className="spinner spinner-light" aria-hidden="true" />}
            {submitting ? 'Requesting estimate…' : 'Estimate risk'}
          </button>
        </div>
      </form>

      <ClinicalDisclaimer />
    </>
  );
}
