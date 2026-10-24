export default function Notice({ tone = 'info', title, children }) {
  return (
    <div className={`notice notice-${tone}`} role={tone === 'danger' ? 'alert' : 'note'}>
      {title && <strong className="notice-title">{title}</strong>}
      <div>{children}</div>
    </div>
  );
}

export function ClinicalDisclaimer() {
  return (
    <Notice tone="info" title="Research use only">
      Outputs are machine-learning estimates from a machine-learning model evaluated on a public research dataset. The
      model has not been clinically validated and must not be used for diagnosis or treatment decisions.
    </Notice>
  );
}

export function Loading({ label = 'Loading…' }) {
  return (
    <div className="card loading" role="status">
      <span className="spinner" aria-hidden="true" />
      {label}
    </div>
  );
}
