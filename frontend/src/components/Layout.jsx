import { useEffect, useState } from 'react';
import { NavLink, Outlet } from 'react-router-dom';
import { getHealth } from '../api/client.js';

const navItems = [
  { to: '/', label: 'Dashboard', end: true },
  { to: '/predict', label: 'Risk Prediction' },
  { to: '/result', label: 'Prediction Result' },
  { to: '/comparison', label: 'Model Comparison' },
];

function ApiStatus() {
  const [state, setState] = useState({ phase: 'loading' });

  useEffect(() => {
    let cancelled = false;
    getHealth()
      .then((h) => !cancelled && setState({ phase: 'ok', modelLoaded: h.model_loaded }))
      .catch(() => !cancelled && setState({ phase: 'offline' }));
    return () => {
      cancelled = true;
    };
  }, []);

  let tone = 'neutral';
  let text = 'Checking API…';
  if (state.phase === 'offline') {
    tone = 'danger';
    text = 'API offline';
  } else if (state.phase === 'ok') {
    tone = state.modelLoaded ? 'success' : 'warning';
    text = state.modelLoaded ? 'API online · model loaded' : 'API online · model not loaded';
  }

  return (
    <span className={`status-pill status-${tone}`}>
      <span className="status-dot" aria-hidden="true" />
      {text}
    </span>
  );
}

export default function Layout() {
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark" aria-hidden="true">
            <svg viewBox="0 0 32 32" width="22" height="22">
              <path d="M16 5c-3.8 5.4-6.5 9.1-6.5 12.5a6.5 6.5 0 0 0 13 0C22.5 14.1 19.8 10.4 16 5z" fill="currentColor" />
            </svg>
          </div>
          <div>
            <div className="brand-title">Diabetes Prediction</div>
            <div className="brand-sub">Research prediction system</div>
          </div>
        </div>
        <nav className="nav">
          {navItems.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.end}
              className={({ isActive }) => `nav-link${isActive ? ' active' : ''}`}
            >
              {item.label}
            </NavLink>
          ))}
        </nav>
        <p className="sidebar-foot">
          Research prototype. Not a medical device and not clinically validated.
        </p>
      </aside>

      <div className="main">
        <header className="topbar">
          <span className="topbar-title">Diabetes Prediction Model</span>
          <ApiStatus />
        </header>
        <main className="content">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
