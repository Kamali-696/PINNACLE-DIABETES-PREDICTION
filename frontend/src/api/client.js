const API_BASE = import.meta.env.VITE_API_BASE_URL || '/api';

export class ApiError extends Error {
  constructor(message, status, fieldErrors = {}) {
    super(message);
    this.status = status;
    this.fieldErrors = fieldErrors;
  }
}

// FastAPI 422 responses carry a list of { loc, msg }; map them to field names.
function parseValidationErrors(detail) {
  const fieldErrors = {};
  for (const item of detail) {
    const field = item.loc?.[item.loc.length - 1];
    if (typeof field === 'string') fieldErrors[field] = item.msg;
  }
  return fieldErrors;
}

async function request(path, options = {}) {
  let res;
  try {
    res = await fetch(`${API_BASE}${path}`, {
      headers: { 'Content-Type': 'application/json' },
      ...options,
    });
  } catch {
    throw new ApiError('Cannot reach the API server. Is the backend running on port 8000?', 0);
  }
  const body = await res.json().catch(() => null);
  if (!res.ok) {
    const detail = body?.detail;
    if (Array.isArray(detail)) {
      throw new ApiError('The server rejected some inputs.', res.status, parseValidationErrors(detail));
    }
    throw new ApiError(typeof detail === 'string' ? detail : `Request failed with status ${res.status}`, res.status);
  }
  return body;
}

export const getHealth = () => request('/health');
export const getMetrics = () => request('/metrics');
export const postPrediction = (features) =>
  request('/predict', { method: 'POST', body: JSON.stringify(features) });
