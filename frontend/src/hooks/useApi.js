import { useEffect, useState } from 'react';

// Runs `fn` once on mount and tracks { data, error, loading }.
export default function useApi(fn) {
  const [state, setState] = useState({ data: null, error: null, loading: true });

  useEffect(() => {
    let cancelled = false;
    fn()
      .then((data) => !cancelled && setState({ data, error: null, loading: false }))
      .catch((error) => !cancelled && setState({ data: null, error, loading: false }));
    return () => {
      cancelled = true;
    };
  }, [fn]);

  return state;
}
