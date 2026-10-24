import { Navigate, Route, Routes } from 'react-router-dom';
import Layout from './components/Layout.jsx';
import Dashboard from './pages/Dashboard.jsx';
import Predict from './pages/Predict.jsx';
import Result from './pages/Result.jsx';
import Comparison from './pages/Comparison.jsx';

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<Dashboard />} />
        <Route path="predict" element={<Predict />} />
        <Route path="result" element={<Result />} />
        <Route path="comparison" element={<Comparison />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  );
}
