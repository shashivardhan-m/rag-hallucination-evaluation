import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { Layout } from './components/layout/Layout';
import { ErrorBoundary } from './components/ui/ErrorBoundary';
import { Dashboard } from './pages/Dashboard';
import { Documents } from './pages/Documents';
import { Playground } from './pages/Playground';
import { Evaluation } from './pages/Evaluation';
import { ExperimentLab } from './pages/ExperimentLab';
import { Analytics } from './pages/Analytics';
import { Settings } from './pages/Settings';


export const App: React.FC = () => {
  return (
    <ErrorBoundary>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Layout />}>
            <Route index element={<Dashboard />} />
            <Route path="documents" element={<Documents />} />
            <Route path="playground" element={<Playground />} />
            <Route path="evaluation" element={<Evaluation />} />
            <Route path="experiments" element={<ExperimentLab />} />
            <Route path="analytics" element={<Analytics />} />
            <Route path="settings" element={<Settings />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </ErrorBoundary>
  );
};

export default App;
