import React, { useEffect, useState } from 'react';
import { useLocation, Link } from 'react-router-dom';
import {
  Cpu,
  RefreshCw,
  ChevronRight,
  Database,
} from 'lucide-react';
import { healthAPI } from '../../api/client';
import { useSettingsStore } from '../../store/settingsStore';

const routeTitleMap: Record<string, { title: string; category: string }> = {
  '/': { title: 'Overview', category: 'Dashboard' },
  '/documents': { title: 'Documents', category: 'Knowledge Base' },
  '/playground': { title: 'RAG Playground', category: 'Testing' },
  '/evaluation': { title: 'Evaluation', category: 'Hallucination Analysis' },
  '/experiments': { title: 'Experiment Lab', category: 'Benchmarking' },
  '/analytics': { title: 'Analytics', category: 'Metrics & Trends' },
  '/settings': { title: 'Settings', category: 'Configuration' },
};

export const TopBar: React.FC = () => {
  const location = useLocation();
  const { llmBackend, llmModel, embeddingModel } = useSettingsStore();

  const [apiStatus, setApiStatus] = useState<'checking' | 'connected' | 'offline'>('checking');
  const [apiVersion, setApiVersion] = useState<string>('');
  const [isRefreshing, setIsRefreshing] = useState(false);

  const checkHealth = async () => {
    setIsRefreshing(true);
    try {
      const res = await healthAPI.check();
      setApiStatus(res.status === 'ok' || res.status === 'healthy' ? 'connected' : 'connected');
      if (res.version) setApiVersion(res.version);
    } catch {
      setApiStatus('offline');
    } finally {
      setIsRefreshing(false);
    }
  };

  useEffect(() => {
    checkHealth();
    // Periodic check every 30 seconds
    const interval = setInterval(checkHealth, 30_000);
    return () => clearInterval(interval);
  }, []);

  const currentRoute = routeTitleMap[location.pathname] || {
    title: 'Dashboard',
    category: 'Suite',
  };

  return (
    <header
      className="sticky top-0 z-20 flex items-center justify-between px-6 border-b backdrop-blur-md transition-colors"
      style={{
        height: 'var(--topbar-height)',
        backgroundColor: 'rgba(10, 10, 15, 0.85)',
        borderColor: 'var(--color-border)',
      }}
    >
      {/* ── Left Breadcrumb & Page Info ── */}
      <div className="flex items-center gap-2.5">
        <Link
          to="/"
          className="text-xs font-medium hover:text-[var(--color-accent-primary)] transition-colors flex items-center gap-1"
          style={{ color: 'var(--color-text-tertiary)' }}
        >
          <span>RAG Eval</span>
        </Link>

        <ChevronRight
          className="w-3.5 h-3.5"
          style={{ color: 'var(--color-text-muted)' }}
        />

        <span
          className="text-xs font-medium"
          style={{ color: 'var(--color-text-secondary)' }}
        >
          {currentRoute.category}
        </span>

        <ChevronRight
          className="w-3.5 h-3.5"
          style={{ color: 'var(--color-text-muted)' }}
        />

        <h1
          className="text-sm font-semibold tracking-tight"
          style={{ color: 'var(--color-text-primary)' }}
        >
          {currentRoute.title}
        </h1>
      </div>

      {/* ── Right Status & Model Info ── */}
      <div className="flex items-center gap-3">
        {/* Active Embedding Model pill */}
        <div
          className="hidden md:flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-mono border"
          style={{
            backgroundColor: 'var(--color-bg-secondary)',
            borderColor: 'var(--color-border)',
            color: 'var(--color-text-secondary)',
          }}
          title={`Embedding: ${embeddingModel}`}
        >
          <Database className="w-3.5 h-3.5 text-teal-400" />
          <span className="truncate max-w-[140px]">
            {embeddingModel.split('/').pop() || embeddingModel}
          </span>
        </div>

        {/* Current LLM Badge */}
        <div
          className="flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-mono border"
          style={{
            backgroundColor: 'rgba(124, 106, 247, 0.08)',
            borderColor: 'rgba(124, 106, 247, 0.25)',
            color: 'var(--color-accent-secondary)',
          }}
          title={`Backend: ${llmBackend} | Model: ${llmModel}`}
        >
          <Cpu className="w-3.5 h-3.5" />
          <span className="font-semibold capitalize">{llmBackend}:</span>
          <span className="truncate max-w-[130px]">{llmModel}</span>
        </div>

        {/* Backend API Health Status */}
        <button
          type="button"
          onClick={checkHealth}
          disabled={isRefreshing}
          className="flex items-center gap-2 px-2.5 py-1 rounded-md text-xs font-medium border transition-all cursor-pointer hover:border-[var(--color-border-active)]"
          style={{
            backgroundColor:
              apiStatus === 'connected'
                ? 'rgba(52, 211, 153, 0.08)'
                : apiStatus === 'offline'
                ? 'rgba(248, 113, 113, 0.08)'
                : 'var(--color-bg-secondary)',
            borderColor:
              apiStatus === 'connected'
                ? 'rgba(52, 211, 153, 0.2)'
                : apiStatus === 'offline'
                ? 'rgba(248, 113, 113, 0.2)'
                : 'var(--color-border)',
            color:
              apiStatus === 'connected'
                ? 'var(--color-emerald)'
                : apiStatus === 'offline'
                ? 'var(--color-rose)'
                : 'var(--color-text-secondary)',
          }}
          title={`API Status: ${apiStatus}. Click to refresh health check.`}
        >
          <span className="relative flex h-2 w-2">
            {apiStatus === 'connected' && (
              <span
                className="animate-ping absolute inline-flex h-full w-full rounded-full opacity-75"
                style={{ backgroundColor: 'var(--color-emerald)' }}
              />
            )}
            <span
              className="relative inline-flex rounded-full h-2 w-2"
              style={{
                backgroundColor:
                  apiStatus === 'connected'
                    ? 'var(--color-emerald)'
                    : apiStatus === 'offline'
                    ? 'var(--color-rose)'
                    : 'var(--color-amber)',
              }}
            />
          </span>

          <span className="hidden sm:inline">
            {apiStatus === 'checking'
              ? 'Checking API...'
              : apiStatus === 'connected'
              ? `API Connected${apiVersion ? ` (${apiVersion})` : ''}`
              : 'API Offline'}
          </span>

          <RefreshCw
            className={`w-3 h-3 ${isRefreshing ? 'animate-spin' : 'opacity-60 hover:opacity-100'}`}
          />
        </button>
      </div>
    </header>
  );
};

export default TopBar;
