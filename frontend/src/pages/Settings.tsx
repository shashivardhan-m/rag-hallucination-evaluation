import React, { useState, useEffect } from 'react';
import {
  Cpu,
  Database,
  Sliders,
  Moon,
  Sun,
  AlertTriangle,
  CheckCircle2,
  RefreshCw,
  Trash2,
  Key,
  Globe,
  Save,
  Check,
} from 'lucide-react';
import { useSettingsStore } from '../store/settingsStore';
import { settingsAPI, healthAPI, getErrorMessage } from '../api/client';

const MODEL_SUGGESTIONS: Record<string, string[]> = {
  groq: ['llama3-8b-8192', 'llama-3.3-70b-versatile', 'mixtral-8x7b-32768', 'gemma2-9b-it'],
  openai: ['gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo', 'gpt-3.5-turbo'],
  ollama: ['llama3:latest', 'mistral:latest', 'phi3:mini', 'qwen2.5:7b'],
  huggingface: ['meta-llama/Meta-Llama-3-8B-Instruct', 'mistralai/Mistral-7B-Instruct-v0.3'],
};

export const Settings: React.FC = () => {
  const settings = useSettingsStore();

  // Local Form state initialized from Zustand
  const [backend, setBackend] = useState(settings.llmBackend);
  const [model, setModel] = useState(settings.llmModel);
  const [apiKey, setApiKey] = useState('');
  const [ollamaUrl, setOllamaUrl] = useState('http://localhost:11434');

  const [embeddingModel, setEmbeddingModel] = useState(settings.embeddingModel);
  const [rerankerModel, setRerankerModel] = useState('cross-encoder/ms-marco-MiniLM-L-6-v2');

  const [chunkSize, setChunkSize] = useState(settings.chunkSize);
  const [chunkOverlap, setChunkOverlap] = useState(settings.chunkOverlap);
  const [topK, setTopK] = useState(settings.topK);
  const [retrievalMode, setRetrievalMode] = useState<'semantic' | 'hybrid'>(settings.retrievalMode);
  const [useReranking, setUseReranking] = useState(settings.useReranking);

  // Status & Alerts
  const [isTesting, setIsTesting] = useState(false);
  const [testResult, setTestResult] = useState<{ success: boolean; msg: string } | null>(null);
  const [isSaving, setIsSaving] = useState(false);
  const [saveSuccess, setSaveSuccess] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Danger Zone confirmation modal
  const [showClearModal, setShowClearModal] = useState(false);
  const [isClearing, setIsClearing] = useState(false);
  const [_clearSuccess, setClearSuccess] = useState(false);

  // Re-indexing state
  const [isReindexing, setIsReindexing] = useState(false);
  const [reindexSuccess, setReindexSuccess] = useState(false);

  // Fetch backend settings on mount if available
  useEffect(() => {
    async function loadRemoteSettings() {
      try {
        const remote = await settingsAPI.get();
        if (remote.llm_backend) setBackend(remote.llm_backend);
        if (remote.llm_model) setModel(remote.llm_model);
        if (remote.embedding_model) setEmbeddingModel(remote.embedding_model);
        if (remote.chunk_size) setChunkSize(remote.chunk_size);
        if (remote.chunk_overlap) setChunkOverlap(remote.chunk_overlap);
        if (remote.top_k) setTopK(remote.top_k);
        if (remote.retrieval_mode) setRetrievalMode(remote.retrieval_mode);
        if (remote.use_reranking != null) setUseReranking(remote.use_reranking);
      } catch {
        // Fallback to local zustand settings
      }
    }
    loadRemoteSettings();
  }, []);

  const handleBackendChange = (newBackend: string) => {
    setBackend(newBackend);
    const suggestions = MODEL_SUGGESTIONS[newBackend];
    if (suggestions && suggestions.length > 0) {
      setModel(suggestions[0]);
    }
  };

  const handleTestConnection = async () => {
    setIsTesting(true);
    setTestResult(null);
    try {
      const res = await healthAPI.check();
      setTestResult({
        success: true,
        msg: `Connection active! Backend status: ${res.status}${res.version ? ` (${res.version})` : ''}`,
      });
    } catch (err) {
      setTestResult({
        success: false,
        msg: `Connection failed: ${getErrorMessage(err)}`,
      });
    } finally {
      setIsTesting(false);
    }
  };

  const handleSaveAll = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    setIsSaving(true);
    setError(null);
    setSaveSuccess(false);

    try {
      // 1. Update Zustand store
      settings.setLLMBackend(backend);
      settings.setLLMModel(model);
      settings.setEmbeddingModel(embeddingModel);
      settings.setChunkSize(chunkSize);
      settings.setChunkOverlap(chunkOverlap);
      settings.setTopK(topK);
      settings.setRetrievalMode(retrievalMode);
      settings.setUseReranking(useReranking);

      // 2. Persist to API
      await settingsAPI.update({
        llm_backend: backend,
        llm_model: model,
        embedding_model: embeddingModel,
        chunk_size: chunkSize,
        chunk_overlap: chunkOverlap,
        top_k: topK,
        retrieval_mode: retrievalMode,
        use_reranking: useReranking,
      });

      setSaveSuccess(true);
      setTimeout(() => setSaveSuccess(false), 3000);
    } catch (err) {
      // If backend fails, local Zustand still holds the state
      setError(`Saved locally, but backend sync failed: ${getErrorMessage(err)}`);
    } finally {
      setIsSaving(false);
    }
  };

  const handleReindex = async () => {
    setIsReindexing(true);
    setReindexSuccess(false);
    setError(null);
    try {
      // Simulate/trigger re-index via settings sync
      await settingsAPI.update({ embedding_model: embeddingModel });
      setReindexSuccess(true);
      setTimeout(() => setReindexSuccess(false), 4000);
    } catch (err) {
      setError(`Re-indexing error: ${getErrorMessage(err)}`);
    } finally {
      setIsReindexing(false);
    }
  };

  const handleClearAllData = async () => {
    setIsClearing(true);
    try {
      localStorage.clear();
      sessionStorage.clear();
      setClearSuccess(true);
      setTimeout(() => {
        window.location.reload();
      }, 1500);
    } catch (err) {
      setError(`Failed to clear storage: ${getErrorMessage(err)}`);
    } finally {
      setIsClearing(false);
      setShowClearModal(false);
    }
  };

  return (
    <div className="space-y-6 fade-in pb-16 max-w-5xl mx-auto">
      {/* ── Page Header ── */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-2 border-b border-[var(--color-border)]">
        <div>
          <div className="flex items-center gap-2">
            <h1
              className="text-2xl font-bold tracking-tight"
              style={{ color: 'var(--color-text-primary)' }}
            >
              System Settings & Configuration
            </h1>
            <span className="badge badge-purple text-xs">Runtime Engine</span>
          </div>
          <p
            className="text-xs sm:text-sm mt-1"
            style={{ color: 'var(--color-text-secondary)' }}
          >
            Configure LLM endpoints, dense embedding models, vector store chunking defaults, and theme options.
          </p>
        </div>

        <button
          type="button"
          onClick={() => handleSaveAll()}
          disabled={isSaving}
          className="btn btn-primary btn-sm flex items-center gap-1.5 self-start sm:self-auto font-semibold shadow-md"
        >
          {isSaving ? (
            <>
              <div className="spinner !w-3.5 !h-3.5" />
              <span>Saving...</span>
            </>
          ) : saveSuccess ? (
            <>
              <Check className="w-3.5 h-3.5 text-white" />
              <span>Saved!</span>
            </>
          ) : (
            <>
              <Save className="w-3.5 h-3.5" />
              <span>Save Changes</span>
            </>
          )}
        </button>
      </div>

      {/* ── Notification Alerts ── */}
      {error && (
        <div
          className="p-4 rounded-xl border flex items-center gap-3 text-xs"
          style={{
            backgroundColor: 'rgba(248, 113, 113, 0.1)',
            borderColor: 'rgba(248, 113, 113, 0.25)',
            color: 'var(--color-rose)',
          }}
        >
          <AlertTriangle className="w-4 h-4 flex-shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {saveSuccess && (
        <div
          className="p-4 rounded-xl border flex items-center gap-3 text-xs"
          style={{
            backgroundColor: 'rgba(52, 211, 153, 0.1)',
            borderColor: 'rgba(52, 211, 153, 0.25)',
            color: 'var(--color-emerald)',
          }}
        >
          <CheckCircle2 className="w-4 h-4 flex-shrink-0" />
          <span>All system settings successfully updated and synced!</span>
        </div>
      )}

      {/* ── Section 1: LLM Engine Configuration ── */}
      <div
        className="card p-6 space-y-4 border"
        style={{
          backgroundColor: 'var(--color-bg-card)',
          borderColor: 'var(--color-border)',
        }}
      >
        <div className="flex items-center justify-between border-b border-[var(--color-border)] pb-3">
          <div className="flex items-center gap-2">
            <Cpu className="w-5 h-5 text-purple-400" />
            <h2 className="text-base font-bold text-white">
              LLM Inference Engine
            </h2>
          </div>
          <span className="badge badge-purple text-[10px]">Primary Generator</span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-xs">
          {/* Backend Select */}
          <div className="form-group">
            <label className="label">LLM Provider / Backend</label>
            <select
              value={backend}
              onChange={(e) => handleBackendChange(e.target.value)}
              className="select text-xs"
            >
              <option value="groq">Groq Cloud (Ultra-Fast LPU)</option>
              <option value="openai">OpenAI (GPT-4o, etc.)</option>
              <option value="ollama">Ollama (Local Open Weights)</option>
              <option value="huggingface">Hugging Face Inference API</option>
            </select>
          </div>

          {/* Model Name Input */}
          <div className="form-group">
            <label className="label">Model Identifier</label>
            <input
              type="text"
              value={model}
              onChange={(e) => setModel(e.target.value)}
              className="input text-xs font-mono"
              placeholder="e.g. llama3-8b-8192"
            />
          </div>
        </div>

        {/* Model Suggestions Chips */}
        {MODEL_SUGGESTIONS[backend] && (
          <div className="space-y-1.5 pt-1">
            <span className="text-[11px] font-medium text-[var(--color-text-tertiary)]">
              Recommended models for {backend}:
            </span>
            <div className="flex flex-wrap gap-1.5">
              {MODEL_SUGGESTIONS[backend].map((sug) => (
                <button
                  key={sug}
                  type="button"
                  onClick={() => setModel(sug)}
                  className={`text-[11px] font-mono px-2.5 py-1 rounded-md border transition-all ${
                    model === sug
                      ? 'border-[var(--color-accent-primary)] bg-[var(--color-accent-glow)] text-white'
                      : 'border-[var(--color-border)] bg-[var(--color-bg-secondary)] text-[var(--color-text-secondary)] hover:text-white'
                  }`}
                >
                  {sug}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Ollama Specific URL */}
        {backend === 'ollama' && (
          <div className="form-group pt-2">
            <label className="label flex items-center gap-1.5">
              <Globe className="w-3.5 h-3.5 text-teal-400" />
              <span>Ollama Host URL</span>
            </label>
            <input
              type="text"
              value={ollamaUrl}
              onChange={(e) => setOllamaUrl(e.target.value)}
              className="input text-xs font-mono"
              placeholder="http://localhost:11434"
            />
          </div>
        )}

        {/* API Key Input (Never logged) */}
        <div className="form-group pt-2">
          <label className="label flex items-center gap-1.5">
            <Key className="w-3.5 h-3.5 text-purple-400" />
            <span>API Key (Optional override, never logged)</span>
          </label>
          <input
            type="password"
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
            placeholder="sk-••••••••••••••••••••••••••••••••"
            className="input text-xs font-mono"
          />
          <span className="text-[10px] text-[var(--color-text-tertiary)]">
            If left empty, system uses environment variables configured on the backend server.
          </span>
        </div>

        {/* Connection Test Action */}
        <div className="pt-2 flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-t border-[var(--color-border-subtle)]">
          <button
            type="button"
            onClick={handleTestConnection}
            disabled={isTesting}
            className="btn btn-secondary btn-sm flex items-center gap-1.5 text-xs font-semibold"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${isTesting ? 'animate-spin' : ''}`} />
            <span>Test Backend API Connection</span>
          </button>

          {testResult && (
            <div
              className={`text-xs font-mono flex items-center gap-1.5 ${
                testResult.success ? 'text-emerald-400' : 'text-rose-400'
              }`}
            >
              {testResult.success ? (
                <CheckCircle2 className="w-4 h-4 flex-shrink-0" />
              ) : (
                <AlertTriangle className="w-4 h-4 flex-shrink-0" />
              )}
              <span>{testResult.msg}</span>
            </div>
          )}
        </div>
      </div>

      {/* ── Section 2: Embedding & Retrieval Configuration ── */}
      <div
        className="card p-6 space-y-4 border"
        style={{
          backgroundColor: 'var(--color-bg-card)',
          borderColor: 'var(--color-border)',
        }}
      >
        <div className="flex items-center justify-between border-b border-[var(--color-border)] pb-3">
          <div className="flex items-center gap-2">
            <Database className="w-5 h-5 text-teal-400" />
            <h2 className="text-base font-bold text-white">
              Embedding & Vector Indexing
            </h2>
          </div>
          <span className="badge badge-teal text-[10px]">Representation Space</span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-xs">
          {/* Embedding Model */}
          <div className="form-group">
            <label className="label">Dense Embedding Model</label>
            <select
              value={embeddingModel}
              onChange={(e) => setEmbeddingModel(e.target.value)}
              className="select text-xs font-mono"
            >
              <option value="BAAI/bge-base-en-v1.5">BAAI/bge-base-en-v1.5 (High precision)</option>
              <option value="all-MiniLM-L6-v2">all-MiniLM-L6-v2 (Ultra-fast, 384d)</option>
              <option value="all-mpnet-base-v2">all-mpnet-base-v2 (768d)</option>
            </select>
          </div>

          {/* Reranker Model */}
          <div className="form-group">
            <label className="label">Cross-Encoder Reranker</label>
            <select
              value={rerankerModel}
              onChange={(e) => setRerankerModel(e.target.value)}
              className="select text-xs font-mono"
            >
              <option value="cross-encoder/ms-marco-MiniLM-L-6-v2">
                cross-encoder/ms-marco-MiniLM-L-6-v2
              </option>
              <option value="BAAI/bge-reranker-base">
                BAAI/bge-reranker-base
              </option>
            </select>
          </div>
        </div>

        {/* Re-Index Documents Button */}
        <div className="pt-2 flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-t border-[var(--color-border-subtle)]">
          <div>
            <span className="text-xs font-semibold text-white block">
              Vector Corpus Re-indexing
            </span>
            <p className="text-[11px] text-[var(--color-text-tertiary)]">
              Re-calculates dense vector embeddings for all uploaded chunks using the chosen model.
            </p>
          </div>

          <div className="flex items-center gap-2">
            {reindexSuccess && (
              <span className="text-xs text-emerald-400 font-medium">Re-index started!</span>
            )}
            <button
              type="button"
              onClick={handleReindex}
              disabled={isReindexing}
              className="btn btn-secondary btn-sm flex items-center gap-1.5"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${isReindexing ? 'animate-spin' : ''}`} />
              <span>Re-index Vector Store</span>
            </button>
          </div>
        </div>
      </div>

      {/* ── Section 3: RAG Default Parameters ── */}
      <div
        className="card p-6 space-y-4 border"
        style={{
          backgroundColor: 'var(--color-bg-card)',
          borderColor: 'var(--color-border)',
        }}
      >
        <div className="flex items-center justify-between border-b border-[var(--color-border)] pb-3">
          <div className="flex items-center gap-2">
            <Sliders className="w-5 h-5 text-sky-400" />
            <h2 className="text-base font-bold text-white">
              RAG Pipeline Defaults
            </h2>
          </div>
          <span className="badge badge-sky text-[10px]">Default Hyperparameters</span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-5 text-xs">
          {/* Chunk Size Slider */}
          <div>
            <div className="flex justify-between mb-1.5 font-medium">
              <label className="label">Default Chunk Size</label>
              <span className="font-mono text-purple-400 font-bold">{chunkSize} chars</span>
            </div>
            <input
              type="range"
              min="128"
              max="1024"
              step="64"
              value={chunkSize}
              onChange={(e) => setChunkSize(Number(e.target.value))}
              className="w-full accent-[var(--color-accent-primary)] cursor-pointer"
            />
            <span className="text-[10px] text-[var(--color-text-tertiary)]">
              Defines the window size during document splitting.
            </span>
          </div>

          {/* Chunk Overlap Slider */}
          <div>
            <div className="flex justify-between mb-1.5 font-medium">
              <label className="label">Default Chunk Overlap</label>
              <span className="font-mono text-teal-400 font-bold">{chunkOverlap} chars</span>
            </div>
            <input
              type="range"
              min="0"
              max="256"
              step="32"
              value={chunkOverlap}
              onChange={(e) => setChunkOverlap(Number(e.target.value))}
              className="w-full accent-[var(--color-teal)] cursor-pointer"
            />
            <span className="text-[10px] text-[var(--color-text-tertiary)]">
              Consecutive character overlap across adjacent chunks.
            </span>
          </div>

          {/* Top-K Slider */}
          <div>
            <div className="flex justify-between mb-1.5 font-medium">
              <label className="label">Default Top-K Chunks</label>
              <span className="font-mono text-sky-400 font-bold">{topK}</span>
            </div>
            <input
              type="range"
              min="1"
              max="20"
              value={topK}
              onChange={(e) => setTopK(Number(e.target.value))}
              className="w-full accent-[var(--color-sky)] cursor-pointer"
            />
            <span className="text-[10px] text-[var(--color-text-tertiary)]">
              Number of nearest neighbor chunks passed to prompt context.
            </span>
          </div>

          {/* Retrieval Mode Radio */}
          <div>
            <label className="label mb-2">Default Retrieval Mode</label>
            <div className="flex gap-4">
              <label className="flex items-center gap-2 cursor-pointer text-xs">
                <input
                  type="radio"
                  name="retrievalMode"
                  value="semantic"
                  checked={retrievalMode === 'semantic'}
                  onChange={() => setRetrievalMode('semantic')}
                  className="accent-[var(--color-accent-primary)]"
                />
                <span>Dense Semantic</span>
              </label>
              <label className="flex items-center gap-2 cursor-pointer text-xs">
                <input
                  type="radio"
                  name="retrievalMode"
                  value="hybrid"
                  checked={retrievalMode === 'hybrid'}
                  onChange={() => setRetrievalMode('hybrid')}
                  className="accent-[var(--color-teal)]"
                />
                <span>Hybrid (BM25 + Dense)</span>
              </label>
            </div>
          </div>
        </div>

        {/* Cross-encoder reranking switch */}
        <div className="pt-3 border-t border-[var(--color-border-subtle)] flex items-center justify-between">
          <div>
            <span className="text-xs font-semibold text-white block">
              Default Cross-Encoder Reranking
            </span>
            <p className="text-[11px] text-[var(--color-text-tertiary)]">
              Whether to re-rank the retrieved candidate passages by default.
            </p>
          </div>
          <input
            type="checkbox"
            checked={useReranking}
            onChange={(e) => setUseReranking(e.target.checked)}
            className="w-4 h-4 rounded accent-[var(--color-accent-primary)] cursor-pointer"
          />
        </div>
      </div>

      {/* ── Section 4: Interface & Display ── */}
      <div
        className="card p-6 space-y-4 border"
        style={{
          backgroundColor: 'var(--color-bg-card)',
          borderColor: 'var(--color-border)',
        }}
      >
        <div className="flex items-center justify-between border-b border-[var(--color-border)] pb-3">
          <div className="flex items-center gap-2">
            {settings.theme === 'dark' ? (
              <Moon className="w-5 h-5 text-purple-400" />
            ) : (
              <Sun className="w-5 h-5 text-amber-400" />
            )}
            <h2 className="text-base font-bold text-white">
              Interface Theme & Appearance
            </h2>
          </div>
          <span className="badge badge-purple text-[10px] uppercase">
            {settings.theme} Mode
          </span>
        </div>

        <div className="flex items-center justify-between">
          <div>
            <span className="text-xs font-semibold text-white block">
              Toggle Visual Color Scheme
            </span>
            <p className="text-[11px] text-[var(--color-text-tertiary)]">
              Switch between Linear dark research mode and daylight high-contrast palette.
            </p>
          </div>

          <button
            type="button"
            onClick={settings.toggleTheme}
            className="btn btn-secondary btn-sm flex items-center gap-2 font-medium"
          >
            {settings.theme === 'dark' ? (
              <>
                <Sun className="w-3.5 h-3.5 text-amber-400" />
                <span>Switch to Light Mode</span>
              </>
            ) : (
              <>
                <Moon className="w-3.5 h-3.5 text-purple-400" />
                <span>Switch to Dark Mode</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* ── Section 5: Danger Zone ── */}
      <div
        className="p-6 rounded-2xl border space-y-4"
        style={{
          backgroundColor: 'rgba(248, 113, 113, 0.05)',
          borderColor: 'rgba(248, 113, 113, 0.25)',
        }}
      >
        <div className="flex items-center gap-2 text-rose-400">
          <AlertTriangle className="w-5 h-5" />
          <h2 className="text-base font-bold">Danger Zone</h2>
        </div>

        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <span className="text-xs font-semibold text-white block">
              Purge Local State & Session Cache
            </span>
            <p className="text-[11px] text-[var(--color-text-tertiary)]">
              Clears all local browser settings, stored keys, and session cache. Vector documents on backend remain preserved.
            </p>
          </div>

          <button
            type="button"
            onClick={() => setShowClearModal(true)}
            className="btn btn-danger btn-sm flex items-center gap-1.5 self-start sm:self-auto font-semibold"
          >
            <Trash2 className="w-3.5 h-3.5" />
            <span>Clear All Local Data</span>
          </button>
        </div>
      </div>

      {/* ── Confirmation Modal ── */}
      {showClearModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/75 backdrop-blur-sm animate-fade-in">
          <div
            className="card max-w-md w-full p-6 space-y-4 shadow-2xl border"
            style={{
              backgroundColor: 'var(--color-bg-card)',
              borderColor: 'rgba(248, 113, 113, 0.4)',
            }}
          >
            <div className="flex items-center gap-3 text-rose-400">
              <AlertTriangle className="w-6 h-6 flex-shrink-0" />
              <h3 className="text-base font-bold">Confirm Local Purge</h3>
            </div>

            <p className="text-xs text-[var(--color-text-secondary)] leading-relaxed">
              Are you sure you want to reset all cached configurations, parameters, and local tokens? This action cannot be reversed.
            </p>

            <div className="flex items-center justify-end gap-3 pt-3">
              <button
                type="button"
                onClick={() => setShowClearModal(false)}
                className="btn btn-secondary btn-sm"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={handleClearAllData}
                disabled={isClearing}
                className="btn btn-danger btn-sm font-semibold"
              >
                {isClearing ? 'Purging...' : 'Yes, Purge Data'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Settings;
