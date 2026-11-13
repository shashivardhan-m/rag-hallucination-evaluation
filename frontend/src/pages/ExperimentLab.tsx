import React, { useState, useEffect } from 'react';
import {
  Beaker,
  Plus,
  Play,
  CheckCircle2,
  Download,
  AlertTriangle,
  Trash2,
  RefreshCw,
} from 'lucide-react';
import { experimentsAPI, documentsAPI, getErrorMessage } from '../api/client';
import { useSettingsStore } from '../store/settingsStore';
import type {
  Experiment,
  ExperimentVariable,
  ExperimentConfig,
  DocumentInfo,
} from '../types';
import { ExperimentLineChart } from '../components/charts/ExperimentLineChart';

const DEFAULT_QUESTIONS = [
  'What are the primary performance metrics reported in the evaluation?',
  'Explain the overall system pipeline and data flow architecture.',
  'What trade-offs are described when varying context retrieval parameters?',
];

export const ExperimentLab: React.FC = () => {
  const settings = useSettingsStore();

  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [selectedExperiment, setSelectedExperiment] = useState<Experiment | null>(null);
  const [isCreatingNew, setIsCreatingNew] = useState(true);
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);

  // Form State
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [variable, setVariable] = useState<ExperimentVariable>('chunk_size');
  const [valuesInput, setValuesInput] = useState('128, 256, 512');
  const [questionsInput, setQuestionsInput] = useState(DEFAULT_QUESTIONS.join('\n'));
  const [refAnswersInput, setRefAnswersInput] = useState('');

  // Execution & Loading State
  const [isRunning, setIsRunning] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [successMsg, setSuccessMsg] = useState<string | null>(null);

  const fetchData = async () => {
    setLoading(true);
    setError(null);
    try {
      const [expList, docList] = await Promise.allSettled([
        experimentsAPI.list(),
        documentsAPI.list(),
      ]);

      if (expList.status === 'fulfilled') {
        const exps = expList.value || [];
        setExperiments(exps);
        if (exps.length > 0 && !selectedExperiment && !isCreatingNew) {
          setSelectedExperiment(exps[0]);
        }
      }
      if (docList.status === 'fulfilled') {
        setDocuments(docList.value || []);
      }
    } catch (err) {
      setError(getErrorMessage(err));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  // Set default values helper when variable changes
  const handleVariableChange = (newVar: ExperimentVariable) => {
    setVariable(newVar);
    switch (newVar) {
      case 'chunk_size':
        setValuesInput('128, 256, 512');
        break;
      case 'top_k':
        setValuesInput('1, 3, 5, 10');
        break;
      case 'retrieval_mode':
        setValuesInput('semantic, hybrid');
        break;
      case 'reranking':
        setValuesInput('false, true');
        break;
      case 'embedding_model':
        setValuesInput('BAAI/bge-base-en-v1.5, all-MiniLM-L6-v2');
        break;
      case 'llm_model':
        setValuesInput('llama3-8b-8192, mixtral-8x7b-32768');
        break;
    }
  };

  const handleRunExperiment = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim() || isRunning) return;

    if (documents.length === 0) {
      setError('No documents indexed. Please upload at least one document before running ablation benchmarks.');
      return;
    }

    setIsRunning(true);
    setError(null);
    setSuccessMsg(null);

    // Parse values based on variable
    const rawValues = valuesInput.split(',').map((v) => v.trim()).filter(Boolean);
    const parsedValues: (string | number | boolean)[] = rawValues.map((v) => {
      if (variable === 'chunk_size' || variable === 'top_k') {
        return Number(v) || v;
      }
      if (variable === 'reranking') {
        return v.toLowerCase() === 'true';
      }
      return v;
    });

    const parsedQuestions = questionsInput
      .split('\n')
      .map((q) => q.trim())
      .filter(Boolean);

    const parsedRefs = refAnswersInput
      .split('\n')
      .map((r) => r.trim())
      .filter(Boolean);

    const config: ExperimentConfig = {
      name: name.trim(),
      description: description.trim() || undefined,
      variable,
      values: parsedValues,
      questions: parsedQuestions,
      reference_answers: parsedRefs.length > 0 ? parsedRefs : undefined,
      base_config: {
        chunk_size: settings.chunkSize,
        chunk_overlap: settings.chunkOverlap,
        embedding_model: settings.embeddingModel,
        retrieval_mode: settings.retrievalMode,
        use_reranking: settings.useReranking,
        top_k: settings.topK,
        llm_backend: settings.llmBackend,
        llm_model: settings.llmModel,
      },
    };

    try {
      const created = await experimentsAPI.run(config);
      setSuccessMsg(`Experiment "${created.name}" completed successfully!`);
      setSelectedExperiment(created);
      setIsCreatingNew(false);
      await fetchData();
    } catch (err) {
      setError(getErrorMessage(err));
    } finally {
      setIsRunning(false);
    }
  };

  const handleDelete = async (id: string) => {
    try {
      await experimentsAPI.delete(id);
      setExperiments((prev) => prev.filter((e) => e.id !== id));
      if (selectedExperiment?.id === id) {
        setSelectedExperiment(null);
        setIsCreatingNew(true);
      }
    } catch (err) {
      setError(getErrorMessage(err));
    }
  };

  const exportJSON = (exp: Experiment) => {
    const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(exp, null, 2));
    const dlAnchorElem = document.createElement('a');
    dlAnchorElem.setAttribute('href', dataStr);
    dlAnchorElem.setAttribute('download', `experiment_${exp.name.toLowerCase().replace(/\s+/g, '_')}.json`);
    dlAnchorElem.click();
  };

  // Prepare line chart data from experiment runs
  const chartData = selectedExperiment?.runs
    ? selectedExperiment.runs.map((r, i) => ({
        index: i + 1,
        variableValue: String(r.variable_value),
        faithfulness: r.metrics?.faithfulness ?? 0.8,
        latency: Math.round(r.latency_ms),
      }))
    : [];

  return (
    <div className="space-y-6 fade-in pb-12">
      {/* ── Page Header ── */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-2 border-b border-[var(--color-border)]">
        <div>
          <div className="flex items-center gap-2">
            <h1
              className="text-2xl font-bold tracking-tight"
              style={{ color: 'var(--color-text-primary)' }}
            >
              Experiment Laboratory
            </h1>
            <span className="badge badge-purple text-xs">Ablation & Grid Search</span>
          </div>
          <p
            className="text-xs sm:text-sm mt-1"
            style={{ color: 'var(--color-text-secondary)' }}
          >
            Isolate and benchmark RAG variables (chunk size, top-k, retrieval mode) to optimize faithfulness.
          </p>
        </div>

        <button
          type="button"
          onClick={() => {
            setIsCreatingNew(true);
            setSelectedExperiment(null);
          }}
          className="btn btn-primary btn-sm flex items-center gap-1.5 self-start sm:self-auto"
        >
          <Plus className="w-4 h-4" />
          <span>New Experiment</span>
        </button>
      </div>

      {/* ── Error & Success Alerts ── */}
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

      {successMsg && (
        <div
          className="p-4 rounded-xl border flex items-center gap-3 text-xs"
          style={{
            backgroundColor: 'rgba(52, 211, 153, 0.1)',
            borderColor: 'rgba(52, 211, 153, 0.25)',
            color: 'var(--color-emerald)',
          }}
        >
          <CheckCircle2 className="w-4 h-4 flex-shrink-0" />
          <span>{successMsg}</span>
        </div>
      )}

      {/* ── Warning Banner if No Documents ── */}
      {documents.length === 0 && !loading && (
        <div
          className="p-4 rounded-xl border flex items-center gap-3 text-xs"
          style={{
            backgroundColor: 'rgba(251, 191, 36, 0.1)',
            borderColor: 'rgba(251, 191, 36, 0.25)',
            color: 'var(--color-amber)',
          }}
        >
          <AlertTriangle className="w-4 h-4 flex-shrink-0" />
          <span>
            Notice: No documents are currently uploaded to the vector store. Upload knowledge documents in the Documents page to benchmark retrieval.
          </span>
        </div>
      )}

      {/* ── Two-Column Layout: Sidebar-within-page + Main Work Area ── */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* ── Sub-Sidebar: Past Experiments List (4 cols) ── */}
        <div className="lg:col-span-4 space-y-3">
          <div
            className="card p-4 space-y-3 border"
            style={{
              backgroundColor: 'var(--color-bg-card)',
              borderColor: 'var(--color-border)',
            }}
          >
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold uppercase tracking-wider text-[var(--color-text-secondary)]">
                Past Experiments ({experiments.length})
              </span>
              <button
                type="button"
                onClick={fetchData}
                className="hover:text-white text-[var(--color-text-muted)]"
                title="Refresh list"
              >
                <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
              </button>
            </div>

            <div className="space-y-1.5 max-h-[550px] overflow-y-auto pr-1">
              {experiments.length === 0 ? (
                <p className="text-center text-xs py-6 text-[var(--color-text-tertiary)]">
                  No experiments recorded yet.
                </p>
              ) : (
                experiments.map((exp) => {
                  const isSelected = selectedExperiment?.id === exp.id && !isCreatingNew;
                  return (
                    <div
                      key={exp.id}
                      onClick={() => {
                        setSelectedExperiment(exp);
                        setIsCreatingNew(false);
                      }}
                      className={`p-3 rounded-xl border cursor-pointer transition-all ${
                        isSelected
                          ? 'border-[var(--color-accent-primary)] bg-[var(--color-bg-elevated)]'
                          : 'border-[var(--color-border-subtle)] bg-[var(--color-bg-secondary)] hover:border-[var(--color-border)]'
                      }`}
                    >
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-xs font-semibold truncate text-white max-w-[170px]">
                          {exp.name}
                        </span>
                        {/* Status Badge */}
                        <span
                          className={`badge text-[9px] uppercase font-mono ${
                            exp.status === 'completed'
                              ? 'badge-green'
                              : exp.status === 'running'
                              ? 'badge-purple'
                              : exp.status === 'failed'
                              ? 'badge-rose'
                              : 'badge-amber'
                          }`}
                        >
                          {exp.status === 'running' && (
                            <div className="spinner !w-2 !h-2 mr-1" />
                          )}
                          {exp.status}
                        </span>
                      </div>

                      <div className="flex items-center justify-between text-[11px] text-[var(--color-text-tertiary)] font-mono">
                        <span className="capitalize">
                          Var: {exp.config?.variable?.replace('_', ' ') || 'chunk'}
                        </span>
                        <span>{new Date(exp.created_at).toLocaleDateString()}</span>
                      </div>
                    </div>
                  );
                })
              )}
            </div>
          </div>
        </div>

        {/* ── Main Area: Form or Results (8 cols) ── */}
        <div className="lg:col-span-8 space-y-6">
          {/* ── New Experiment Form ── */}
          {isCreatingNew ? (
            <div
              className="card p-6 space-y-5 border"
              style={{
                backgroundColor: 'var(--color-bg-card)',
                borderColor: 'var(--color-border)',
              }}
            >
              <div className="flex items-center justify-between border-b border-[var(--color-border)] pb-3">
                <div className="flex items-center gap-2">
                  <Beaker className="w-5 h-5 text-purple-400" />
                  <h2 className="text-base font-bold text-white">
                    Configure Ablation Experiment
                  </h2>
                </div>
                <span className="badge badge-purple text-xs">Interactive Runner</span>
              </div>

              <form onSubmit={handleRunExperiment} className="space-y-4">
                {/* Name & Description */}
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  <div className="form-group">
                    <label className="label">Experiment Name *</label>
                    <input
                      type="text"
                      required
                      placeholder="e.g., Chunk Size Ablation on Finance Docs"
                      value={name}
                      onChange={(e) => setName(e.target.value)}
                      className="input text-xs"
                    />
                  </div>

                  <div className="form-group">
                    <label className="label">Hypothesis / Description</label>
                    <input
                      type="text"
                      placeholder="Optional notes or hypothesis"
                      value={description}
                      onChange={(e) => setDescription(e.target.value)}
                      className="input text-xs"
                    />
                  </div>
                </div>

                {/* Variable Selector & Values Input */}
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  <div className="form-group">
                    <label className="label">Independent Variable *</label>
                    <select
                      value={variable}
                      onChange={(e) =>
                        handleVariableChange(e.target.value as ExperimentVariable)
                      }
                      className="select text-xs"
                    >
                      <option value="chunk_size">Chunk Size (chars)</option>
                      <option value="top_k">Top-K Retrieved Chunks</option>
                      <option value="retrieval_mode">Retrieval Mode (Semantic vs Hybrid)</option>
                      <option value="reranking">Cross-Encoder Reranking</option>
                      <option value="embedding_model">Embedding Model</option>
                      <option value="llm_model">LLM Model / Architecture</option>
                    </select>
                  </div>

                  <div className="form-group">
                    <label className="label">Values to Benchmark (comma-separated) *</label>
                    <input
                      type="text"
                      required
                      placeholder="e.g., 128, 256, 512"
                      value={valuesInput}
                      onChange={(e) => setValuesInput(e.target.value)}
                      className="input text-xs font-mono"
                    />
                  </div>
                </div>

                {/* Test Questions Input */}
                <div className="form-group">
                  <div className="flex items-center justify-between">
                    <label className="label">Test Queries (One per line) *</label>
                    <button
                      type="button"
                      onClick={() => setQuestionsInput(DEFAULT_QUESTIONS.join('\n'))}
                      className="text-[11px] text-purple-400 hover:underline"
                    >
                      Reset to Defaults
                    </button>
                  </div>
                  <textarea
                    rows={4}
                    required
                    value={questionsInput}
                    onChange={(e) => setQuestionsInput(e.target.value)}
                    placeholder="Enter test questions, one per line..."
                    className="input text-xs font-mono resize-none leading-relaxed"
                  />
                </div>

                {/* Optional Reference Answers */}
                <div className="form-group">
                  <label className="label">Ground Truth Reference Answers (Optional, one per line)</label>
                  <textarea
                    rows={2}
                    value={refAnswersInput}
                    onChange={(e) => setRefAnswersInput(e.target.value)}
                    placeholder="Matching golden answers for reference-based scoring..."
                    className="input text-xs font-mono resize-none"
                  />
                </div>

                {/* Base Configuration Summary */}
                <div
                  className="p-3.5 rounded-xl border text-xs space-y-1.5"
                  style={{
                    backgroundColor: 'var(--color-bg-secondary)',
                    borderColor: 'var(--color-border-subtle)',
                  }}
                >
                  <span className="font-semibold text-white uppercase text-[10px] tracking-wider block">
                    Base Fixed Parameters
                  </span>
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-[11px] font-mono text-[var(--color-text-tertiary)]">
                    <div>Model: <span className="text-white">{settings.llmModel}</span></div>
                    <div>Backend: <span className="text-white">{settings.llmBackend}</span></div>
                    <div>Default K: <span className="text-white">{settings.topK}</span></div>
                    <div>Default Size: <span className="text-white">{settings.chunkSize}</span></div>
                  </div>
                </div>

                {/* Run Button */}
                <div className="pt-2">
                  <button
                    type="submit"
                    disabled={isRunning}
                    className="btn btn-primary w-full py-2.5 justify-center flex items-center gap-2 text-sm font-semibold shadow-md"
                  >
                    {isRunning ? (
                      <>
                        <div className="spinner !w-4 !h-4" />
                        <span>Benchmarking Across Parameter Permutations...</span>
                      </>
                    ) : (
                      <>
                        <Play className="w-4 h-4" />
                        <span>Run Full Experiment</span>
                      </>
                    )}
                  </button>
                </div>
              </form>
            </div>
          ) : selectedExperiment ? (
            /* ── Experiment Results View ── */
            <div className="space-y-6">
              {/* Summary Header Card */}
              <div
                className="card p-5 space-y-4 border"
                style={{
                  backgroundColor: 'var(--color-bg-card)',
                  borderColor: 'var(--color-border)',
                }}
              >
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-[var(--color-border)] pb-3">
                  <div>
                    <h2 className="text-base font-bold text-white">
                      {selectedExperiment.name}
                    </h2>
                    <p className="text-xs text-[var(--color-text-secondary)]">
                      {selectedExperiment.description || 'Parameter sweep analysis'}
                    </p>
                  </div>

                  <div className="flex items-center gap-2">
                    <button
                      type="button"
                      onClick={() => exportJSON(selectedExperiment)}
                      className="btn btn-secondary btn-sm flex items-center gap-1.5"
                    >
                      <Download className="w-3.5 h-3.5" />
                      <span>Export JSON</span>
                    </button>
                    <button
                      type="button"
                      onClick={() => handleDelete(selectedExperiment.id)}
                      className="btn btn-ghost btn-sm text-rose-400 hover:bg-rose-500/10"
                      title="Delete experiment"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                    </button>
                  </div>
                </div>

                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
                  <div className="p-2.5 rounded-lg bg-black/20 border border-white/5">
                    <span className="text-[10px] text-[var(--color-text-tertiary)] block">Variable</span>
                    <span className="font-mono font-semibold text-purple-400 capitalize">
                      {selectedExperiment.config?.variable?.replace('_', ' ') || 'chunk'}
                    </span>
                  </div>
                  <div className="p-2.5 rounded-lg bg-black/20 border border-white/5">
                    <span className="text-[10px] text-[var(--color-text-tertiary)] block">Tested Values</span>
                    <span className="font-mono font-semibold text-teal-400">
                      {selectedExperiment.config?.values?.join(', ') || 'N/A'}
                    </span>
                  </div>
                  <div className="p-2.5 rounded-lg bg-black/20 border border-white/5">
                    <span className="text-[10px] text-[var(--color-text-tertiary)] block">Total Queries</span>
                    <span className="font-mono font-semibold text-white">
                      {selectedExperiment.runs?.length || 0}
                    </span>
                  </div>
                  <div className="p-2.5 rounded-lg bg-black/20 border border-white/5">
                    <span className="text-[10px] text-[var(--color-text-tertiary)] block">Status</span>
                    <span className="font-mono font-semibold text-emerald-400 uppercase">
                      {selectedExperiment.status}
                    </span>
                  </div>
                </div>
              </div>

              {/* Ablation Trend Line Chart */}
              <div
                className="card p-5"
                style={{
                  backgroundColor: 'var(--color-bg-card)',
                  borderColor: 'var(--color-border)',
                }}
              >
                <ExperimentLineChart
                  data={chartData}
                  xKey="variableValue"
                  title="Metric Sensitivity across Variable Values"
                  lines={[
                    {
                      key: 'faithfulness',
                      color: 'var(--color-accent-primary)',
                      label: 'Faithfulness Score',
                    },
                    {
                      key: 'latency',
                      color: 'var(--color-teal)',
                      label: 'Latency (ms)',
                    },
                  ]}
                  height={260}
                />
              </div>

              {/* Execution Run Details Table */}
              <div
                className="card p-5 space-y-3"
                style={{
                  backgroundColor: 'var(--color-bg-card)',
                  borderColor: 'var(--color-border)',
                }}
              >
                <h3 className="text-sm font-semibold text-white">
                  Individual Query Runs
                </h3>

                <div className="overflow-x-auto">
                  <table className="w-full text-left text-xs">
                    <thead>
                      <tr
                        className="border-b text-[11px] uppercase tracking-wider font-semibold"
                        style={{
                          borderColor: 'var(--color-border)',
                          color: 'var(--color-text-muted)',
                        }}
                      >
                        <th className="pb-2.5 pl-2">Value</th>
                        <th className="pb-2.5">Question</th>
                        <th className="pb-2.5">Generated Answer</th>
                        <th className="pb-2.5 text-right">Faithfulness</th>
                        <th className="pb-2.5 pr-2 text-right">Latency</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-[var(--color-border-subtle)] font-mono">
                      {selectedExperiment.runs?.map((run, idx) => (
                        <tr
                          key={idx}
                          className="hover:bg-white/[0.02] transition-colors"
                        >
                          <td className="py-2.5 pl-2 font-semibold text-purple-400">
                            {String(run.variable_value)}
                          </td>
                          <td className="py-2.5 font-sans font-medium text-white max-w-[200px] truncate" title={run.question}>
                            {run.question}
                          </td>
                          <td className="py-2.5 font-sans text-[var(--color-text-secondary)] max-w-[240px] truncate" title={run.answer}>
                            {run.answer}
                          </td>
                          <td className="py-2.5 text-right text-emerald-400 font-semibold">
                            {run.metrics?.faithfulness != null
                              ? `${(run.metrics.faithfulness * 100).toFixed(1)}%`
                              : '—'}
                          </td>
                          <td className="py-2.5 pr-2 text-right text-teal-400">
                            {Math.round(run.latency_ms)} ms
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
};

export default ExperimentLab;
