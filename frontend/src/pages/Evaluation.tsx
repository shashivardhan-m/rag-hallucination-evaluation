import React, { useState, useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';
import {
  ChevronDown,
  ChevronUp,
  Sparkles,
  AlertTriangle,
  RefreshCw,
  Table,
} from 'lucide-react';
import { evaluationAPI, qaAPI, getErrorMessage } from '../api/client';
import type {
  MetricDefinition,
  QASession,
  EvaluationResponse,
} from '../types';
import { MetricBadge } from '../components/ui/MetricBadge';
import { MetricBarChart } from '../components/charts/MetricBarChart';

export const Evaluation: React.FC = () => {
  const [searchParams] = useSearchParams();
  const preselectedSessionId = searchParams.get('session_id');

  // Definitions state
  const [definitions, setDefinitions] = useState<Record<string, MetricDefinition>>({});
  const [expandedDefinitions, setExpandedDefinitions] = useState<Record<string, boolean>>({});
  const [showDefinitions, setShowDefinitions] = useState(false);

  // Sessions state
  const [sessions, setSessions] = useState<QASession[]>([]);
  const [selectedSessionId, setSelectedSessionId] = useState<string>(preselectedSessionId || '');
  const [activeSession, setActiveSession] = useState<QASession | null>(null);

  // Form state
  const [referenceAnswer, setReferenceAnswer] = useState('');
  const [isComputing, setIsComputing] = useState(false);
  const [currentEval, setCurrentEval] = useState<EvaluationResponse | null>(null);

  // Historical table state
  const [historicalEvals, setHistoricalEvals] = useState<EvaluationResponse[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchInitialData = async () => {
    setLoading(true);
    setError(null);
    try {
      const [defsData, sessionsData, historyData] = await Promise.allSettled([
        evaluationAPI.getMetricDefinitions(),
        qaAPI.getSessions(50),
        evaluationAPI.getResults(20),
      ]);

      if (defsData.status === 'fulfilled') {
        setDefinitions(defsData.value || {});
      }
      if (sessionsData.status === 'fulfilled') {
        const sess = sessionsData.value || [];
        setSessions(sess);
        if (preselectedSessionId) {
          const match = sess.find((s) => s.id === preselectedSessionId);
          if (match) setActiveSession(match);
        } else if (sess.length > 0 && !selectedSessionId) {
          setSelectedSessionId(sess[0].id);
          setActiveSession(sess[0]);
        }
      }
      if (historyData.status === 'fulfilled') {
        setHistoricalEvals(historyData.value || []);
      }
    } catch (err) {
      setError(getErrorMessage(err));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchInitialData();
  }, [preselectedSessionId]);

  const handleSelectSession = (id: string) => {
    setSelectedSessionId(id);
    const found = sessions.find((s) => s.id === id);
    setActiveSession(found || null);
    setCurrentEval(null);
  };

  const handleCompute = async () => {
    if (!activeSession || isComputing) return;

    setIsComputing(true);
    setError(null);

    try {
      const res = await evaluationAPI.evaluate({
        qa_session_id: activeSession.id,
        question: activeSession.question,
        answer: activeSession.rag_answer ?? '',  // 'answer' is the required field
        context_chunks: activeSession.rag_answer ? [activeSession.rag_answer] : [],
        reference_answer: referenceAnswer.trim() || undefined,
      });
      setCurrentEval(res);
      // Refresh historical list
      try {
        const updatedHistory = await evaluationAPI.getResults(20);
        setHistoricalEvals(updatedHistory);
      } catch {}
    } catch (err) {
      setError(getErrorMessage(err));
    } finally {
      setIsComputing(false);
    }
  };

  const toggleDef = (name: string) => {
    setExpandedDefinitions((prev) => ({ ...prev, [name]: !prev[name] }));
  };

  // Prepare chart data from evaluation metrics
  const chartData = currentEval
    ? [
        ...currentEval.metrics.map((m) => ({
          name: m.metric_name,
          value: m.metric_value,
          fill: m.metric_value >= 0.7 ? '#34d399' : m.metric_value >= 0.4 ? '#fbbf24' : '#f87171',
        })),
        {
          name: 'Hallucination Rate',
          value: currentEval.hallucination_rate,
          fill: currentEval.hallucination_rate <= 0.2 ? '#34d399' : '#f87171',
        },
      ]
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
              Hallucination Evaluation
            </h1>
            <span className="badge badge-rose text-xs">Claim Verification</span>
          </div>
          <p
            className="text-xs sm:text-sm mt-1"
            style={{ color: 'var(--color-text-secondary)' }}
          >
            Compute reference-free faithfulness, semantic relevance, and reference-based accuracy metrics.
          </p>
        </div>

        <button
          type="button"
          onClick={fetchInitialData}
          disabled={loading}
          className="btn btn-secondary btn-sm flex items-center gap-1.5 self-start sm:self-auto"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
          <span>Refresh</span>
        </button>
      </div>

      {/* ── Error Banner ── */}
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

      {/* ── Top Section: Metric Definitions Table (Collapsible) ── */}
      <div
        className="card p-5 space-y-3"
        style={{
          backgroundColor: 'var(--color-bg-card)',
          borderColor: 'var(--color-border)',
        }}
      >
        <div
          className="flex items-center justify-between cursor-pointer"
          onClick={() => setShowDefinitions(!showDefinitions)}
        >
          <div className="flex items-center gap-2">
            <Table className="w-4 h-4 text-purple-400" />
            <h3
              className="text-sm font-semibold tracking-tight"
              style={{ color: 'var(--color-text-primary)' }}
            >
              Evaluation Metric Definitions & Theoretical Foundations
            </h3>
            <span className="badge badge-purple text-[10px]">
              {Object.keys(definitions).length || 6} Metrics
            </span>
          </div>

          <button
            type="button"
            className="btn btn-ghost btn-sm p-1 text-xs text-[var(--color-text-secondary)]"
          >
            {showDefinitions ? (
              <ChevronUp className="w-4 h-4" />
            ) : (
              <ChevronDown className="w-4 h-4" />
            )}
          </button>
        </div>

        {showDefinitions && (
          <div className="overflow-x-auto pt-2">
            <table className="w-full text-left text-xs">
              <thead>
                <tr
                  className="border-b text-[11px] uppercase tracking-wider font-semibold"
                  style={{
                    borderColor: 'var(--color-border)',
                    color: 'var(--color-text-muted)',
                  }}
                >
                  <th className="pb-2.5 pl-2">Metric</th>
                  <th className="pb-2.5">Category</th>
                  <th className="pb-2.5">Range</th>
                  <th className="pb-2.5">Formula / Theoretical Basis</th>
                  <th className="pb-2.5 pr-2">Description</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[var(--color-border-subtle)] font-mono">
                {Object.entries(definitions).length > 0 ? (
                  Object.entries(definitions).map(([name, def]) => {
                    const isExpanded = expandedDefinitions[name];
                    return (
                      <tr
                        key={name}
                        className="hover:bg-white/[0.02] transition-colors cursor-pointer"
                        onClick={() => toggleDef(name)}
                      >
                        <td className="py-2.5 pl-2 font-semibold text-white">
                          {def.name || name}
                        </td>
                        <td className="py-2.5">
                          <span
                            className={`badge text-[10px] font-sans ${
                              def.is_reference_based ? 'badge-teal' : 'badge-purple'
                            }`}
                          >
                            {def.is_reference_based ? 'Reference-Based' : 'Reference-Free'}
                          </span>
                        </td>
                        <td className="py-2.5 text-purple-400 font-semibold">
                          {def.range || '0.0 - 1.0'}
                        </td>
                        <td className="py-2.5 text-[var(--color-text-secondary)] max-w-xs truncate">
                          <code>{def.formula || 'GroundedClaims / TotalClaims'}</code>
                        </td>
                        <td className="py-2.5 pr-2 font-sans text-[var(--color-text-secondary)] max-w-sm">
                          <div className={!isExpanded ? 'truncate' : ''}>
                            {def.description}
                          </div>
                        </td>
                      </tr>
                    );
                  })
                ) : (
                  // Fallback sample definitions
                  <>
                    <tr className="hover:bg-white/[0.02]">
                      <td className="py-2.5 pl-2 font-semibold text-white">Faithfulness</td>
                      <td className="py-2.5"><span className="badge badge-purple text-[10px] font-sans">Reference-Free</span></td>
                      <td className="py-2.5 text-purple-400 font-semibold">0.0 - 1.0</td>
                      <td className="py-2.5 text-[var(--color-text-secondary)]"><code>|Supported Claims| / |Total Claims|</code></td>
                      <td className="py-2.5 pr-2 font-sans text-[var(--color-text-secondary)]">Measures whether all claims in generated answer can be inferred from context.</td>
                    </tr>
                    <tr className="hover:bg-white/[0.02]">
                      <td className="py-2.5 pl-2 font-semibold text-white">Answer Relevance</td>
                      <td className="py-2.5"><span className="badge badge-purple text-[10px] font-sans">Reference-Free</span></td>
                      <td className="py-2.5 text-purple-400 font-semibold">0.0 - 1.0</td>
                      <td className="py-2.5 text-[var(--color-text-secondary)]"><code>CosineSim(embed(Q), embed(Q_gen))</code></td>
                      <td className="py-2.5 pr-2 font-sans text-[var(--color-text-secondary)]">Evaluates if the answer directly addresses the prompt.</td>
                    </tr>
                    <tr className="hover:bg-white/[0.02]">
                      <td className="py-2.5 pl-2 font-semibold text-white">Hallucination Rate</td>
                      <td className="py-2.5"><span className="badge badge-rose text-[10px] font-sans">Reference-Free</span></td>
                      <td className="py-2.5 text-rose-400 font-semibold">0.0 - 1.0</td>
                      <td className="py-2.5 text-[var(--color-text-secondary)]"><code>1.0 - Faithfulness</code></td>
                      <td className="py-2.5 pr-2 font-sans text-[var(--color-text-secondary)]">Proportion of claims that contradict or cannot be verified by retrieved text.</td>
                    </tr>
                  </>
                )}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* ── Middle: Session Selector & Q&A Inspection ── */}
      <div
        className="card p-6 space-y-5 border"
        style={{
          backgroundColor: 'var(--color-bg-card)',
          borderColor: 'var(--color-border)',
        }}
      >
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div>
            <h2 className="text-base font-bold text-white">
              Select Session for Hallucination Audit
            </h2>
            <p className="text-xs text-[var(--color-text-secondary)]">
              Choose an executed Q&A session to evaluate claim groundedness.
            </p>
          </div>

          <div className="w-full sm:w-80">
            <select
              value={selectedSessionId}
              onChange={(e) => handleSelectSession(e.target.value)}
              className="select text-xs py-2"
            >
              <option value="">-- Choose a past Q&A session --</option>
              {sessions.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.question.slice(0, 45)}... ({new Date(s.session_time || Date.now()).toLocaleTimeString()})
                </option>
              ))}
            </select>
          </div>
        </div>

        {activeSession ? (
          <div className="space-y-4 pt-2 border-t border-[var(--color-border-subtle)]">
            {/* Question Display */}
            <div
              className="p-3.5 rounded-xl border text-xs"
              style={{
                backgroundColor: 'var(--color-bg-tertiary)',
                borderColor: 'var(--color-border)',
              }}
            >
              <span className="font-semibold text-purple-400 block mb-1">
                Prompt / Question:
              </span>
              <p className="font-medium text-white text-sm">{activeSession.question}</p>
            </div>

            {/* Side-by-Side Answers Comparison */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
              <div
                className="p-4 rounded-xl border space-y-2"
                style={{
                  backgroundColor: 'var(--color-bg-secondary)',
                  borderColor: 'rgba(248, 113, 113, 0.2)',
                }}
              >
                <div className="flex items-center justify-between">
                  <span className="badge badge-rose text-[10px] font-semibold">
                    Standard LLM Output
                  </span>
                  <span className="text-[10px] text-[var(--color-text-tertiary)] font-mono">
                    {Math.round(activeSession.standard_latency_ms)} ms
                  </span>
                </div>
                <p className="text-[var(--color-text-secondary)] leading-relaxed select-text">
                  {activeSession.standard_answer}
                </p>
              </div>

              <div
                className="p-4 rounded-xl border space-y-2"
                style={{
                  backgroundColor: 'var(--color-bg-secondary)',
                  borderColor: 'rgba(45, 212, 191, 0.2)',
                }}
              >
                <div className="flex items-center justify-between">
                  <span className="badge badge-teal text-[10px] font-semibold">
                    RAG Answer (Subject to Evaluation)
                  </span>
                  <span className="text-[10px] text-teal-400 font-mono">
                    {Math.round(activeSession.rag_latency_ms)} ms
                  </span>
                </div>
                <p className="text-white leading-relaxed select-text">
                  {activeSession.rag_answer}
                </p>
              </div>
            </div>

            {/* Optional Reference Answer and Compute Button */}
            <div
              className="p-4 rounded-xl border space-y-3"
              style={{
                backgroundColor: 'var(--color-bg-secondary)',
                borderColor: 'var(--color-border)',
              }}
            >
              <div className="flex flex-col sm:flex-row items-center justify-between gap-3">
                <div className="w-full sm:flex-1">
                  <label className="label">
                    Ground Truth Reference Answer (Required for Reference-Based Metrics)
                  </label>
                  <input
                    type="text"
                    placeholder="Enter golden answer to compute ROUGE, BERTScore, Correctness..."
                    value={referenceAnswer}
                    onChange={(e) => setReferenceAnswer(e.target.value)}
                    className="input text-xs py-2"
                  />
                </div>

                <button
                  type="button"
                  onClick={handleCompute}
                  disabled={isComputing}
                  className="btn btn-primary btn-md flex items-center gap-2 font-semibold w-full sm:w-auto self-end mt-4 sm:mt-0"
                >
                  {isComputing ? (
                    <>
                      <div className="spinner !w-4 !h-4" />
                      <span>Computing Metrics...</span>
                    </>
                  ) : (
                    <>
                      <Sparkles className="w-4 h-4" />
                      <span>Compute Metrics</span>
                    </>
                  )}
                </button>
              </div>
            </div>
          </div>
        ) : (
          <div className="p-8 text-center border border-dashed rounded-xl text-xs text-[var(--color-text-tertiary)]">
            No session selected. Please pick a past session or run a query in Playground.
          </div>
        )}
      </div>

      {/* ── Results: Gauges, Badges & Bar Chart ── */}
      {currentEval && (
        <div
          className="card p-6 space-y-6 border animate-slide-up"
          style={{
            backgroundColor: 'var(--color-bg-card)',
            borderColor: 'var(--color-border-active)',
          }}
        >
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-base font-bold text-white">
                Evaluation Output & Hallucination Diagnostics
              </h2>
              <p className="text-xs text-[var(--color-text-secondary)]">
                Evaluated at {new Date(currentEval.evaluated_at).toLocaleString()}
              </p>
            </div>
          </div>

          {/* Prominent Hallucination Gauge Banner */}
          <div
            className="p-5 rounded-2xl border flex flex-col sm:flex-row items-center justify-between gap-6"
            style={{
              background:
                currentEval.hallucination_rate <= 0.15
                  ? 'linear-gradient(135deg, rgba(52, 211, 153, 0.12), rgba(18, 18, 30, 0.8))'
                  : currentEval.hallucination_rate <= 0.35
                  ? 'linear-gradient(135deg, rgba(251, 191, 36, 0.12), rgba(18, 18, 30, 0.8))'
                  : 'linear-gradient(135deg, rgba(248, 113, 113, 0.15), rgba(18, 18, 30, 0.8))',
              borderColor:
                currentEval.hallucination_rate <= 0.15
                  ? 'rgba(52, 211, 153, 0.3)'
                  : currentEval.hallucination_rate <= 0.35
                  ? 'rgba(251, 191, 36, 0.3)'
                  : 'rgba(248, 113, 113, 0.3)',
            }}
          >
            <div className="space-y-1">
              <span className="text-xs uppercase tracking-wider font-bold" style={{ color: 'var(--color-text-secondary)' }}>
                Computed Hallucination Rate
              </span>
              <div className="flex items-baseline gap-2">
                <span
                  className="text-4xl font-extrabold font-mono"
                  style={{
                    color:
                      currentEval.hallucination_rate <= 0.15
                        ? 'var(--color-emerald)'
                        : currentEval.hallucination_rate <= 0.35
                        ? 'var(--color-amber)'
                        : 'var(--color-rose)',
                  }}
                >
                  {(currentEval.hallucination_rate * 100).toFixed(1)}%
                </span>
                <span className="text-xs font-semibold text-white/80">
                  {currentEval.hallucination_rate <= 0.15
                    ? 'High Faithfulness (Minimal Hallucination)'
                    : currentEval.hallucination_rate <= 0.35
                    ? 'Moderate Grounding (Check Claims)'
                    : 'Severe Hallucination Risk'}
                </span>
              </div>
              <p className="text-xs text-[var(--color-text-secondary)]">
                {currentEval.hallucination_rate <= 0.15
                  ? 'All core claims are grounded in context evidence.'
                  : 'Unsubstantiated entities or reasoning steps detected in generation.'}
              </p>
            </div>

            {/* Metric Badges Cluster */}
            <div className="flex flex-wrap gap-2 justify-end max-w-md">
              {currentEval.metrics.map((m, idx) => (
                <MetricBadge
                  key={idx}
                  name={m.metric_name}
                  value={m.metric_value}
                  isReferenceBased={m.is_reference_based}
                />
              ))}
            </div>
          </div>

          {/* Metric Bar Chart */}
          <div className="pt-2">
            <MetricBarChart
              data={chartData}
              title="Metric Comparison Breakdown"
              height={240}
            />
          </div>
        </div>
      )}

      {/* ── Historical Evaluations Table (Last 20) ── */}
      <div
        className="card p-5 space-y-4"
        style={{
          backgroundColor: 'var(--color-bg-card)',
          borderColor: 'var(--color-border)',
        }}
      >
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-sm font-semibold text-white">
              Historical Evaluation Records
            </h3>
            <p className="text-xs text-[var(--color-text-tertiary)]">
              Latest 20 computed evaluations across all sessions
            </p>
          </div>
          <span className="badge badge-teal text-xs">
            {historicalEvals.length} Records
          </span>
        </div>

        {historicalEvals.length === 0 ? (
          <div className="p-8 text-center border border-dashed rounded-xl text-xs text-[var(--color-text-tertiary)]">
            No historical evaluations recorded yet. Run your first evaluation above!
          </div>
        ) : (
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
                  <th className="pb-3 pl-2">Session ID</th>
                  <th className="pb-3">Timestamp</th>
                  <th className="pb-3 text-right">Hallucination Rate</th>
                  <th className="pb-3 text-right">Faithfulness</th>
                  <th className="pb-3 text-right">Relevance</th>
                  <th className="pb-3 pr-2 text-right">Metrics Computed</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[var(--color-border-subtle)] font-mono">
                {historicalEvals.map((item, idx) => {
                  const faithfulness = item.metrics.find(
                    (m) => m.metric_name.toLowerCase().includes('faith')
                  )?.metric_value;
                  const relevance = item.metrics.find(
                    (m) => m.metric_name.toLowerCase().includes('relev')
                  )?.metric_value;

                  return (
                    <tr
                      key={idx}
                      className="hover:bg-white/[0.02] transition-colors"
                    >
                      <td className="py-2.5 pl-2 font-semibold text-purple-400">
                        {item.session_id ? item.session_id.slice(0, 8) : `eval-${idx}`}...
                      </td>
                      <td className="py-2.5 text-[var(--color-text-tertiary)]">
                        {item.evaluated_at
                          ? new Date(item.evaluated_at).toLocaleString()
                          : 'Recent'}
                      </td>
                      <td className="py-2.5 text-right">
                        <span
                          className={`font-bold ${
                            item.hallucination_rate <= 0.15
                              ? 'text-emerald-400'
                              : item.hallucination_rate <= 0.35
                              ? 'text-amber-400'
                              : 'text-rose-400'
                          }`}
                        >
                          {(item.hallucination_rate * 100).toFixed(1)}%
                        </span>
                      </td>
                      <td className="py-2.5 text-right text-emerald-400">
                        {faithfulness != null
                          ? `${(faithfulness * 100).toFixed(1)}%`
                          : '—'}
                      </td>
                      <td className="py-2.5 text-right text-teal-400">
                        {relevance != null
                          ? `${(relevance * 100).toFixed(1)}%`
                          : '—'}
                      </td>
                      <td className="py-2.5 pr-2 text-right">
                        <span className="badge badge-purple text-[10px]">
                          {item.metrics.length} metrics
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};

export default Evaluation;
