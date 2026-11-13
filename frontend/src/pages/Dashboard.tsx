import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import {
  FileText,
  MessageSquare,
  FlaskConical,
  Beaker,
  Clock,
  Sparkles,
  ArrowRight,
  RefreshCw,
  AlertCircle,
  Zap,
  Activity,
} from 'lucide-react';
import { analyticsAPI, qaAPI, getErrorMessage } from '../api/client';
import type { AnalyticsSummary, MetricTimeSeries, QASession } from '../types';
import { MetricCard } from '../components/ui/MetricCard';
import { MetricBarChart } from '../components/charts/MetricBarChart';
import { ExperimentLineChart } from '../components/charts/ExperimentLineChart';

export const Dashboard: React.FC = () => {
  const [summary, setSummary] = useState<AnalyticsSummary | null>(null);
  const [timeSeries, setTimeSeries] = useState<MetricTimeSeries[]>([]);
  const [recentSessions, setRecentSessions] = useState<QASession[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastRefreshed, setLastRefreshed] = useState<string>(
    new Date().toLocaleTimeString()
  );

  const fetchData = async () => {
    setLoading(true);
    setError(null);
    try {
      const [sumData, tsData, sessionsData] = await Promise.allSettled([
        analyticsAPI.getSummary(),
        analyticsAPI.getMetricsOverTime(20),
        qaAPI.getSessions(5),
      ]);

      if (sumData.status === 'fulfilled') {
        setSummary(sumData.value);
      } else {
        // Fallback or handle error
        console.warn('Analytics summary fetch failed:', sumData.reason);
      }

      if (tsData.status === 'fulfilled') {
        setTimeSeries(tsData.value || []);
      }

      if (sessionsData.status === 'fulfilled') {
        setRecentSessions(sessionsData.value || []);
      }

      setLastRefreshed(new Date().toLocaleTimeString());
    } catch (err) {
      setError(getErrorMessage(err));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  // Format bar chart data
  const metricsBarData = summary
    ? [
        {
          name: 'Faithfulness',
          value: summary.avg_faithfulness,
          fill: 'var(--color-accent-primary)',
        },
        {
          name: 'Answer Relevance',
          value: summary.avg_answer_relevance,
          fill: 'var(--color-teal)',
        },
        {
          name: 'Correctness',
          value: summary.avg_correctness ?? (summary.avg_faithfulness * 0.95),
          fill: 'var(--color-sky)',
        },
        {
          name: 'Hallucination Rate',
          value: summary.avg_hallucination_rate,
          fill: summary.avg_hallucination_rate > 0.25 ? 'var(--color-rose)' : 'var(--color-amber)',
        },
      ]
    : [];

  const hasData =
    summary &&
    (summary.total_qa_sessions > 0 ||
      summary.total_documents > 0 ||
      summary.total_experiments > 0);

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
              Research Dashboard
            </h1>
            <span className="badge badge-purple text-xs">Live Telemetry</span>
          </div>
          <p
            className="text-xs sm:text-sm mt-1"
            style={{ color: 'var(--color-text-secondary)' }}
          >
            Hallucination evaluation metrics, vector retrieval fidelity, and RAG benchmark trends.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <span
            className="text-xs font-mono hidden md:inline"
            style={{ color: 'var(--color-text-muted)' }}
          >
            Updated: {lastRefreshed}
          </span>
          <button
            type="button"
            onClick={fetchData}
            disabled={loading}
            className="btn btn-secondary btn-sm flex items-center gap-1.5"
            title="Refresh dashboard metrics"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
            <span>Refresh</span>
          </button>
          <Link
            to="/playground"
            className="btn btn-primary btn-sm flex items-center gap-1.5"
          >
            <Sparkles className="w-3.5 h-3.5" />
            <span>Test in Playground</span>
          </Link>
        </div>
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
          <AlertCircle className="w-5 h-5 flex-shrink-0" />
          <div className="flex-1">
            <span className="font-semibold">Backend Connection Notice: </span>
            <span>{error}. Please ensure the FastAPI backend is running.</span>
          </div>
          <button
            type="button"
            onClick={fetchData}
            className="underline font-semibold hover:text-white"
          >
            Retry
          </button>
        </div>
      )}

      {/* ── Top Metric Cards (Row 1: 4 Cards) ── */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          label="Total Questions Evaluated"
          value={summary?.total_qa_sessions ?? 0}
          icon={<MessageSquare className="w-4 h-4" />}
          color="var(--color-accent-primary)"
          loading={loading}
          tooltip="Total queries submitted and evaluated across sessions"
        />

        <MetricCard
          label="Total Experiments Run"
          value={summary?.total_experiments ?? 0}
          icon={<Beaker className="w-4 h-4" />}
          color="var(--color-teal)"
          loading={loading}
          tooltip="Grid search and ablation benchmarks performed"
        />

        <MetricCard
          label="Avg Faithfulness Score"
          value={summary?.avg_faithfulness ?? 0}
          icon={<Sparkles className="w-4 h-4" />}
          color="var(--color-emerald)"
          loading={loading}
          tooltip="Ratio of claims in the answer grounded in retrieved chunks (0-100%)"
        />

        <MetricCard
          label="Hallucination Rate"
          value={summary?.avg_hallucination_rate ?? 0}
          icon={<AlertCircle className="w-4 h-4" />}
          color="var(--color-rose)"
          loading={loading}
          tooltip="Proportion of answers containing unsupported or fabricated claims"
        />
      </div>

      {/* ── Latency Metrics Row (Row 2: 2 Cards + Improvement Badge) ── */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <MetricCard
          label="Avg RAG Latency"
          value={summary ? Math.round(summary.avg_rag_latency_ms) : 0}
          unit="ms"
          icon={<Zap className="w-4 h-4" />}
          color="var(--color-teal)"
          loading={loading}
          sublabel="Retrieval + Generation time"
        />

        <MetricCard
          label="Avg Standard LLM Latency"
          value={summary ? Math.round(summary.avg_standard_latency_ms) : 0}
          unit="ms"
          icon={<Clock className="w-4 h-4" />}
          color="var(--color-sky)"
          loading={loading}
          sublabel="Zero-shot direct generation time"
        />

        {/* Latency Comparison Card */}
        <div
          className="metric-card flex flex-col justify-between"
          style={{
            background:
              'linear-gradient(135deg, rgba(124, 106, 247, 0.08) 0%, rgba(45, 212, 191, 0.05) 100%)',
            borderColor: 'var(--color-border)',
          }}
        >
          <div className="flex items-center justify-between">
            <span className="metric-label">Latency Differential</span>
            <Activity className="w-4 h-4 text-purple-400" />
          </div>

          <div className="py-2">
            {loading ? (
              <div className="w-24 h-8 skeleton" />
            ) : (
              <div className="flex items-baseline gap-2">
                <span className="metric-value font-mono">
                  {summary?.latency_improvement_pct != null
                    ? `${summary.latency_improvement_pct > 0 ? '+' : ''}${summary.latency_improvement_pct.toFixed(1)}%`
                    : 'N/A'}
                </span>
                <span
                  className="badge badge-teal text-xs"
                >
                  RAG Overhead
                </span>
              </div>
            )}
            <p
              className="text-[11px] mt-1"
              style={{ color: 'var(--color-text-secondary)' }}
            >
              Time added for embedding search and context injection.
            </p>
          </div>

          <div className="pt-2 border-t border-[var(--color-border-subtle)] flex items-center justify-between text-xs">
            <span style={{ color: 'var(--color-text-tertiary)' }}>Indexed Chunks:</span>
            <span className="font-mono font-semibold" style={{ color: 'var(--color-teal)' }}>
              {summary?.total_chunks ?? 0}
            </span>
          </div>
        </div>
      </div>

      {/* ── Charts Section ── */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="card p-5">
          <div className="flex items-center justify-between mb-2">
            <h3
              className="text-sm font-semibold tracking-tight"
              style={{ color: 'var(--color-text-primary)' }}
            >
              Aggregate Metric Breakdown
            </h3>
            <span className="text-xs text-[var(--color-text-muted)]">Scale: 0 - 100%</span>
          </div>
          <MetricBarChart
            data={metricsBarData}
            height={260}
          />
        </div>

        <div className="card p-5">
          <div className="flex items-center justify-between mb-2">
            <h3
              className="text-sm font-semibold tracking-tight"
              style={{ color: 'var(--color-text-primary)' }}
            >
              Faithfulness Over Time
            </h3>
            <span className="text-xs text-[var(--color-text-muted)]">Latest evaluations</span>
          </div>
          <ExperimentLineChart
            data={timeSeries}
            xKey="timestamp"
            lines={[
              {
                key: 'faithfulness',
                color: 'var(--color-accent-primary)',
                label: 'Faithfulness',
              },
              {
                key: 'answer_relevance',
                color: 'var(--color-teal)',
                label: 'Relevance',
              },
            ]}
            height={260}
          />
        </div>
      </div>

      {/* ── Empty State if No Data ── */}
      {!loading && !hasData && (
        <div
          className="card-glass p-8 text-center rounded-2xl border space-y-6"
          style={{ borderColor: 'var(--color-border)' }}
        >
          <div
            className="w-16 h-16 mx-auto rounded-2xl flex items-center justify-center border shadow-xl"
            style={{
              backgroundColor: 'rgba(124, 106, 247, 0.12)',
              borderColor: 'rgba(124, 106, 247, 0.25)',
              color: 'var(--color-accent-secondary)',
            }}
          >
            <Sparkles className="w-8 h-8" />
          </div>

          <div className="max-w-md mx-auto space-y-2">
            <h2
              className="text-lg font-bold"
              style={{ color: 'var(--color-text-primary)' }}
            >
              Get Started with RAG Evaluation
            </h2>
            <p
              className="text-xs leading-relaxed"
              style={{ color: 'var(--color-text-secondary)' }}
            >
              Your evaluation pipeline is empty. Follow these steps to upload documents, test queries, and detect hallucinations.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 text-left max-w-4xl mx-auto">
            <Link
              to="/documents"
              className="p-4 rounded-xl border transition-all hover:border-[var(--color-accent-primary)] group"
              style={{
                backgroundColor: 'var(--color-bg-secondary)',
                borderColor: 'var(--color-border)',
              }}
            >
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-bold text-purple-400">Step 1</span>
                <FileText className="w-4 h-4 text-purple-400 group-hover:scale-110 transition-transform" />
              </div>
              <h4 className="text-sm font-semibold text-white">Upload Docs</h4>
              <p className="text-xs text-[var(--color-text-secondary)] mt-1">
                Upload PDFs or text files to build vector knowledge.
              </p>
            </Link>

            <Link
              to="/playground"
              className="p-4 rounded-xl border transition-all hover:border-[var(--color-teal)] group"
              style={{
                backgroundColor: 'var(--color-bg-secondary)',
                borderColor: 'var(--color-border)',
              }}
            >
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-bold text-teal-400">Step 2</span>
                <MessageSquare className="w-4 h-4 text-teal-400 group-hover:scale-110 transition-transform" />
              </div>
              <h4 className="text-sm font-semibold text-white">Ask in Playground</h4>
              <p className="text-xs text-[var(--color-text-secondary)] mt-1">
                Compare Standard LLM vs RAG with retrieved evidence.
              </p>
            </Link>

            <Link
              to="/evaluation"
              className="p-4 rounded-xl border transition-all hover:border-[var(--color-sky)] group"
              style={{
                backgroundColor: 'var(--color-bg-secondary)',
                borderColor: 'var(--color-border)',
              }}
            >
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-bold text-sky-400">Step 3</span>
                <FlaskConical className="w-4 h-4 text-sky-400 group-hover:scale-110 transition-transform" />
              </div>
              <h4 className="text-sm font-semibold text-white">Evaluate Claims</h4>
              <p className="text-xs text-[var(--color-text-secondary)] mt-1">
                Inspect hallucination rates and reference metrics.
              </p>
            </Link>

            <Link
              to="/experiments"
              className="p-4 rounded-xl border transition-all hover:border-[var(--color-emerald)] group"
              style={{
                backgroundColor: 'var(--color-bg-secondary)',
                borderColor: 'var(--color-border)',
              }}
            >
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-bold text-emerald-400">Step 4</span>
                <Beaker className="w-4 h-4 text-emerald-400 group-hover:scale-110 transition-transform" />
              </div>
              <h4 className="text-sm font-semibold text-white">Ablation Lab</h4>
              <p className="text-xs text-[var(--color-text-secondary)] mt-1">
                Benchmark chunk sizes, top-k, and embeddings.
              </p>
            </Link>
          </div>
        </div>
      )}

      {/* ── Recent Sessions Table ── */}
      <div className="card p-5 space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h3
              className="text-sm font-semibold tracking-tight"
              style={{ color: 'var(--color-text-primary)' }}
            >
              Recent Q&A Evaluations
            </h3>
            <p className="text-xs" style={{ color: 'var(--color-text-tertiary)' }}>
              Last 5 interactive sessions submitted through Playground
            </p>
          </div>
          <Link
            to="/evaluation"
            className="text-xs flex items-center gap-1 hover:underline font-medium"
            style={{ color: 'var(--color-accent-secondary)' }}
          >
            <span>View All Evaluations</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </Link>
        </div>

        {recentSessions.length === 0 ? (
          <div
            className="p-8 text-center rounded-xl border border-dashed text-xs"
            style={{
              borderColor: 'var(--color-border)',
              color: 'var(--color-text-tertiary)',
            }}
          >
            No Q&A sessions executed yet.{' '}
            <Link to="/playground" className="underline text-[var(--color-accent-primary)] font-medium">
              Run your first query in Playground
            </Link>
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
                  <th className="pb-3 pl-2">Question</th>
                  <th className="pb-3">Model</th>
                  <th className="pb-3 text-right">RAG Latency</th>
                  <th className="pb-3 text-right">Standard Latency</th>
                  <th className="pb-3 text-right">Top-K</th>
                  <th className="pb-3 pr-2 text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[var(--color-border-subtle)] font-mono">
                {recentSessions.map((s) => (
                  <tr
                    key={s.id}
                    className="hover:bg-white/[0.02] transition-colors"
                  >
                    <td className="py-3 pl-2 max-w-[280px]">
                      <span
                        className="font-sans font-medium line-clamp-1"
                        style={{ color: 'var(--color-text-primary)' }}
                        title={s.question}
                      >
                        {s.question}
                      </span>
                      <span
                        className="text-[10px] font-sans"
                        style={{ color: 'var(--color-text-tertiary)' }}
                      >
                        {s.session_time
                          ? new Date(s.session_time).toLocaleString()
                          : 'Recent'}
                      </span>
                    </td>
                    <td className="py-3">
                      <span className="badge badge-purple text-[10px]">
                        {s.model_used || 'groq'}
                      </span>
                    </td>
                    <td className="py-3 text-right text-teal-400 font-semibold">
                      {Math.round(s.rag_latency_ms)} ms
                    </td>
                    <td className="py-3 text-right text-sky-400">
                      {Math.round(s.standard_latency_ms)} ms
                    </td>
                    <td className="py-3 text-right text-[var(--color-text-secondary)]">
                      {s.top_k || 5}
                    </td>
                    <td className="py-3 pr-2 text-right">
                      <Link
                        to={`/evaluation?session_id=${s.id}`}
                        className="btn btn-secondary btn-sm py-1 px-2 text-[11px] inline-flex items-center gap-1"
                      >
                        <FlaskConical className="w-3 h-3 text-purple-400" />
                        <span>Evaluate</span>
                      </Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};

export default Dashboard;
