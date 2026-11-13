import React, { useState } from 'react';
import {
  Send,
  Sparkles,
  Zap,
  Clock,
  Layers,
  FlaskConical,
  RotateCcw,
  Sliders,
  ChevronDown,
  ChevronUp,
  Bookmark,
  AlertCircle,
  Copy,
  Check,
} from 'lucide-react';
import { qaAPI, evaluationAPI, getErrorMessage } from '../api/client';
import { useSettingsStore } from '../store/settingsStore';
import type { QAResponse, EvaluationResponse } from '../types';
import { EvidencePanel } from '../components/ui/EvidencePanel';
import { MetricBadge } from '../components/ui/MetricBadge';

const SAMPLE_QUESTIONS = [
  'What are the primary findings and numerical results presented in the document?',
  'Explain the methodology and experimental setup used in this study.',
  'What are the main limitations identified by the authors?',
  'How does the proposed approach compare with previous state-of-the-art baselines?',
];

export const Playground: React.FC = () => {
  const settings = useSettingsStore();

  // Query configuration state
  const [question, setQuestion] = useState('');
  const [referenceAnswer, setReferenceAnswer] = useState('');
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [showRefInput, setShowRefInput] = useState(false);

  // Local overrides (defaults from global settings)
  const [backend, setBackend] = useState(settings.llmBackend);
  const [model, setModel] = useState(settings.llmModel);
  const [topK, setTopK] = useState(settings.topK);
  const [retrievalMode, setRetrievalMode] = useState<'semantic' | 'hybrid'>(settings.retrievalMode);
  const [useReranking, setUseReranking] = useState(settings.useReranking);

  // Execution states
  const [isLoading, setIsLoading] = useState(false);
  const [response, setResponse] = useState<QAResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Evaluation states
  const [isEvaluating, setIsEvaluating] = useState(false);
  const [evalResult, setEvalResult] = useState<EvaluationResponse | null>(null);
  const [evalError, setEvalError] = useState<string | null>(null);

  // Copy states
  const [copiedStd, setCopiedStd] = useState(false);
  const [copiedRag, setCopiedRag] = useState(false);

  const handleAsk = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!question.trim() || isLoading) return;

    setIsLoading(true);
    setError(null);
    setResponse(null);
    setEvalResult(null);
    setEvalError(null);

    try {
      const res = await qaAPI.ask({
        question: question.trim(),
        top_k: topK,
        retrieval_mode: retrievalMode,
        use_reranking: useReranking,
        llm_backend: backend,
        llm_model: model,
        reference_answer: referenceAnswer.trim() || undefined,
      });
      setResponse(res);
    } catch (err) {
      setError(getErrorMessage(err));
    } finally {
      setIsLoading(false);
    }
  };

  const handleEvaluate = async () => {
    if (!response || isEvaluating) return;

    setIsEvaluating(true);
    setEvalError(null);

    try {
      const contextChunks = response.rag_answer?.retrieved_chunks
        ? response.rag_answer?.retrieved_chunks.map((c) => c.content)
        : [];

      const evalData = await evaluationAPI.evaluate({
        qa_session_id: response.session_id,
        question: response.question,
        answer: response.rag_answer?.answer ?? "",
        context_chunks: contextChunks,
        reference_answer: referenceAnswer.trim() || undefined,
      });

      setEvalResult(evalData);
    } catch (err) {
      setEvalError(getErrorMessage(err));
    } finally {
      setIsEvaluating(false);
    }
  };

  const copyText = (text: string, type: 'std' | 'rag') => {
    navigator.clipboard.writeText(text);
    if (type === 'std') {
      setCopiedStd(true);
      setTimeout(() => setCopiedStd(false), 2000);
    } else {
      setCopiedRag(true);
      setTimeout(() => setCopiedRag(false), 2000);
    }
  };

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
              RAG vs Standard LLM Playground
            </h1>
            <span className="badge badge-purple text-xs">Side-by-Side</span>
          </div>
          <p
            className="text-xs sm:text-sm mt-1"
            style={{ color: 'var(--color-text-secondary)' }}
          >
            Submit identical prompts simultaneously to standard direct generation and context-augmented RAG.
          </p>
        </div>

        {response && (
          <button
            type="button"
            onClick={() => {
              setResponse(null);
              setEvalResult(null);
              setQuestion('');
            }}
            className="btn btn-secondary btn-sm flex items-center gap-1.5 self-start sm:self-auto"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            <span>Clear Session</span>
          </button>
        )}
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
          <AlertCircle className="w-4 h-4 flex-shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* ── Two Column Layout (40% Left / 60% Right) ── */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* ── Left Configuration & Prompt Panel (40%) ── */}
        <div className="lg:col-span-5 space-y-4">
          <div
            className="card p-5 space-y-4 shadow-sm"
            style={{
              backgroundColor: 'var(--color-bg-card)',
              borderColor: 'var(--color-border)',
            }}
          >
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold uppercase tracking-wider text-[var(--color-text-secondary)]">
                Query Formulation
              </span>
              <span className="text-[11px] font-mono text-[var(--color-text-tertiary)]">
                Ctrl + Enter to send
              </span>
            </div>

            {/* Prompt Textarea */}
            <div className="form-group">
              <textarea
                rows={5}
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
                    handleAsk();
                  }
                }}
                placeholder="Ask a technical or factual question based on your indexed knowledge base..."
                className="input resize-none text-xs leading-relaxed"
                style={{ minHeight: '110px' }}
              />
            </div>

            {/* Sample Question Pills */}
            <div className="space-y-1.5">
              <span className="text-[11px] font-medium text-[var(--color-text-tertiary)]">
                Sample Prompts:
              </span>
              <div className="flex flex-wrap gap-1.5">
                {SAMPLE_QUESTIONS.map((q, idx) => (
                  <button
                    key={idx}
                    type="button"
                    onClick={() => setQuestion(q)}
                    className="text-[11px] text-left px-2.5 py-1 rounded-md border truncate max-w-full hover:border-[var(--color-accent-primary)] hover:text-white transition-all"
                    style={{
                      backgroundColor: 'var(--color-bg-secondary)',
                      borderColor: 'var(--color-border)',
                      color: 'var(--color-text-secondary)',
                    }}
                  >
                    {q}
                  </button>
                ))}
              </div>
            </div>

            {/* Optional Reference Answer Accordion */}
            <div className="pt-2 border-t border-[var(--color-border-subtle)]">
              <button
                type="button"
                onClick={() => setShowRefInput(!showRefInput)}
                className="text-xs font-medium flex items-center justify-between w-full hover:text-white transition-colors"
                style={{ color: 'var(--color-text-secondary)' }}
              >
                <span className="flex items-center gap-1.5">
                  <Bookmark className="w-3.5 h-3.5 text-teal-400" />
                  <span>Ground Truth Reference Answer (Optional)</span>
                </span>
                {showRefInput ? (
                  <ChevronUp className="w-3.5 h-3.5" />
                ) : (
                  <ChevronDown className="w-3.5 h-3.5" />
                )}
              </button>

              {showRefInput && (
                <div className="mt-2.5 space-y-1">
                  <textarea
                    rows={3}
                    value={referenceAnswer}
                    onChange={(e) => setReferenceAnswer(e.target.value)}
                    placeholder="Enter the golden / ground truth answer to calculate reference-based metrics (e.g. Correctness, ROUGE)..."
                    className="input text-xs resize-none"
                  />
                </div>
              )}
            </div>

            {/* Retrieval & Model Settings Accordion */}
            <div className="pt-2 border-t border-[var(--color-border-subtle)]">
              <button
                type="button"
                onClick={() => setShowAdvanced(!showAdvanced)}
                className="text-xs font-medium flex items-center justify-between w-full hover:text-white transition-colors"
                style={{ color: 'var(--color-text-secondary)' }}
              >
                <span className="flex items-center gap-1.5">
                  <Sliders className="w-3.5 h-3.5 text-purple-400" />
                  <span>Retrieval & Model Parameters</span>
                </span>
                {showAdvanced ? (
                  <ChevronUp className="w-3.5 h-3.5" />
                ) : (
                  <ChevronDown className="w-3.5 h-3.5" />
                )}
              </button>

              {showAdvanced && (
                <div className="mt-3 space-y-3 text-xs">
                  {/* Top-K Slider */}
                  <div>
                    <div className="flex justify-between mb-1">
                      <label className="label">Top-K Chunks</label>
                      <span className="font-mono text-purple-400 font-bold">
                        {topK}
                      </span>
                    </div>
                    <input
                      type="range"
                      min="1"
                      max="15"
                      value={topK}
                      onChange={(e) => setTopK(Number(e.target.value))}
                      className="w-full accent-[var(--color-accent-primary)] cursor-pointer"
                    />
                  </div>

                  {/* Retrieval Mode */}
                  <div className="form-group">
                    <label className="label">Retrieval Mode</label>
                    <select
                      value={retrievalMode}
                      onChange={(e) =>
                        setRetrievalMode(e.target.value as 'semantic' | 'hybrid')
                      }
                      className="select text-xs py-1.5"
                    >
                      <option value="semantic">Dense Semantic (Embedding Cosine)</option>
                      <option value="hybrid">Hybrid (Dense + BM25 Keyword)</option>
                    </select>
                  </div>

                  {/* Reranking Toggle */}
                  <div className="flex items-center justify-between pt-1">
                    <div>
                      <span className="font-medium text-[var(--color-text-secondary)]">
                        Cross-Encoder Reranking
                      </span>
                      <p className="text-[10px] text-[var(--color-text-tertiary)]">
                        Re-scores top-k using cross-encoder
                      </p>
                    </div>
                    <input
                      type="checkbox"
                      checked={useReranking}
                      onChange={(e) => setUseReranking(e.target.checked)}
                      className="w-4 h-4 rounded accent-[var(--color-accent-primary)] cursor-pointer"
                    />
                  </div>

                  {/* LLM Backend & Model */}
                  <div className="grid grid-cols-2 gap-2 pt-1 border-t border-[var(--color-border-subtle)]">
                    <div className="form-group">
                      <label className="label">LLM Backend</label>
                      <select
                        value={backend}
                        onChange={(e) => setBackend(e.target.value)}
                        className="select text-xs py-1"
                      >
                        <option value="groq">Groq Cloud</option>
                        <option value="openai">OpenAI</option>
                        <option value="ollama">Local Ollama</option>
                        <option value="huggingface">Hugging Face</option>
                      </select>
                    </div>

                    <div className="form-group">
                      <label className="label">Model Name</label>
                      <input
                        type="text"
                        value={model}
                        onChange={(e) => setModel(e.target.value)}
                        className="input text-xs py-1 font-mono"
                        placeholder="llama3-8b-8192"
                      />
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Submit Button */}
            <button
              type="button"
              onClick={() => handleAsk()}
              disabled={isLoading || !question.trim()}
              className="btn btn-primary w-full py-2.5 justify-center flex items-center gap-2 text-sm font-semibold shadow-md"
            >
              {isLoading ? (
                <>
                  <div className="spinner !w-4 !h-4" />
                  <span>Executing Parallel Generations...</span>
                </>
              ) : (
                <>
                  <Send className="w-4 h-4" />
                  <span>Ask Standard LLM & RAG</span>
                </>
              )}
            </button>
          </div>
        </div>

        {/* ── Right Comparison & Evidence Panel (60%) ── */}
        <div className="lg:col-span-7 space-y-4">
          {isLoading && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="card p-5 space-y-3 animate-pulse">
                <div className="flex items-center justify-between">
                  <div className="w-28 h-5 skeleton" />
                  <div className="w-16 h-4 skeleton" />
                </div>
                <div className="space-y-2 pt-2">
                  <div className="w-full h-3 skeleton" />
                  <div className="w-5/6 h-3 skeleton" />
                  <div className="w-4/6 h-3 skeleton" />
                </div>
                <div className="pt-4 flex items-center gap-2 text-xs text-[var(--color-text-tertiary)]">
                  <div className="spinner !w-3 !h-3" />
                  <span>Calling Standard LLM...</span>
                </div>
              </div>

              <div className="card p-5 space-y-3 animate-pulse">
                <div className="flex items-center justify-between">
                  <div className="w-28 h-5 skeleton" />
                  <div className="w-16 h-4 skeleton" />
                </div>
                <div className="space-y-2 pt-2">
                  <div className="w-full h-3 skeleton" />
                  <div className="w-5/6 h-3 skeleton" />
                  <div className="w-4/6 h-3 skeleton" />
                </div>
                <div className="pt-4 flex items-center gap-2 text-xs text-teal-400">
                  <div className="spinner !w-3 !h-3" />
                  <span>Retrieving + Generating Context...</span>
                </div>
              </div>
            </div>
          )}

          {!isLoading && !response && (
            <div
              className="card p-12 text-center rounded-2xl border border-dashed flex flex-col items-center justify-center gap-3"
              style={{
                borderColor: 'var(--color-border)',
                backgroundColor: 'var(--color-bg-secondary)',
              }}
            >
              <Sparkles className="w-10 h-10 text-purple-400 opacity-30" />
              <div className="space-y-1">
                <h3 className="text-sm font-semibold text-white">
                  Playground Ready
                </h3>
                <p className="text-xs text-[var(--color-text-secondary)] max-w-md">
                  Type a prompt on the left or select a sample question to trigger dual generation and compare contextual groundedness.
                </p>
              </div>
            </div>
          )}

          {response && (
            <div className="space-y-4">
              {/* Dual Response Cards */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* ── Standard LLM Card ── */}
                <div
                  className="card p-5 flex flex-col justify-between border space-y-4"
                  style={{
                    backgroundColor: 'var(--color-bg-card)',
                    borderColor: 'rgba(248, 113, 113, 0.2)',
                  }}
                >
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <span className="badge badge-rose font-mono text-[11px] font-semibold">
                          Standard LLM
                        </span>
                        <span className="text-[10px] text-[var(--color-text-tertiary)] font-mono">
                          Zero-Shot
                        </span>
                      </div>
                      <button
                        type="button"
                        onClick={() => copyText(response.standard_answer?.answer ?? '', 'std')}
                        className="btn btn-ghost btn-sm p-1 text-[11px]"
                        title="Copy answer"
                      >
                        {copiedStd ? (
                          <Check className="w-3.5 h-3.5 text-emerald-400" />
                        ) : (
                          <Copy className="w-3.5 h-3.5" />
                        )}
                      </button>
                    </div>

                    <div className="flex items-center gap-2 text-[11px] font-mono text-[var(--color-text-tertiary)]">
                      <span className="flex items-center gap-1">
                        <Clock className="w-3 h-3 text-sky-400" />
                        {Math.round(response.standard_answer?.latency_ms ?? 0)} ms
                      </span>
                      <span>•</span>
                      <span className="truncate max-w-[120px]">
                        {response.standard_answer?.model}
                      </span>
                    </div>

                    <div
                      className="p-3.5 rounded-xl border text-xs leading-relaxed select-text"
                      style={{
                        backgroundColor: 'var(--color-bg-tertiary)',
                        borderColor: 'var(--color-border-subtle)',
                        color: 'var(--color-text-primary)',
                      }}
                    >
                      {response.standard_answer?.answer}
                    </div>
                  </div>

                  <div className="text-[10px] text-[var(--color-text-muted)] italic">
                    Answer synthesized directly from internal parametric weights without retrieved evidence.
                  </div>
                </div>

                {/* ── RAG Card ── */}
                <div
                  className="card p-5 flex flex-col justify-between border space-y-4"
                  style={{
                    backgroundColor: 'var(--color-bg-card)',
                    borderColor: 'rgba(45, 212, 191, 0.25)',
                  }}
                >
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <span className="badge badge-teal font-mono text-[11px] font-semibold">
                          RAG Pipeline
                        </span>
                        <span className="text-[10px] text-teal-400 font-mono">
                          Grounded
                        </span>
                      </div>
                      <button
                        type="button"
                        onClick={() => copyText(response.rag_answer?.answer ?? '', 'rag')}
                        className="btn btn-ghost btn-sm p-1 text-[11px]"
                        title="Copy answer"
                      >
                        {copiedRag ? (
                          <Check className="w-3.5 h-3.5 text-emerald-400" />
                        ) : (
                          <Copy className="w-3.5 h-3.5" />
                        )}
                      </button>
                    </div>

                    <div className="flex items-center gap-2 text-[11px] font-mono text-[var(--color-text-tertiary)]">
                      <span className="flex items-center gap-1 text-teal-400">
                        <Zap className="w-3 h-3" />
                        {Math.round(response.rag_answer?.latency_ms ?? 0)} ms
                      </span>
                      <span>•</span>
                      <span className="truncate max-w-[120px]">
                        {response.rag_answer?.model}
                      </span>
                    </div>

                    <div
                      className="p-3.5 rounded-xl border text-xs leading-relaxed select-text"
                      style={{
                        backgroundColor: 'var(--color-bg-tertiary)',
                        borderColor: 'var(--color-border-subtle)',
                        color: 'var(--color-text-primary)',
                      }}
                    >
                      {response.rag_answer?.answer}
                    </div>
                  </div>

                  <div className="text-[10px] text-teal-400/80 italic">
                    Answer grounded in {response.rag_answer?.retrieved_chunks?.length || 0} retrieved context chunks.
                  </div>
                </div>
              </div>

              {/* ── Evaluation Action & Results Bar ── */}
              <div
                className="card p-4 space-y-3 border"
                style={{
                  backgroundColor: 'var(--color-bg-secondary)',
                  borderColor: 'var(--color-border)',
                }}
              >
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                  <div className="space-y-0.5">
                    <div className="flex items-center gap-2">
                      <FlaskConical className="w-4 h-4 text-purple-400" />
                      <h4 className="text-xs font-bold text-white uppercase tracking-wider">
                        Automated Hallucination Evaluation
                      </h4>
                    </div>
                    <p className="text-[11px] text-[var(--color-text-tertiary)]">
                      Calculate Faithfulness, Claim Verification, and Hallucination Index on the RAG answer.
                    </p>
                  </div>

                  <button
                    type="button"
                    onClick={handleEvaluate}
                    disabled={isEvaluating}
                    className="btn btn-secondary btn-sm flex items-center gap-1.5 self-start sm:self-auto font-semibold"
                  >
                    {isEvaluating ? (
                      <>
                        <div className="spinner !w-3.5 !h-3.5" />
                        <span>Evaluating Claims...</span>
                      </>
                    ) : (
                      <>
                        <Sparkles className="w-3.5 h-3.5 text-teal-400" />
                        <span>Evaluate This Response</span>
                      </>
                    )}
                  </button>
                </div>

                {evalError && (
                  <p className="text-xs text-rose-400">{evalError}</p>
                )}

                {/* Display Computed Metrics */}
                {evalResult && (
                  <div className="pt-3 border-t border-[var(--color-border-subtle)] space-y-2">
                    <span className="text-[11px] font-semibold text-[var(--color-text-secondary)] uppercase">
                      Computed Verification Scores
                    </span>
                    <div className="flex flex-wrap gap-2">
                      {evalResult.metrics.map((m, i) => (
                        <MetricBadge
                          key={i}
                          name={m.metric_name}
                          value={m.metric_value}
                          isReferenceBased={m.is_reference_based}
                        />
                      ))}
                      <MetricBadge
                        name="Hallucination Rate"
                        value={evalResult.hallucination_rate}
                        isReferenceBased={false}
                      />
                    </div>
                  </div>
                )}
              </div>

              {/* ── Evidence Used Section ── */}
              <div
                className="card p-5 space-y-3"
                style={{
                  backgroundColor: 'var(--color-bg-card)',
                  borderColor: 'var(--color-border)',
                }}
              >
                <div className="flex items-center gap-2 mb-2">
                  <Layers className="w-4 h-4 text-teal-400" />
                  <h3 className="text-sm font-semibold text-white">
                    Evidence Retrieved from Knowledge Base
                  </h3>
                </div>

                <EvidencePanel
                  chunks={response.rag_answer?.retrieved_chunks || []}
                  loading={false}
                />
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Playground;
