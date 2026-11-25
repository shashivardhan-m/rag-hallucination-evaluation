// ============================================================
// Shared TypeScript Types for RAG Hallucination Evaluator
// ============================================================

// ── Documents ──────────────────────────────────────────────

export interface DocumentInfo {
  id: string;
  filename: string;
  file_type: string;
  file_size: number;
  num_chunks: number;
  upload_time: string;
  description?: string;
}

export interface DocumentChunk {
  id: string;
  document_id: string;
  chunk_index: number;
  content: string;
  char_count: number;
  token_estimate: number;
  embedding_model: string;
}

// Actual API response from POST /api/documents/upload (HTTP 201)
export interface DocumentUploadResponse extends DocumentInfo {
  chunks_indexed?: number;  // alias for num_chunks in upload response
  message: string;
}

// ── Retrieval ──────────────────────────────────────────────

export interface RetrievedChunk {
  chunk_id: string;
  document_id: string;
  filename: string;
  chunk_index: number;
  content: string;
  score: number;           // actual API field name
  rank?: number;
  similarity_score?: number; // alias kept for compat
  char_count?: number;
}

export interface RetrievalQuery {
  query: string;
  top_k?: number;
  mode?: 'semantic' | 'hybrid';
  use_reranking?: boolean;
}

export interface RetrievalResult {
  query: string;
  chunks: RetrievedChunk[];  // actual API field (not "results")
  total_retrieved: number;
  retrieval_mode: string;
  reranked: boolean;
  latency_ms: number;
}

export interface VectorStoreStats {
  total_chunks: number;
  total_documents: number;
  embedding_model: string;
  index_type: string;
}

// ── Q&A ────────────────────────────────────────────────────

export interface QARequest {
  question: string;
  top_k?: number;
  retrieval_mode?: 'semantic' | 'hybrid';
  use_reranking?: boolean;
  llm_backend?: string;
  llm_model?: string;
  reference_answer?: string;
}

export interface AnswerResult {
  answer: string;
  latency_ms: number;
  model: string;
  backend: string;
  retrieved_chunks?: RetrievedChunk[];
  token_count?: number;
}

// Matches actual /api/qa/ask response schema (QAResponse in backend)
export interface QAResponse {
  session_id: string;
  question: string;
  standard_answer: AnswerResult | null;  // key: standard_answer (not "standard")
  rag_answer: AnswerResult | null;       // key: rag_answer (not "rag")
  created_at: string;
}

// Matches QASessionInfo (list view from /api/qa/sessions)
export interface QASession {
  id: string;
  question: string;
  standard_answer: string | null;   // stored as text in DB
  rag_answer: string | null;
  standard_latency_ms: number;
  rag_latency_ms: number;
  model_used: string | null;
  session_time: string;             // actual DB field name
  top_k: number;
}

// ── Evaluation ─────────────────────────────────────────────

export interface MetricDefinition {
  name: string;
  description: string;
  formula: string;
  range: string;
  is_reference_based: boolean;
  higher_is_better: boolean;
}

export interface MetricResult {
  metric_name: string;
  metric_value: number;
  is_reference_based: boolean;
  details?: Record<string, number>;
}

// Matches actual EvaluationRequest schema (required: question, answer)
export interface EvaluationRequest {
  qa_session_id?: string;       // optional session ID to persist result
  question: string;
  answer: string;               // the RAG answer to evaluate
  context_chunks: string[];     // retrieved context texts
  reference_answer?: string;    // optional ground truth
  retrieved_ids?: string[];     // optional for precision/recall
  relevant_ids?: string[];
}

export interface EvaluationResponse {
  evaluation_id?: string;
  session_id?: string;        // present when linked to a QA session
  qa_session_id?: string;     // alias
  metrics: MetricResult[];
  hallucination_rate: number;
  evaluated_at: string;
  details?: Record<string, any>;
}

// ── Experiments ────────────────────────────────────────────

export type ExperimentVariable =
  | 'chunk_size'
  | 'embedding_model'
  | 'retrieval_mode'
  | 'reranking'
  | 'top_k'
  | 'llm_model'
  | 'context_relevance';

export interface ExperimentConfig {
  name: string;
  description?: string;
  variable: ExperimentVariable;
  values: (string | number | boolean)[];
  questions: string[];
  reference_answers?: string[];
  benchmark_dataset?: 'ragtruth' | 'halueval' | 'all' | string;
  num_runs?: number;
  base_config?: {
    chunk_size?: number;
    chunk_overlap?: number;
    embedding_model?: string;
    retrieval_mode?: 'semantic' | 'bm25' | 'hybrid' | string;
    use_reranking?: boolean;
    top_k?: number;
    llm_backend?: string;
    llm_model?: string;
  };
}

export interface ExperimentRun {
  variable_value: string | number | boolean;
  question: string;
  answer: string;
  metrics: Record<string, number>;
  latency_ms: number;
}

export interface Experiment {
  id: string;
  name: string;
  description?: string;
  config: ExperimentConfig;
  runs: ExperimentRun[];
  status: 'pending' | 'running' | 'completed' | 'failed';
  created_at: string;
  completed_at?: string;
  num_queries: number;
  error?: string;
}

// ── Analytics ──────────────────────────────────────────────

export interface AnalyticsSummary {
  total_documents: number;
  total_chunks: number;
  total_qa_sessions: number;         // actual backend field
  total_questions?: number;          // alias used in some pages (= total_qa_sessions)
  total_evaluations: number;
  total_experiments: number;
  avg_faithfulness: number;
  avg_answer_relevance: number;
  avg_answer_correctness: number;
  avg_correctness?: number;          // alias for avg_answer_correctness
  avg_hallucination_rate: number;    // actual backend field
  hallucination_rate?: number;       // alias used in some pages (= avg_hallucination_rate)
  avg_rag_latency_ms: number;
  avg_standard_latency_ms: number;
  latency_improvement_pct?: number;  // computed on frontend if needed
}

export interface ModelComparison {
  model: string;
  backend: string;
  avg_faithfulness: number;
  avg_relevance: number;
  avg_latency_ms: number;
  num_sessions: number;
}

export interface MetricTimeSeries {
  timestamp: string;
  faithfulness?: number;
  answer_relevance?: number;
  hallucination_rate?: number;
}

// ── Settings ───────────────────────────────────────────────

export interface AppSettings {
  llm_backend: string;
  llm_model: string;
  embedding_model: string;
  chunk_size: number;
  chunk_overlap: number;
  top_k: number;
  use_reranking: boolean;
  retrieval_mode: 'semantic' | 'hybrid';
}

// ── API Errors ─────────────────────────────────────────────

export interface APIError {
  detail: string;
  code?: string;
}

// ── UI State ───────────────────────────────────────────────

export type LoadingState = 'idle' | 'loading' | 'success' | 'error';

export interface AsyncState<T> {
  data: T | null;
  status: LoadingState;
  error: string | null;
}
