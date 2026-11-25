// ============================================================
// Axios API Client — typed endpoints for the FastAPI backend
// ============================================================
import axios, { AxiosError } from 'axios';
import type {
  DocumentInfo,
  DocumentUploadResponse,
  DocumentChunk,
  RetrievalQuery,
  RetrievalResult,
  VectorStoreStats,
  QARequest,
  QAResponse,
  QASession,
  EvaluationRequest,
  EvaluationResponse,
  MetricResult,
  MetricDefinition,
  ExperimentConfig,
  Experiment,
  ExperimentRun,
  AnalyticsSummary,
  ModelComparison,
  MetricTimeSeries,
  AppSettings,
} from '../types';

const BASE_URL = '/api';

export const apiClient = axios.create({
  baseURL: BASE_URL,
  headers: { 'Content-Type': 'application/json' },
  timeout: 120_000, // 2 min — LLM calls can be slow
});

// ── Error Handling ──────────────────────────────────────────
export function getErrorMessage(error: unknown): string {
  if (error instanceof AxiosError) {
    const detail = error.response?.data?.detail;
    if (typeof detail === 'string') return detail;
    if (Array.isArray(detail)) return detail.map((d: { msg: string }) => d.msg).join(', ');
    return error.message;
  }
  if (error instanceof Error) return error.message;
  return 'An unexpected error occurred';
}

// ── Documents ──────────────────────────────────────────────
export const documentsAPI = {
  async upload(file: File, description?: string): Promise<DocumentUploadResponse> {
    const form = new FormData();
    form.append('file', file);
    if (description) form.append('description', description);
    const { data } = await apiClient.post<DocumentUploadResponse>('/documents/upload', form, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return data;
  },

  async list(): Promise<DocumentInfo[]> {
    const { data } = await apiClient.get<{ documents: DocumentInfo[] }>('/documents/');
    return data.documents;
  },

  async get(id: string): Promise<DocumentInfo> {
    const { data } = await apiClient.get<DocumentInfo>(`/documents/${id}`);
    return data;
  },

  async delete(id: string): Promise<void> {
    await apiClient.delete(`/documents/${id}`);
  },

  async getChunks(id: string, page = 1, pageSize = 20): Promise<DocumentChunk[]> {
    const { data } = await apiClient.get<{ chunks: DocumentChunk[] }>(
      `/documents/${id}/chunks?page=${page}&page_size=${pageSize}`
    );
    return data.chunks;
  },
};

// ── Retrieval ──────────────────────────────────────────────
export const retrievalAPI = {
  async search(query: RetrievalQuery): Promise<RetrievalResult> {
    const { data } = await apiClient.post<RetrievalResult>('/retrieval/search', query);
    return data;
  },

  async getStats(): Promise<VectorStoreStats> {
    const { data } = await apiClient.get<VectorStoreStats>('/retrieval/stats');
    return data;
  },
};

export const qaAPI = {
  async ask(request: QARequest): Promise<QAResponse> {
    const { data } = await apiClient.post<QAResponse>('/qa/ask', request);
    return data;
  },

  async getSessions(limit = 50): Promise<QASession[]> {
    const { data } = await apiClient.get<{ total: number; sessions: QASession[] }>(`/qa/sessions?limit=${limit}`);
    return data.sessions;
  },

  async getSession(id: string): Promise<QASession> {
    const { data } = await apiClient.get<QASession>(`/qa/sessions/${id}`);
    return data;
  },
};

// ── Evaluation ─────────────────────────────────────────────
export const evaluationAPI = {
  async evaluate(request: EvaluationRequest): Promise<EvaluationResponse> {
    const { data } = await apiClient.post<any>('/evaluation/evaluate', request);
    
    // Normalize metrics dictionary to MetricResult list
    let metricList: MetricResult[] = [];
    let hr = 0.0;
    if (data.metrics && !Array.isArray(data.metrics)) {
      hr = data.metrics['hallucination_rate'] ?? (1.0 - (data.metrics['faithfulness'] ?? 1.0));
      metricList = Object.entries(data.metrics).map(([k, v]) => ({
        metric_name: k,
        metric_value: Number(v),
        is_reference_based: ['answer_correctness', 'rouge_l_f1', 'bertscore_f1', 'context_recall'].includes(k),
      }));
    } else if (Array.isArray(data.metrics)) {
      metricList = data.metrics;
      hr = data.hallucination_rate ?? 0.0;
    }

    return {
      evaluation_id: data.evaluation_id,
      session_id: data.qa_session_id,
      qa_session_id: data.qa_session_id,
      metrics: metricList,
      hallucination_rate: hr,
      evaluated_at: new Date().toISOString(),
      details: data.details,
    };
  },

  async getMetricDefinitions(): Promise<Record<string, MetricDefinition>> {
    const { data } = await apiClient.get<Record<string, MetricDefinition>>('/evaluation/metrics/definitions');
    return data;
  },

  async getResults(limit = 100): Promise<EvaluationResponse[]> {
    const { data } = await apiClient.get<{ total: number; results: any[] }>(`/evaluation/results?limit=${limit}`);
    const raw = data.results ?? [];

    // Group flat DB rows by qa_session_id or notes into EvaluationResponse objects
    const sessions = new Map<string, { metrics: MetricResult[]; evaluated_at: string; session_id: string }>();
    for (const r of raw) {
      const key = r.qa_session_id || r.notes || r.id;
      if (!sessions.has(key)) {
        sessions.set(key, {
          metrics: [],
          evaluated_at: r.computed_at,
          session_id: r.qa_session_id || key,
        });
      }
      sessions.get(key)!.metrics.push({
        metric_name: r.metric_name,
        metric_value: r.metric_value,
        is_reference_based: r.is_reference_based,
      });
    }

    const grouped: EvaluationResponse[] = [];
    sessions.forEach((val) => {
      const hr = val.metrics.find((m) => m.metric_name === 'hallucination_rate')?.metric_value
        ?? (1.0 - (val.metrics.find((m) => m.metric_name === 'faithfulness')?.metric_value ?? 1.0));
      grouped.push({
        session_id: val.session_id,
        metrics: val.metrics,
        hallucination_rate: hr,
        evaluated_at: val.evaluated_at,
      });
    });

    return grouped;
  },

  async getSessionMetrics(sessionId: string): Promise<EvaluationResponse> {
    const { data } = await apiClient.get<any>(`/evaluation/sessions/${sessionId}/metrics`);
    let metricList: MetricResult[] = [];
    let hr = 0.0;
    if (data.metrics && !Array.isArray(data.metrics)) {
      hr = data.metrics['hallucination_rate'] ?? (1.0 - (data.metrics['faithfulness'] ?? 1.0));
      metricList = Object.entries(data.metrics).map(([k, v]) => ({
        metric_name: k,
        metric_value: Number(v),
        is_reference_based: ['answer_correctness', 'rouge_l_f1', 'bertscore_f1', 'context_recall'].includes(k),
      }));
    }
    return {
      session_id: sessionId,
      metrics: metricList,
      hallucination_rate: hr,
      evaluated_at: new Date().toISOString(),
    };
  },
};

function normalizeExperiment(data: any): Experiment {
  if (!data) return data;
  let runs: ExperimentRun[] = data.runs || [];
  if ((!runs || runs.length === 0) && data.results?.conditions) {
    runs = [];
    for (const cond of data.results.conditions) {
      if (cond.queries && cond.queries.length > 0) {
        for (const q of cond.queries) {
          runs.push({
            variable_value: cond.condition_value,
            question: q.question,
            answer: q.answer,
            metrics: q.metrics || {},
            latency_ms: q.latency_ms || cond.average_latency_ms || 0,
          });
        }
      } else {
        runs.push({
          variable_value: cond.condition_value,
          question: `Condition: ${cond.condition_value}`,
          answer: `Average faithfulness: ${cond.average_metrics?.faithfulness ?? 'N/A'}`,
          metrics: cond.average_metrics || {},
          latency_ms: cond.average_latency_ms || 0,
        });
      }
    }
  }
  return {
    ...data,
    runs,
  };
}

// ── Experiments ────────────────────────────────────────────
export const experimentsAPI = {
  async run(config: ExperimentConfig | any): Promise<Experiment> {
    const payload = config.config ? config : {
      name: config.name || 'Experiment',
      description: config.description,
      config: {
        variable: config.variable,
        values: config.values,
        questions: config.questions,
        reference_answers: config.reference_answers,
        benchmark_dataset: config.benchmark_dataset,
        num_runs: config.num_runs,
        default_top_k: config.base_config?.top_k,
        default_retrieval_mode: config.base_config?.retrieval_mode,
        default_use_reranking: config.base_config?.use_reranking,
        backend: config.base_config?.llm_backend,
        model: config.base_config?.llm_model,
      },
    };
    const { data } = await apiClient.post<any>('/experiments/run', payload, {
      timeout: 600_000, // 10 min for large experiments
    });
    return normalizeExperiment(data);
  },

  async list(): Promise<Experiment[]> {
    const { data } = await apiClient.get<any>('/experiments/');
    const list = data?.experiments || (Array.isArray(data) ? data : []);
    return list.map(normalizeExperiment);
  },

  async get(id: string): Promise<Experiment> {
    const { data } = await apiClient.get<any>(`/experiments/${id}`);
    return normalizeExperiment(data);
  },

  async delete(id: string): Promise<void> {
    await apiClient.delete(`/experiments/${id}`);
  },
};

// ── Analytics ──────────────────────────────────────────────
export const analyticsAPI = {
  async getSummary(): Promise<AnalyticsSummary> {
    const { data } = await apiClient.get<AnalyticsSummary>('/analytics/summary');
    return data;
  },

  async getModelComparison(): Promise<ModelComparison[]> {
    const { data } = await apiClient.get<any>('/analytics/model-comparison');
    const rawList = Array.isArray(data) ? data : (data?.models || []);
    return rawList.map((m: any) => ({
      model: m.model || m.model_name || 'default',
      backend: m.backend || 'groq',
      avg_faithfulness: m.avg_faithfulness ?? m.faithfulness ?? 0.0,
      avg_relevance: m.avg_relevance ?? m.answer_relevance ?? 0.0,
      avg_latency_ms: m.avg_latency_ms ?? m.avg_rag_latency_ms ?? m.avg_standard_latency_ms ?? 0.0,
      num_sessions: m.num_sessions ?? m.session_count ?? 0,
      ...m,
    }));
  },

  async getMetricsOverTime(limit = 50, metric = 'faithfulness'): Promise<any[]> {
    const { data } = await apiClient.get<any>(`/analytics/metrics-over-time?metric_name=${metric}&limit=${limit}`);
    const points = Array.isArray(data) ? data : (data?.points || []);
    return points.map((p: any) => ({
      timestamp: p.timestamp ? new Date(p.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : '',
      [metric]: p.value,
      value: p.value,
    }));
  },

  async getHallucinationTrend(limit = 50): Promise<any[]> {
    const { data } = await apiClient.get<any>(`/analytics/hallucination-trend?limit=${limit}`);
    const points = Array.isArray(data) ? data : (data?.points || []);
    return points.map((p: any) => ({
      timestamp: p.timestamp ? new Date(p.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : '',
      hallucination_rate: p.value,
      value: p.value,
    }));
  },

  async getHallucinationTaxonomy(): Promise<{ taxonomy: Record<string, number>; descriptions: Record<string, string> }> {
    const { data } = await apiClient.get<any>('/analytics/hallucination-taxonomy');
    return data;
  },

  async getResearchFindings(): Promise<any> {
    const { data } = await apiClient.get<any>('/analytics/research-findings');
    return data;
  },
};

// ── Reports ────────────────────────────────────────────────
export const reportsAPI = {
  downloadCSV(): void {
    window.open(`${BASE_URL}/reports/export/csv`, '_blank');
  },
  downloadJSON(): void {
    window.open(`${BASE_URL}/reports/export/json`, '_blank');
  },
  async getTextReport(): Promise<string> {
    const { data } = await apiClient.get<any>('/reports/report/text');
    return typeof data === 'string' ? data : (data.report || JSON.stringify(data, null, 2));
  },
};

// ── Settings ───────────────────────────────────────────────
export const settingsAPI = {
  async get(): Promise<AppSettings> {
    const { data } = await apiClient.get<any>('/settings');
    return {
      llm_backend: data.llm_backend || data.default_llm_backend || 'groq',
      llm_model: data.llm_model || data.default_groq_model || 'llama3-8b-8192',
      embedding_model: data.embedding_model || 'BAAI/bge-base-en-v1.5',
      chunk_size: data.chunk_size || data.default_chunk_size || 512,
      chunk_overlap: data.chunk_overlap || data.default_chunk_overlap || 128,
      top_k: data.top_k || data.default_top_k || 5,
      use_reranking: data.use_reranking ?? false,
      retrieval_mode: data.retrieval_mode || 'semantic',
      ...data,
    };
  },
  async update(settings: Partial<AppSettings>): Promise<AppSettings> {
    const { data } = await apiClient.patch<any>('/settings', settings);
    return {
      llm_backend: data.llm_backend || data.default_llm_backend || 'groq',
      llm_model: data.llm_model || data.default_groq_model || 'llama3-8b-8192',
      embedding_model: data.embedding_model || 'BAAI/bge-base-en-v1.5',
      chunk_size: data.chunk_size || data.default_chunk_size || 512,
      chunk_overlap: data.chunk_overlap || data.default_chunk_overlap || 128,
      top_k: data.top_k || data.default_top_k || 5,
      use_reranking: data.use_reranking ?? false,
      retrieval_mode: data.retrieval_mode || 'semantic',
      ...data,
    };
  },
};

// ── Health ─────────────────────────────────────────────────
export const healthAPI = {
  async check(): Promise<{ status: string; version: string }> {
    const { data } = await apiClient.get('/health');
    return data;
  },
};
