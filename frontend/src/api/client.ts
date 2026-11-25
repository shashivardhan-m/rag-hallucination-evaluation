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

// ── Experiments ────────────────────────────────────────────
export const experimentsAPI = {
  async run(config: ExperimentConfig): Promise<Experiment> {
    const { data } = await apiClient.post<Experiment>('/experiments/run', config, {
      timeout: 600_000, // 10 min for large experiments
    });
    return data;
  },

  async list(): Promise<Experiment[]> {
    const { data } = await apiClient.get<{ experiments: Experiment[] }>('/experiments/');
    return data.experiments;
  },

  async get(id: string): Promise<Experiment> {
    const { data } = await apiClient.get<Experiment>(`/experiments/${id}`);
    return data;
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
    const { data } = await apiClient.get<ModelComparison[]>('/analytics/model-comparison');
    return data;
  },

  async getMetricsOverTime(limit = 50): Promise<MetricTimeSeries[]> {
    const { data } = await apiClient.get<MetricTimeSeries[]>(`/analytics/metrics-over-time?limit=${limit}`);
    return data;
  },

  async getHallucinationTrend(limit = 50): Promise<MetricTimeSeries[]> {
    const { data } = await apiClient.get<MetricTimeSeries[]>(`/analytics/hallucination-trend?limit=${limit}`);
    return data;
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
    const { data } = await apiClient.get<AppSettings>('/settings');
    return data;
  },
  async update(settings: Partial<AppSettings>): Promise<AppSettings> {
    const { data } = await apiClient.patch<AppSettings>('/settings', settings);
    return data;
  },
};

// ── Health ─────────────────────────────────────────────────
export const healthAPI = {
  async check(): Promise<{ status: string; version: string }> {
    const { data } = await apiClient.get('/health');
    return data;
  },
};
