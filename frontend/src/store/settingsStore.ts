// ============================================================
// Global Settings Store — Zustand
// ============================================================
import { create } from 'zustand';
import { persist } from 'zustand/middleware';

type Theme = 'dark' | 'light';

interface SettingsState {
  // Theme
  theme: Theme;
  setTheme: (theme: Theme) => void;
  toggleTheme: () => void;

  // LLM Configuration
  llmBackend: string;
  llmModel: string;
  setLLMBackend: (backend: string) => void;
  setLLMModel: (model: string) => void;

  // RAG Configuration
  embeddingModel: string;
  chunkSize: number;
  chunkOverlap: number;
  topK: number;
  useReranking: boolean;
  retrievalMode: 'semantic' | 'hybrid';
  setEmbeddingModel: (model: string) => void;
  setChunkSize: (size: number) => void;
  setChunkOverlap: (overlap: number) => void;
  setTopK: (k: number) => void;
  setUseReranking: (v: boolean) => void;
  setRetrievalMode: (mode: 'semantic' | 'hybrid') => void;

  // UI state
  sidebarCollapsed: boolean;
  toggleSidebar: () => void;
}

export const useSettingsStore = create<SettingsState>()(
  persist(
    (set, get) => ({
      // Theme
      theme: 'dark',
      setTheme: (theme) => {
        set({ theme });
        document.documentElement.classList.toggle('light', theme === 'light');
      },
      toggleTheme: () => {
        const next = get().theme === 'dark' ? 'light' : 'dark';
        get().setTheme(next);
      },

      // LLM
      llmBackend: 'groq',
      llmModel: 'llama3-8b-8192',
      setLLMBackend: (llmBackend) => set({ llmBackend }),
      setLLMModel: (llmModel) => set({ llmModel }),

      // RAG
      embeddingModel: 'BAAI/bge-base-en-v1.5',
      chunkSize: 512,
      chunkOverlap: 128,
      topK: 5,
      useReranking: false,
      retrievalMode: 'semantic',
      setEmbeddingModel: (embeddingModel) => set({ embeddingModel }),
      setChunkSize: (chunkSize) => set({ chunkSize }),
      setChunkOverlap: (chunkOverlap) => set({ chunkOverlap }),
      setTopK: (topK) => set({ topK }),
      setUseReranking: (useReranking) => set({ useReranking }),
      setRetrievalMode: (retrievalMode) => set({ retrievalMode }),

      // UI
      sidebarCollapsed: false,
      toggleSidebar: () => set((s) => ({ sidebarCollapsed: !s.sidebarCollapsed })),
    }),
    {
      name: 'rag-eval-settings',
      partialize: (state) => ({
        theme: state.theme,
        llmBackend: state.llmBackend,
        llmModel: state.llmModel,
        embeddingModel: state.embeddingModel,
        chunkSize: state.chunkSize,
        chunkOverlap: state.chunkOverlap,
        topK: state.topK,
        useReranking: state.useReranking,
        retrievalMode: state.retrievalMode,
      }),
    }
  )
);

// Apply theme on initial load
const savedTheme = (() => {
  try {
    const stored = localStorage.getItem('rag-eval-settings');
    if (stored) return JSON.parse(stored)?.state?.theme;
  } catch {}
  return 'dark';
})();
if (savedTheme === 'light') {
  document.documentElement.classList.add('light');
}
