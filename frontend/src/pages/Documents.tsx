import React, { useState, useEffect, useRef } from 'react';
import {
  UploadCloud,
  FileText,
  Database,
  Layers,
  Sparkles,
  AlertCircle,
  CheckCircle2,
  X,
  RefreshCw,
  Search,
  BookOpen,
  Sliders,
} from 'lucide-react';
import { documentsAPI, retrievalAPI, getErrorMessage } from '../api/client';
import { useSettingsStore } from '../store/settingsStore';
import type { DocumentInfo, DocumentChunk, VectorStoreStats } from '../types';
import { DocumentCard } from '../components/ui/DocumentCard';
import { LoadingState } from '../components/ui/LoadingState';

export const Documents: React.FC = () => {
  const { chunkSize, chunkOverlap, embeddingModel, setChunkSize, setChunkOverlap } = useSettingsStore();

  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [stats, setStats] = useState<VectorStoreStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [successMsg, setSuccessMsg] = useState<string | null>(null);

  // Upload Form State
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [description, setDescription] = useState('');
  const [isUploading, setIsUploading] = useState(false);
  const [isDragOver, setIsDragOver] = useState(false);
  const [showConfig, setShowConfig] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Search / Filter
  const [searchQuery, setSearchQuery] = useState('');

  // Chunks Modal State
  const [selectedDocForChunks, setSelectedDocForChunks] = useState<DocumentInfo | null>(null);
  const [chunks, setChunks] = useState<DocumentChunk[]>([]);
  const [loadingChunks, setLoadingChunks] = useState(false);

  const fetchDocsAndStats = async () => {
    setLoading(true);
    setError(null);
    try {
      const [docsData, statsData] = await Promise.allSettled([
        documentsAPI.list(),
        retrievalAPI.getStats(),
      ]);

      if (docsData.status === 'fulfilled') {
        setDocuments(docsData.value || []);
      }
      if (statsData.status === 'fulfilled') {
        setStats(statsData.value);
      }
    } catch (err) {
      setError(getErrorMessage(err));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDocsAndStats();
  }, []);

  const handleFileSelect = (file: File) => {
    setError(null);
    setSuccessMsg(null);
    const validExtensions = ['.pdf', '.txt', '.md'];
    const hasValidExt = validExtensions.some((ext) =>
      file.name.toLowerCase().endsWith(ext)
    );

    if (!hasValidExt) {
      setError('Unsupported file type. Please upload a PDF, TXT, or MD document.');
      return;
    }

    if (file.size > 25 * 1024 * 1024) {
      setError('File is too large. Maximum supported file size is 25MB.');
      return;
    }

    setSelectedFile(file);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  };

  const handleUpload = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedFile) return;

    setIsUploading(true);
    setError(null);
    setSuccessMsg(null);

    try {
      const response = await documentsAPI.upload(selectedFile, description);
      setSuccessMsg(
        `Successfully ingested "${selectedFile.name}" into ${response.num_chunks} chunks!`
      );
      setSelectedFile(null);
      setDescription('');
      if (fileInputRef.current) fileInputRef.current.value = '';
      await fetchDocsAndStats();
    } catch (err) {
      setError(getErrorMessage(err));
    } finally {
      setIsUploading(false);
    }
  };

  const handleDelete = async (id: string) => {
    try {
      await documentsAPI.delete(id);
      setDocuments((prev) => prev.filter((d) => d.id !== id));
      // Refresh stats
      try {
        const newStats = await retrievalAPI.getStats();
        setStats(newStats);
      } catch {}
    } catch (err) {
      setError(getErrorMessage(err));
    }
  };

  const handleViewChunks = async (doc: DocumentInfo) => {
    setSelectedDocForChunks(doc);
    setLoadingChunks(true);
    try {
      const chunksData = await documentsAPI.getChunks(doc.id, 1, 50);
      setChunks(chunksData || []);
    } catch (err) {
      setError(`Failed to load chunks: ${getErrorMessage(err)}`);
    } finally {
      setLoadingChunks(false);
    }
  };

  const filteredDocs = documents.filter((d) =>
    d.filename.toLowerCase().includes(searchQuery.toLowerCase()) ||
    (d.description && d.description.toLowerCase().includes(searchQuery.toLowerCase()))
  );

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
              Document Knowledge Base
            </h1>
            <span className="badge badge-teal text-xs">Vector Store</span>
          </div>
          <p
            className="text-xs sm:text-sm mt-1"
            style={{ color: 'var(--color-text-secondary)' }}
          >
            Upload reference literature, view semantic chunking parameters, and manage vector indices.
          </p>
        </div>

        <button
          type="button"
          onClick={fetchDocsAndStats}
          disabled={loading}
          className="btn btn-secondary btn-sm flex items-center gap-1.5 self-start sm:self-auto"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
          <span>Refresh Store</span>
        </button>
      </div>

      {/* ── VectorStore Stats Bar ── */}
      <div
        className="card p-4 grid grid-cols-2 md:grid-cols-4 gap-4"
        style={{
          backgroundColor: 'var(--color-bg-secondary)',
          borderColor: 'var(--color-border)',
        }}
      >
        <div className="flex items-center gap-3">
          <div className="p-2.5 rounded-xl bg-purple-500/10 text-purple-400 border border-purple-500/20">
            <BookOpen className="w-5 h-5" />
          </div>
          <div>
            <span className="text-[11px] font-medium" style={{ color: 'var(--color-text-tertiary)' }}>
              Total Documents
            </span>
            <div className="text-xl font-bold font-mono" style={{ color: 'var(--color-text-primary)' }}>
              {stats?.total_documents ?? documents.length}
            </div>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <div className="p-2.5 rounded-xl bg-teal-500/10 text-teal-400 border border-teal-500/20">
            <Layers className="w-5 h-5" />
          </div>
          <div>
            <span className="text-[11px] font-medium" style={{ color: 'var(--color-text-tertiary)' }}>
              Embedded Chunks
            </span>
            <div className="text-xl font-bold font-mono text-teal-400">
              {stats?.total_chunks ?? 0}
            </div>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <div className="p-2.5 rounded-xl bg-sky-500/10 text-sky-400 border border-sky-500/20">
            <Database className="w-5 h-5" />
          </div>
          <div>
            <span className="text-[11px] font-medium" style={{ color: 'var(--color-text-tertiary)' }}>
              Embedding Model
            </span>
            <div
              className="text-xs font-mono font-semibold truncate max-w-[150px]"
              title={stats?.embedding_model || embeddingModel}
              style={{ color: 'var(--color-text-primary)' }}
            >
              {(stats?.embedding_model || embeddingModel).split('/').pop()}
            </div>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <div className="p-2.5 rounded-xl bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
            <Sparkles className="w-5 h-5" />
          </div>
          <div>
            <span className="text-[11px] font-medium" style={{ color: 'var(--color-text-tertiary)' }}>
              Vector Index
            </span>
            <div className="text-xs font-mono font-semibold text-emerald-400">
              {stats?.index_type || 'FAISS (Cosine)'}
            </div>
          </div>
        </div>
      </div>

      {/* ── Alerts ── */}
      {error && (
        <div
          className="p-4 rounded-xl border flex items-center justify-between gap-3 text-xs"
          style={{
            backgroundColor: 'rgba(248, 113, 113, 0.1)',
            borderColor: 'rgba(248, 113, 113, 0.25)',
            color: 'var(--color-rose)',
          }}
        >
          <div className="flex items-center gap-2">
            <AlertCircle className="w-4 h-4 flex-shrink-0" />
            <span>{error}</span>
          </div>
          <button
            type="button"
            onClick={() => setError(null)}
            className="hover:text-white"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      )}

      {successMsg && (
        <div
          className="p-4 rounded-xl border flex items-center justify-between gap-3 text-xs"
          style={{
            backgroundColor: 'rgba(52, 211, 153, 0.1)',
            borderColor: 'rgba(52, 211, 153, 0.25)',
            color: 'var(--color-emerald)',
          }}
        >
          <div className="flex items-center gap-2">
            <CheckCircle2 className="w-4 h-4 flex-shrink-0" />
            <span>{successMsg}</span>
          </div>
          <button
            type="button"
            onClick={() => setSuccessMsg(null)}
            className="hover:text-white"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      )}

      {/* ── Drag & Drop Upload Section ── */}
      <div className="card p-6 space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <UploadCloud className="w-5 h-5 text-[var(--color-accent-primary)]" />
            <h2
              className="text-base font-semibold"
              style={{ color: 'var(--color-text-primary)' }}
            >
              Ingest Document
            </h2>
          </div>

          <button
            type="button"
            onClick={() => setShowConfig(!showConfig)}
            className="btn btn-ghost btn-sm text-xs flex items-center gap-1.5"
          >
            <Sliders className="w-3.5 h-3.5" />
            <span>{showConfig ? 'Hide Chunk Config' : 'Chunk Settings'}</span>
          </button>
        </div>

        {/* Optional Chunking Controls */}
        {showConfig && (
          <div
            className="p-4 rounded-xl border grid grid-cols-1 sm:grid-cols-2 gap-4 text-xs"
            style={{
              backgroundColor: 'var(--color-bg-tertiary)',
              borderColor: 'var(--color-border)',
            }}
          >
            <div>
              <div className="flex justify-between mb-1.5 font-medium">
                <label className="label">Chunk Size (chars / tokens)</label>
                <span className="font-mono text-purple-400">{chunkSize}</span>
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
                Higher values preserve broader semantic context.
              </span>
            </div>

            <div>
              <div className="flex justify-between mb-1.5 font-medium">
                <label className="label">Chunk Overlap (chars)</label>
                <span className="font-mono text-teal-400">{chunkOverlap}</span>
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
                Prevents boundary loss between sequential chunks.
              </span>
            </div>
          </div>
        )}

        {/* Dropzone Container */}
        <form onSubmit={handleUpload} className="space-y-4">
          <div
            className={`dropzone ${isDragOver ? 'drag-over' : ''} transition-all`}
            onDragOver={(e) => {
              e.preventDefault();
              setIsDragOver(true);
            }}
            onDragLeave={() => setIsDragOver(false)}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
          >
            <input
              type="file"
              ref={fileInputRef}
              accept=".pdf,.txt,.md"
              onChange={(e) => {
                if (e.target.files && e.target.files.length > 0) {
                  handleFileSelect(e.target.files[0]);
                }
              }}
              className="hidden"
            />

            <div className="flex flex-col items-center justify-center gap-2">
              <div className="p-3 rounded-full bg-[var(--color-accent-glow)] text-[var(--color-accent-secondary)]">
                <UploadCloud className="w-8 h-8" />
              </div>
              <p className="text-sm font-semibold text-white">
                {selectedFile
                  ? `Selected: ${selectedFile.name}`
                  : 'Click or drag document here to upload'}
              </p>
              <p className="text-xs text-[var(--color-text-secondary)]">
                Supports PDF, TXT, and Markdown files up to 25MB
              </p>
            </div>
          </div>

          {/* Selected File Details & Metadata */}
          {selectedFile && (
            <div
              className="p-3.5 rounded-xl border flex flex-col sm:flex-row items-center justify-between gap-3 text-xs"
              style={{
                backgroundColor: 'var(--color-bg-secondary)',
                borderColor: 'var(--color-border)',
              }}
            >
              <div className="flex items-center gap-2 min-w-0">
                <FileText className="w-4 h-4 text-purple-400 flex-shrink-0" />
                <span className="font-semibold truncate max-w-[240px]">
                  {selectedFile.name}
                </span>
                <span className="font-mono text-[var(--color-text-tertiary)]">
                  ({(selectedFile.size / 1024).toFixed(1)} KB)
                </span>
              </div>

              <div className="flex items-center gap-2 w-full sm:w-auto">
                <input
                  type="text"
                  placeholder="Optional description (e.g., Q3 Report)"
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                  className="input py-1 text-xs sm:w-64"
                />

                <button
                  type="submit"
                  disabled={isUploading}
                  className="btn btn-primary btn-sm flex items-center gap-1.5"
                >
                  {isUploading ? (
                    <>
                      <div className="spinner !w-3.5 !h-3.5" />
                      <span>Chunking...</span>
                    </>
                  ) : (
                    <>
                      <UploadCloud className="w-3.5 h-3.5" />
                      <span>Ingest</span>
                    </>
                  )}
                </button>

                <button
                  type="button"
                  onClick={() => setSelectedFile(null)}
                  disabled={isUploading}
                  className="btn btn-ghost btn-sm p-1.5"
                  title="Cancel selection"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
            </div>
          )}
        </form>
      </div>

      {/* ── Document List Header & Search ── */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pt-2">
        <div className="flex items-center gap-2">
          <h2
            className="text-lg font-bold"
            style={{ color: 'var(--color-text-primary)' }}
          >
            Indexed Documents ({documents.length})
          </h2>
        </div>

        <div className="relative w-full sm:w-72">
          <Search className="w-3.5 h-3.5 absolute left-3 top-1/2 -translate-y-1/2 text-[var(--color-text-muted)]" />
          <input
            type="text"
            placeholder="Search documents by name or notes..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="input pl-8 py-1.5 text-xs"
          />
        </div>
      </div>

      {/* ── Document Cards Grid ── */}
      {loading ? (
        <LoadingState message="Loading documents from vector store..." />
      ) : filteredDocs.length === 0 ? (
        <div
          className="card p-12 text-center rounded-2xl border border-dashed flex flex-col items-center justify-center gap-3"
          style={{
            borderColor: 'var(--color-border)',
            backgroundColor: 'var(--color-bg-secondary)',
          }}
        >
          <BookOpen className="w-12 h-12 opacity-30 text-[var(--color-text-secondary)]" />
          <div className="space-y-1">
            <h3 className="text-sm font-semibold text-white">
              {searchQuery ? 'No documents match your query' : 'No documents in knowledge base'}
            </h3>
            <p className="text-xs text-[var(--color-text-secondary)] max-w-sm">
              {searchQuery
                ? 'Try searching with a different term or clear the filter.'
                : 'Upload your reference material or research papers above to start building the evaluation vector corpus.'}
            </p>
          </div>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {filteredDocs.map((doc) => (
            <DocumentCard
              key={doc.id}
              doc={doc}
              onDelete={handleDelete}
              onViewChunks={handleViewChunks}
            />
          ))}
        </div>
      )}

      {/* ── Chunks Modal ── */}
      {selectedDocForChunks && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm animate-fade-in">
          <div
            className="card max-w-3xl w-full max-h-[85vh] flex flex-col p-6 space-y-4 shadow-2xl border"
            style={{
              backgroundColor: 'var(--color-bg-card)',
              borderColor: 'var(--color-border-active)',
            }}
          >
            {/* Modal Header */}
            <div className="flex items-center justify-between pb-3 border-b border-[var(--color-border)]">
              <div className="flex items-center gap-2 min-w-0">
                <FileText className="w-5 h-5 text-purple-400 flex-shrink-0" />
                <div className="min-w-0">
                  <h3 className="text-sm font-bold truncate text-white">
                    {selectedDocForChunks.filename}
                  </h3>
                  <p className="text-[11px] text-[var(--color-text-tertiary)]">
                    Inspection of generated chunks and embeddings
                  </p>
                </div>
              </div>
              <button
                type="button"
                onClick={() => setSelectedDocForChunks(null)}
                className="btn btn-ghost btn-sm p-1.5"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            {/* Modal Body: Chunks list */}
            <div className="flex-1 overflow-y-auto space-y-3 pr-1">
              {loadingChunks ? (
                <LoadingState message="Loading chunk vectors..." size="sm" />
              ) : chunks.length === 0 ? (
                <p className="text-center text-xs py-8 text-[var(--color-text-tertiary)]">
                  No chunks found for this document.
                </p>
              ) : (
                chunks.map((chunk) => (
                  <div
                    key={chunk.id || chunk.chunk_index}
                    className="p-3.5 rounded-xl border text-xs font-mono space-y-2"
                    style={{
                      backgroundColor: 'var(--color-bg-secondary)',
                      borderColor: 'var(--color-border-subtle)',
                    }}
                  >
                    <div className="flex items-center justify-between text-[11px]">
                      <span className="badge badge-purple text-[10px]">
                        Chunk #{chunk.chunk_index}
                      </span>
                      <span className="text-[var(--color-text-tertiary)]">
                        ~{chunk.token_estimate || Math.round(chunk.char_count / 4)} tokens ({chunk.char_count} chars)
                      </span>
                    </div>
                    <p
                      className="text-xs leading-relaxed text-[var(--color-text-primary)] select-text"
                    >
                      {chunk.content}
                    </p>
                  </div>
                ))
              )}
            </div>

            {/* Modal Footer */}
            <div className="pt-3 border-t border-[var(--color-border)] flex justify-end">
              <button
                type="button"
                onClick={() => setSelectedDocForChunks(null)}
                className="btn btn-secondary btn-sm"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Documents;
