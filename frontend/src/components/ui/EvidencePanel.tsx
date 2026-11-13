import React, { useState } from 'react';
import {
  FileText,
  ChevronDown,
  ChevronUp,
  Copy,
  Check,
  Layers,
  Sparkles,
} from 'lucide-react';
import type { RetrievedChunk } from '../../types';

export interface EvidencePanelProps {
  chunks: RetrievedChunk[];
  loading?: boolean;
}

export const EvidencePanel: React.FC<EvidencePanelProps> = ({
  chunks,
  loading = false,
}) => {
  const [expandedChunks, setExpandedChunks] = useState<Record<string, boolean>>({});
  const [copiedId, setCopiedId] = useState<string | null>(null);

  const toggleExpand = (id: string) => {
    setExpandedChunks((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  const copyToClipboard = (text: string, id: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  if (loading) {
    return (
      <div className="space-y-3">
        {[1, 2, 3].map((i) => (
          <div key={i} className="card p-4 animate-pulse space-y-3">
            <div className="flex items-center justify-between">
              <div className="w-32 h-4 skeleton" />
              <div className="w-20 h-4 skeleton" />
            </div>
            <div className="w-full h-12 skeleton" />
          </div>
        ))}
      </div>
    );
  }

  if (!chunks || chunks.length === 0) {
    return (
      <div
        className="p-8 text-center rounded-xl border border-dashed flex flex-col items-center justify-center gap-2"
        style={{
          borderColor: 'var(--color-border)',
          backgroundColor: 'var(--color-bg-tertiary)',
        }}
      >
        <Layers
          className="w-8 h-8 opacity-40"
          style={{ color: 'var(--color-text-secondary)' }}
        />
        <p className="text-sm font-medium" style={{ color: 'var(--color-text-secondary)' }}>
          No evidence chunks retrieved
        </p>
        <p className="text-xs" style={{ color: 'var(--color-text-tertiary)' }}>
          Ask a question in RAG mode to inspect vector store contexts.
        </p>
      </div>
    );
  }

  const getScoreColor = (score: number) => {
    if (score >= 0.7) return 'var(--color-emerald)';
    if (score >= 0.4) return 'var(--color-amber)';
    return 'var(--color-rose)';
  };

  const getScoreBadgeClass = (score: number) => {
    if (score >= 0.7) return 'badge-green';
    if (score >= 0.4) return 'badge-amber';
    return 'badge-rose';
  };

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between px-1">
        <span
          className="text-xs font-semibold uppercase tracking-wider flex items-center gap-1.5"
          style={{ color: 'var(--color-text-secondary)' }}
        >
          <Sparkles className="w-3.5 h-3.5 text-teal-400" />
          Retrieved Contexts ({chunks.length})
        </span>
        <button
          type="button"
          onClick={() => {
            const allExpanded = chunks.every(
              (c) => expandedChunks[c.chunk_id || `${c.document_id}-${c.chunk_index}`]
            );
            const newState: Record<string, boolean> = {};
            chunks.forEach((c) => {
              const key = c.chunk_id || `${c.document_id}-${c.chunk_index}`;
              newState[key] = !allExpanded;
            });
            setExpandedChunks(newState);
          }}
          className="text-xs hover:underline cursor-pointer"
          style={{ color: 'var(--color-accent-secondary)' }}
        >
          {chunks.every(
            (c) => expandedChunks[c.chunk_id || `${c.document_id}-${c.chunk_index}`]
          )
            ? 'Collapse All'
            : 'Expand All'}
        </button>
      </div>

      {chunks.map((chunk, index) => {
        const chunkKey = chunk.chunk_id || `${chunk.document_id}-${chunk.chunk_index}-${index}`;
        const isExpanded = !!expandedChunks[chunkKey];
        const score = typeof chunk.similarity_score === 'number' ? chunk.similarity_score : 0;
        const scorePct = Math.round(score * 100);

        return (
          <div
            key={chunkKey}
            className="rounded-xl border transition-all duration-150 overflow-hidden"
            style={{
              backgroundColor: 'var(--color-bg-secondary)',
              borderColor: 'var(--color-border)',
            }}
          >
            {/* ── Chunk Header ── */}
            <div
              className="flex items-center justify-between p-3 cursor-pointer hover:bg-white/[0.02]"
              onClick={() => toggleExpand(chunkKey)}
            >
              <div className="flex items-center gap-2 flex-wrap min-w-0">
                <span
                  className="badge badge-purple font-mono text-[11px] flex items-center gap-1"
                >
                  <FileText className="w-3 h-3" />
                  <span className="truncate max-w-[160px]">{chunk.filename || 'Document'}</span>
                </span>

                <span
                  className="badge text-[10px] font-mono"
                  style={{
                    backgroundColor: 'var(--color-bg-elevated)',
                    color: 'var(--color-text-secondary)',
                  }}
                >
                  Chunk #{chunk.chunk_index ?? index}
                </span>

                {chunk.char_count && (
                  <span
                    className="text-[10px] font-mono"
                    style={{ color: 'var(--color-text-tertiary)' }}
                  >
                    {chunk.char_count} chars
                  </span>
                )}
              </div>

              {/* Right Side: Similarity Score & Expand */}
              <div className="flex items-center gap-3">
                <div className="flex items-center gap-1.5" title={`Similarity: ${score.toFixed(4)}`}>
                  <div
                    className="w-16 h-1.5 rounded-full overflow-hidden hidden sm:block"
                    style={{ backgroundColor: 'var(--color-bg-tertiary)' }}
                  >
                    <div
                      className="h-full rounded-full transition-all"
                      style={{
                        width: `${Math.min(100, Math.max(0, scorePct))}%`,
                        backgroundColor: getScoreColor(score),
                      }}
                    />
                  </div>
                  <span
                    className={`badge ${getScoreBadgeClass(score)} font-mono font-semibold text-[11px]`}
                  >
                    {scorePct}% match
                  </span>
                </div>

                <div
                  className="p-1 rounded hover:bg-white/5 transition-colors"
                  style={{ color: 'var(--color-text-secondary)' }}
                >
                  {isExpanded ? (
                    <ChevronUp className="w-4 h-4" />
                  ) : (
                    <ChevronDown className="w-4 h-4" />
                  )}
                </div>
              </div>
            </div>

            {/* ── Chunk Content ── */}
            <div
              className="px-3 pb-3 border-t"
              style={{
                borderColor: 'var(--color-border-subtle)',
                backgroundColor: 'var(--color-bg-tertiary)',
              }}
            >
              <div className="relative pt-2.5">
                <p
                  className={`text-xs font-mono leading-relaxed select-text ${
                    !isExpanded ? 'line-clamp-3' : ''
                  }`}
                  style={{ color: 'var(--color-text-primary)' }}
                >
                  {chunk.content}
                </p>

                {/* Bottom toolbar inside chunk */}
                <div className="mt-2.5 pt-2 flex items-center justify-between border-t border-[var(--color-border-subtle)] text-[11px]">
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      toggleExpand(chunkKey);
                    }}
                    className="hover:underline cursor-pointer"
                    style={{ color: 'var(--color-accent-secondary)' }}
                  >
                    {isExpanded ? 'Show less' : 'Read full text'}
                  </button>

                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      copyToClipboard(chunk.content, chunkKey);
                    }}
                    className="btn btn-ghost btn-sm py-1 px-2 text-[11px] flex items-center gap-1"
                    title="Copy chunk text"
                  >
                    {copiedId === chunkKey ? (
                      <>
                        <Check className="w-3 h-3 text-emerald-400" />
                        <span className="text-emerald-400">Copied</span>
                      </>
                    ) : (
                      <>
                        <Copy className="w-3 h-3" />
                        <span>Copy</span>
                      </>
                    )}
                  </button>
                </div>
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
};

export default EvidencePanel;
