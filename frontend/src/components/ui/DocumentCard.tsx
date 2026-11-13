import React, { useState } from 'react';
import {
  FileText,
  FileCode,
  Trash2,
  Calendar,
  Layers,
  HardDrive,
  Eye,
  AlertCircle,
  File,
} from 'lucide-react';
import type { DocumentInfo } from '../../types';

export interface DocumentCardProps {
  doc: DocumentInfo;
  onDelete: (id: string) => void;
  onViewChunks?: (doc: DocumentInfo) => void;
}

export const DocumentCard: React.FC<DocumentCardProps> = ({
  doc,
  onDelete,
  onViewChunks,
}) => {
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);

  // Format file size
  const formatBytes = (bytes: number): string => {
    if (!bytes || bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${parseFloat((bytes / Math.pow(k, i)).toFixed(1))} ${sizes[i]}`;
  };

  // Format upload time
  const formatDate = (dateStr: string): string => {
    try {
      const date = new Date(dateStr);
      return date.toLocaleDateString(undefined, {
        month: 'short',
        day: 'numeric',
        year: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
      });
    } catch {
      return dateStr;
    }
  };

  const getFileIcon = (type: string, name: string) => {
    const ext = name.split('.').pop()?.toLowerCase();
    if (ext === 'pdf' || type.includes('pdf')) {
      return <FileText className="w-5 h-5 text-rose-400" />;
    }
    if (ext === 'md' || ext === 'markdown') {
      return <FileCode className="w-5 h-5 text-teal-400" />;
    }
    if (ext === 'txt' || type.includes('text')) {
      return <FileText className="w-5 h-5 text-sky-400" />;
    }
    return <File className="w-5 h-5 text-purple-400" />;
  };

  const handleDelete = async () => {
    if (!confirmDelete) {
      setConfirmDelete(true);
      // Auto-reset confirmation after 4 seconds
      setTimeout(() => setConfirmDelete(false), 4000);
      return;
    }
    setIsDeleting(true);
    try {
      await onDelete(doc.id);
    } finally {
      setIsDeleting(false);
      setConfirmDelete(false);
    }
  };

  return (
    <div
      className="card relative flex flex-col justify-between group transition-all duration-200 hover:border-[var(--color-border-active)]"
      style={{
        backgroundColor: 'var(--color-bg-card)',
        borderColor: 'var(--color-border)',
      }}
    >
      <div>
        {/* Top Header: File Icon, Name, and Badges */}
        <div className="flex items-start gap-3">
          <div
            className="flex items-center justify-center w-10 h-10 rounded-xl flex-shrink-0"
            style={{
              backgroundColor: 'var(--color-bg-tertiary)',
              border: '1px solid var(--color-border)',
            }}
          >
            {getFileIcon(doc.file_type, doc.filename)}
          </div>

          <div className="flex-1 min-w-0">
            <h3
              className="text-sm font-semibold truncate hover:text-white transition-colors"
              title={doc.filename}
              style={{ color: 'var(--color-text-primary)' }}
            >
              {doc.filename}
            </h3>

            <div className="flex items-center gap-2 mt-1 flex-wrap">
              <span className="badge badge-purple text-[10px] font-mono uppercase">
                {doc.filename.split('.').pop() || doc.file_type || 'file'}
              </span>

              <span
                className="badge text-[10px] font-mono flex items-center gap-1"
                style={{
                  backgroundColor: 'var(--color-bg-tertiary)',
                  color: 'var(--color-text-secondary)',
                }}
              >
                <Layers className="w-3 h-3 text-teal-400" />
                {doc.num_chunks} chunks
              </span>

              <span
                className="badge text-[10px] font-mono flex items-center gap-1"
                style={{
                  backgroundColor: 'var(--color-bg-tertiary)',
                  color: 'var(--color-text-secondary)',
                }}
              >
                <HardDrive className="w-3 h-3" />
                {formatBytes(doc.file_size)}
              </span>
            </div>
          </div>
        </div>

        {/* Description if present */}
        {doc.description && (
          <p
            className="mt-3 text-xs line-clamp-2 leading-relaxed"
            style={{ color: 'var(--color-text-secondary)' }}
          >
            {doc.description}
          </p>
        )}
      </div>

      {/* Footer: Date & Actions */}
      <div
        className="mt-4 pt-3 flex items-center justify-between border-t"
        style={{ borderColor: 'var(--color-border-subtle)' }}
      >
        <div
          className="flex items-center gap-1.5 text-[11px]"
          style={{ color: 'var(--color-text-muted)' }}
        >
          <Calendar className="w-3.5 h-3.5" />
          <span>{formatDate(doc.upload_time)}</span>
        </div>

        <div className="flex items-center gap-1.5">
          {onViewChunks && (
            <button
              type="button"
              onClick={() => onViewChunks(doc)}
              className="btn btn-ghost btn-sm py-1 px-2 text-xs flex items-center gap-1"
              title="View document chunks"
            >
              <Eye className="w-3.5 h-3.5 text-teal-400" />
              <span>Chunks</span>
            </button>
          )}

          <button
            type="button"
            onClick={handleDelete}
            disabled={isDeleting}
            className={`btn btn-sm py-1 px-2 text-xs flex items-center gap-1 transition-all ${
              confirmDelete
                ? 'btn-danger font-semibold animate-pulse'
                : 'btn-ghost text-rose-400 hover:bg-rose-500/10'
            }`}
            title={confirmDelete ? 'Click to permanently delete' : 'Delete document'}
          >
            {confirmDelete ? (
              <>
                <AlertCircle className="w-3.5 h-3.5" />
                <span>Confirm?</span>
              </>
            ) : (
              <>
                <Trash2 className="w-3.5 h-3.5" />
                <span>Delete</span>
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
};

export default DocumentCard;
