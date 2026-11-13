import React from 'react';
import { Loader2 } from 'lucide-react';

export interface LoadingStateProps {
  message?: string;
  submessage?: string;
  fullPage?: boolean;
  size?: 'sm' | 'md' | 'lg';
}

export const LoadingState: React.FC<LoadingStateProps> = ({
  message = 'Loading...',
  submessage,
  fullPage = false,
  size = 'md',
}) => {
  const sizeMap = {
    sm: { icon: 'w-5 h-5', text: 'text-xs' },
    md: { icon: 'w-8 h-8', text: 'text-sm' },
    lg: { icon: 'w-12 h-12', text: 'text-base' },
  }[size];

  const content = (
    <div className="flex flex-col items-center justify-center gap-3 p-6 text-center fade-in">
      <div className="relative">
        <Loader2
          className={`${sizeMap.icon} animate-spin text-[var(--color-accent-primary)]`}
        />
        <div
          className="absolute inset-0 rounded-full blur-md opacity-30"
          style={{ backgroundColor: 'var(--color-accent-primary)' }}
        />
      </div>

      <div className="space-y-1">
        <p
          className={`${sizeMap.text} font-medium`}
          style={{ color: 'var(--color-text-primary)' }}
        >
          {message}
        </p>
        {submessage && (
          <p
            className="text-xs"
            style={{ color: 'var(--color-text-tertiary)' }}
          >
            {submessage}
          </p>
        )}
      </div>
    </div>
  );

  if (fullPage) {
    return (
      <div className="fixed inset-0 z-50 flex items-center justify-center bg-[var(--color-bg-primary)]/80 backdrop-blur-sm">
        <div
          className="p-8 rounded-2xl border shadow-2xl"
          style={{
            backgroundColor: 'var(--color-bg-card)',
            borderColor: 'var(--color-border)',
          }}
        >
          {content}
        </div>
      </div>
    );
  }

  return <div className="w-full py-8 flex items-center justify-center">{content}</div>;
};

// ── Reusable Skeletons ──
export const SkeletonCard: React.FC<{ rows?: number }> = ({ rows = 3 }) => (
  <div className="card space-y-4 animate-pulse">
    <div className="flex items-center justify-between">
      <div className="w-1/3 h-5 skeleton" />
      <div className="w-16 h-4 skeleton" />
    </div>
    <div className="space-y-2">
      {Array.from({ length: rows }).map((_, i) => (
        <div
          key={i}
          className="h-3 skeleton rounded"
          style={{ width: `${85 - i * 15}%` }}
        />
      ))}
    </div>
  </div>
);

export const SkeletonTable: React.FC<{ rows?: number; cols?: number }> = ({
  rows = 5,
  cols = 4,
}) => (
  <div
    className="rounded-xl border overflow-hidden"
    style={{
      backgroundColor: 'var(--color-bg-card)',
      borderColor: 'var(--color-border)',
    }}
  >
    <div
      className="p-3 border-b flex gap-4"
      style={{ borderColor: 'var(--color-border)' }}
    >
      {Array.from({ length: cols }).map((_, i) => (
        <div key={i} className="flex-1 h-4 skeleton rounded" />
      ))}
    </div>
    <div className="p-3 space-y-3">
      {Array.from({ length: rows }).map((_, r) => (
        <div key={r} className="flex gap-4">
          {Array.from({ length: cols }).map((_, c) => (
            <div key={c} className="flex-1 h-3.5 skeleton rounded" />
          ))}
        </div>
      ))}
    </div>
  </div>
);

export default LoadingState;
