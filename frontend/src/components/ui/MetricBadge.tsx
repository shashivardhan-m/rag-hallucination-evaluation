import React from 'react';
import { CheckCircle2, AlertTriangle, XCircle, Bookmark, Sparkles } from 'lucide-react';

export interface MetricBadgeProps {
  name: string;
  value: number;
  isReferenceBased?: boolean;
  className?: string;
}

export const MetricBadge: React.FC<MetricBadgeProps> = ({
  name,
  value,
  isReferenceBased = false,
  className = '',
}) => {
  const isHallucination = name.toLowerCase().includes('hallucination');

  // Determine threshold status
  let status: 'good' | 'mid' | 'bad' = 'mid';
  if (isHallucination) {
    if (value <= 0.15) status = 'good';
    else if (value <= 0.35) status = 'mid';
    else status = 'bad';
  } else {
    if (value >= 0.7) status = 'good';
    else if (value >= 0.4) status = 'mid';
    else status = 'bad';
  }

  const badgeStyles = {
    good: {
      bg: 'rgba(52, 211, 153, 0.12)',
      border: 'rgba(52, 211, 153, 0.25)',
      text: 'var(--color-emerald)',
      icon: CheckCircle2,
    },
    mid: {
      bg: 'rgba(251, 191, 36, 0.12)',
      border: 'rgba(251, 191, 36, 0.25)',
      text: 'var(--color-amber)',
      icon: AlertTriangle,
    },
    bad: {
      bg: 'rgba(248, 113, 113, 0.12)',
      border: 'rgba(248, 113, 113, 0.25)',
      text: 'var(--color-rose)',
      icon: XCircle,
    },
  }[status];

  const Icon = badgeStyles.icon;
  const formattedVal = (value * 100).toFixed(1);

  return (
    <div
      className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-lg border text-xs font-mono transition-all ${className}`}
      style={{
        backgroundColor: badgeStyles.bg,
        borderColor: badgeStyles.border,
        color: badgeStyles.text,
      }}
      title={`${name}: ${value.toFixed(4)} (${isReferenceBased ? 'Reference-based' : 'Reference-free'})`}
    >
      <Icon className="w-3.5 h-3.5 flex-shrink-0" />
      <span className="font-semibold text-white/90 truncate max-w-[150px]">
        {name}
      </span>
      <span className="font-bold text-[13px]">{formattedVal}%</span>

      <span
        className="text-[9px] px-1 py-0.2 rounded font-sans uppercase font-medium opacity-75"
        style={{
          backgroundColor: 'rgba(0, 0, 0, 0.25)',
          color: 'var(--color-text-secondary)',
        }}
      >
        {isReferenceBased ? (
          <span className="flex items-center gap-0.5">
            <Bookmark className="w-2.5 h-2.5" /> Ref
          </span>
        ) : (
          <span className="flex items-center gap-0.5">
            <Sparkles className="w-2.5 h-2.5" /> Free
          </span>
        )}
      </span>
    </div>
  );
};

export default MetricBadge;
