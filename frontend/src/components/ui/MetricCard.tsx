import React from 'react';
import type { ReactNode } from 'react';
import { TrendingUp, TrendingDown, Minus } from 'lucide-react';

export interface MetricCardProps {
  label: string;
  value: number | string;
  unit?: string;
  delta?: number;
  color?: string; // hex or CSS variable
  icon?: ReactNode;
  loading?: boolean;
  tooltip?: string;
  sublabel?: string;
}

export const MetricCard: React.FC<MetricCardProps> = ({
  label,
  value,
  unit,
  delta,
  color = 'var(--color-accent-primary)',
  icon,
  loading = false,
  tooltip,
  sublabel,
}) => {
  if (loading) {
    return (
      <div className="metric-card animate-pulse">
        <div className="flex items-center justify-between">
          <div className="w-8 h-8 rounded-lg skeleton" />
          <div className="w-16 h-4 skeleton" />
        </div>
        <div className="mt-3 space-y-2">
          <div className="w-24 h-8 skeleton" />
          <div className="w-32 h-4 skeleton" />
        </div>
        <div className="mt-3 w-full h-1.5 skeleton rounded-full" />
      </div>
    );
  }

  // Format value
  const isNumeric = typeof value === 'number';
  const isRatio = isNumeric && value >= 0 && value <= 1 && unit !== 'ms' && unit !== 's';
  const displayValue = isNumeric
    ? isRatio
      ? `${(value * 100).toFixed(1)}%`
      : value.toLocaleString(undefined, { maximumFractionDigits: 2 })
    : value;

  // Determine score color if ratio
  const getScoreColor = (val: number) => {
    if (label.toLowerCase().includes('hallucination')) {
      // Lower is better for hallucination
      if (val <= 0.15) return 'var(--color-emerald)';
      if (val <= 0.35) return 'var(--color-amber)';
      return 'var(--color-rose)';
    }
    // Higher is better for faithfulness, relevance, etc.
    if (val >= 0.7) return 'var(--color-emerald)';
    if (val >= 0.4) return 'var(--color-amber)';
    return 'var(--color-rose)';
  };

  return (
    <div
      className="metric-card relative group hover:shadow-lg transition-all duration-200"
      title={tooltip}
    >
      {/* Top Header: Icon & Delta */}
      <div className="flex items-center justify-between w-full">
        {icon && (
          <div
            className="flex items-center justify-center w-8 h-8 rounded-lg transition-transform group-hover:scale-105"
            style={{
              backgroundColor: `color-mix(in srgb, ${color} 12%, transparent)`,
              color: color,
              border: `1px solid color-mix(in srgb, ${color} 25%, transparent)`,
            }}
          >
            {icon}
          </div>
        )}

        {typeof delta === 'number' && (
          <div
            className={`flex items-center gap-1 text-xs font-semibold px-2 py-0.5 rounded-full ${
              delta > 0
                ? 'text-emerald-400 bg-emerald-500/10 border border-emerald-500/20'
                : delta < 0
                ? 'text-rose-400 bg-rose-500/10 border border-rose-500/20'
                : 'text-gray-400 bg-gray-500/10'
            }`}
          >
            {delta > 0 ? (
              <TrendingUp className="w-3 h-3" />
            ) : delta < 0 ? (
              <TrendingDown className="w-3 h-3" />
            ) : (
              <Minus className="w-3 h-3" />
            )}
            <span>
              {delta > 0 ? `+${delta.toFixed(1)}%` : `${delta.toFixed(1)}%`}
            </span>
          </div>
        )}
      </div>

      {/* Main Metric Value */}
      <div className="mt-2 flex items-baseline gap-1.5">
        <span
          className="metric-value"
          style={{
            color: isRatio ? getScoreColor(value as number) : 'var(--color-text-primary)',
          }}
        >
          {displayValue}
        </span>
        {unit && (
          <span
            className="text-sm font-semibold"
            style={{ color: 'var(--color-text-tertiary)' }}
          >
            {unit}
          </span>
        )}
      </div>

      {/* Label and Sublabel */}
      <div className="flex flex-col">
        <span className="metric-label">{label}</span>
        {sublabel && (
          <span
            className="text-[11px] mt-0.5"
            style={{ color: 'var(--color-text-muted)' }}
          >
            {sublabel}
          </span>
        )}
      </div>

      {/* Ratio / Progress bar for 0-1 values */}
      {isRatio && (
        <div className="mt-2 w-full">
          <div
            className="w-full h-1.5 rounded-full overflow-hidden"
            style={{ backgroundColor: 'var(--color-bg-tertiary)' }}
          >
            <div
              className="h-full rounded-full transition-all duration-500 ease-out"
              style={{
                width: `${Math.min(100, Math.max(0, (value as number) * 100))}%`,
                backgroundColor: getScoreColor(value as number),
              }}
            />
          </div>
        </div>
      )}
    </div>
  );
};

export default MetricCard;
