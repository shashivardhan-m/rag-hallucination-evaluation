import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';

export interface ExperimentLineChartProps {
  data: Array<Record<string, any>>;
  lines: Array<{ key: string; color: string; label: string; strokeWidth?: number }>;
  xKey: string;
  title?: string;
  height?: number;
  yDomain?: [number | string, number | string];
  formatY?: (val: number) => string;
}

export const ExperimentLineChart: React.FC<ExperimentLineChartProps> = ({
  data,
  lines,
  xKey,
  title,
  height = 280,
  yDomain,
  formatY,
}) => {
  if (!data || data.length === 0) {
    return (
      <div
        className="flex items-center justify-center rounded-xl border border-dashed text-xs"
        style={{
          height: `${height}px`,
          borderColor: 'var(--color-border)',
          color: 'var(--color-text-tertiary)',
        }}
      >
        No trend data recorded yet
      </div>
    );
  }

  return (
    <div className="w-full">
      {title && (
        <h4
          className="text-xs font-semibold uppercase tracking-wider mb-3"
          style={{ color: 'var(--color-text-secondary)' }}
        >
          {title}
        </h4>
      )}

      <div style={{ width: '100%', height: `${height}px` }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart
            data={data}
            margin={{ top: 10, right: 15, left: -10, bottom: 10 }}
          >
            <CartesianGrid
              strokeDasharray="3 3"
              stroke="#1e1e32"
              vertical={false}
            />
            <XAxis
              dataKey={xKey}
              stroke="#5c5c82"
              tick={{ fill: '#9898b8', fontSize: 11 }}
              tickLine={{ stroke: '#1e1e32' }}
              tickFormatter={(val: string | number) => {
                if (typeof val === 'string' && val.includes('T')) {
                  try {
                    const d = new Date(val);
                    return `${d.getMonth() + 1}/${d.getDate()} ${d.getHours()}:${String(d.getMinutes()).padStart(2, '0')}`;
                  } catch {
                    return val;
                  }
                }
                return String(val);
              }}
            />
            <YAxis
              stroke="#5c5c82"
              tick={{ fill: '#9898b8', fontSize: 11 }}
              tickLine={{ stroke: '#1e1e32' }}
              domain={yDomain || [0, 'auto']}
              tickFormatter={(v: number) => {
                if (formatY) return formatY(v);
                if (v <= 1 && v >= 0) return `${Math.round(v * 100)}%`;
                return `${v}`;
              }}
            />
            <Tooltip
              content={({ active, payload, label }) => {
                if (active && payload && payload.length) {
                  return (
                    <div
                      className="p-3 rounded-lg border shadow-xl text-xs backdrop-blur-md space-y-1.5"
                      style={{
                        backgroundColor: 'rgba(18, 18, 30, 0.95)',
                        borderColor: 'var(--color-border-active)',
                        color: 'var(--color-text-primary)',
                      }}
                    >
                      <div className="font-semibold text-white/90 border-b border-gray-800 pb-1">
                        {xKey}: {String(label)}
                      </div>
                      {payload.map((entry, index) => {
                        const val = Number(entry.value);
                        const isRatio = val >= 0 && val <= 1;
                        return (
                          <div
                            key={index}
                            className="flex items-center justify-between gap-4 font-mono"
                          >
                            <span className="flex items-center gap-1.5">
                              <span
                                className="w-2 h-2 rounded-full"
                                style={{ backgroundColor: entry.color }}
                              />
                              <span style={{ color: 'var(--color-text-secondary)' }}>
                                {entry.name}:
                              </span>
                            </span>
                            <span className="font-bold text-white">
                              {formatY ? formatY(val) : isRatio ? `${(val * 100).toFixed(1)}%` : val}
                            </span>
                          </div>
                        );
                      })}
                    </div>
                  );
                }
                return null;
              }}
            />
            <Legend
              wrapperStyle={{
                paddingTop: '10px',
                fontSize: '11px',
                color: '#9898b8',
              }}
            />
            {lines.map((line) => (
              <Line
                key={line.key}
                type="monotone"
                dataKey={line.key}
                name={line.label}
                stroke={line.color}
                strokeWidth={line.strokeWidth || 2}
                dot={{ r: 3, fill: line.color, strokeWidth: 1, stroke: '#0a0a0f' }}
                activeDot={{ r: 5, strokeWidth: 2, stroke: '#ffffff' }}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default ExperimentLineChart;
