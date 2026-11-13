import React from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts';

export interface MetricBarChartProps {
  data: Array<{ name: string; value: number; fill?: string }>;
  title?: string;
  color?: string;
  height?: number;
  valueFormatter?: (val: number) => string;
}

export const MetricBarChart: React.FC<MetricBarChartProps> = ({
  data,
  title,
  color = 'var(--color-accent-primary)',
  height = 260,
  valueFormatter = (v: number) => (v <= 1 && v >= 0 ? `${(v * 100).toFixed(1)}%` : `${v}`),
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
        No chart data available
      </div>
    );
  }

  // Predefined palette if bar-specific fills aren't provided
  const palette = [
    '#7c6af7',
    '#2dd4bf',
    '#38bdf8',
    '#34d399',
    '#a78bfa',
    '#fbbf24',
    '#f87171',
  ];

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
          <BarChart
            data={data}
            margin={{ top: 10, right: 10, left: -15, bottom: 25 }}
          >
            <CartesianGrid
              strokeDasharray="3 3"
              stroke="#1e1e32"
              vertical={false}
            />
            <XAxis
              dataKey="name"
              stroke="#5c5c82"
              tick={{ fill: '#9898b8', fontSize: 11 }}
              tickLine={{ stroke: '#1e1e32' }}
              interval={0}
              angle={-20}
              textAnchor="end"
            />
            <YAxis
              stroke="#5c5c82"
              tick={{ fill: '#9898b8', fontSize: 11 }}
              tickLine={{ stroke: '#1e1e32' }}
              tickFormatter={(v: number) => (v <= 1 && v >= 0 ? `${Math.round(v * 100)}%` : `${v}`)}
            />
            <Tooltip
              content={({ active, payload }) => {
                if (active && payload && payload.length) {
                  const item = payload[0];
                  const rawVal = Number(item.value);
                  return (
                    <div
                      className="px-3 py-2 rounded-lg border shadow-xl text-xs backdrop-blur-md"
                      style={{
                        backgroundColor: 'rgba(18, 18, 30, 0.95)',
                        borderColor: 'var(--color-border-active)',
                        color: 'var(--color-text-primary)',
                      }}
                    >
                      <div className="font-semibold text-white/90 mb-1">
                        {item.payload.name}
                      </div>
                      <div className="flex items-center gap-1.5 font-mono">
                        <span
                          className="w-2 h-2 rounded-full"
                          style={{
                            backgroundColor:
                              item.payload.fill || item.color || color,
                          }}
                        />
                        <span style={{ color: 'var(--color-text-secondary)' }}>
                          Score:
                        </span>
                        <span className="font-bold text-white">
                          {valueFormatter(rawVal)}
                        </span>
                      </div>
                    </div>
                  );
                }
                return null;
              }}
            />
            <Bar
              dataKey="value"
              radius={[6, 6, 0, 0]}
              maxBarSize={45}
            >
              {data.map((entry, index) => (
                <Cell
                  key={`cell-${index}`}
                  fill={entry.fill || palette[index % palette.length]}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default MetricBarChart;
