import React from 'react';
import {
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  Legend,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';

export interface ComparisonRadarProps {
  data: Array<{ metric: string; standard: number; rag: number }>;
  title?: string;
  height?: number;
}

export const ComparisonRadar: React.FC<ComparisonRadarProps> = ({
  data,
  title,
  height = 300,
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
        No comparison data available
      </div>
    );
  }

  return (
    <div className="w-full">
      {title && (
        <h4
          className="text-xs font-semibold uppercase tracking-wider mb-2"
          style={{ color: 'var(--color-text-secondary)' }}
        >
          {title}
        </h4>
      )}

      <div style={{ width: '100%', height: `${height}px` }}>
        <ResponsiveContainer width="100%" height="100%">
          <RadarChart
            cx="50%"
            cy="50%"
            outerRadius="70%"
            data={data}
            margin={{ top: 10, right: 20, bottom: 10, left: 20 }}
          >
            <PolarGrid stroke="#1e1e32" />
            <PolarAngleAxis
              dataKey="metric"
              tick={{ fill: '#9898b8', fontSize: 11 }}
            />
            <PolarRadiusAxis
              angle={30}
              domain={[0, 1]}
              stroke="#5c5c82"
              tick={{ fill: '#5c5c82', fontSize: 10 }}
              tickFormatter={(v) => `${Math.round(v * 100)}%`}
            />
            <Tooltip
              content={({ active, payload }) => {
                if (active && payload && payload.length) {
                  const metricName = payload[0]?.payload?.metric;
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
                        {metricName}
                      </div>
                      {payload.map((entry, index) => {
                        const val = Number(entry.value);
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
                              {(val * 100).toFixed(1)}%
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
                fontSize: '11px',
                color: '#9898b8',
                paddingTop: '8px',
              }}
            />
            <Radar
              name="Standard LLM"
              dataKey="standard"
              stroke="#f87171"
              fill="#f87171"
              fillOpacity={0.25}
              strokeWidth={2}
            />
            <Radar
              name="RAG Pipeline"
              dataKey="rag"
              stroke="#2dd4bf"
              fill="#2dd4bf"
              fillOpacity={0.3}
              strokeWidth={2}
            />
          </RadarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default ComparisonRadar;
