import React from 'react';
import { NavLink } from 'react-router-dom';
import {
  LayoutDashboard,
  FileText,
  MessageSquare,
  FlaskConical,
  Beaker,
  BarChart2,
  Settings,
  Moon,
  Sun,
  Sparkles,
  ShieldCheck,
} from 'lucide-react';
import { useSettingsStore } from '../../store/settingsStore';

interface NavItem {
  name: string;
  to: string;
  icon: React.ComponentType<{ className?: string }>;
  badge?: string;
}

const navItems: NavItem[] = [
  { name: 'Overview', to: '/', icon: LayoutDashboard },
  { name: 'Documents', to: '/documents', icon: FileText },
  { name: 'RAG Playground', to: '/playground', icon: MessageSquare },
  { name: 'Evaluation', to: '/evaluation', icon: FlaskConical },
  { name: 'Experiment Lab', to: '/experiments', icon: Beaker },
  { name: 'Analytics', to: '/analytics', icon: BarChart2 },
  { name: 'Settings', to: '/settings', icon: Settings },
];

export const Sidebar: React.FC = () => {
  const { theme, toggleTheme } = useSettingsStore();

  return (
    <aside
      className="fixed left-0 top-0 bottom-0 z-30 flex flex-col justify-between border-r select-none transition-all duration-200"
      style={{
        width: 'var(--sidebar-width)',
        backgroundColor: 'var(--color-bg-secondary)',
        borderColor: 'var(--color-border)',
      }}
    >
      {/* ── Top Header / Logo ── */}
      <div>
        <div
          className="flex items-center gap-3 px-5 py-5 border-b"
          style={{ borderColor: 'var(--color-border)' }}
        >
          <div
            className="flex items-center justify-center w-9 h-9 rounded-xl shadow-lg"
            style={{
              background: 'linear-gradient(135deg, var(--color-accent-primary), #9333ea)',
              color: '#ffffff',
            }}
          >
            <Sparkles className="w-5 h-5 animate-pulse" />
          </div>
          <div className="flex flex-col">
            <div className="flex items-center gap-1.5">
              <span
                className="font-bold tracking-tight text-base"
                style={{
                  background: 'linear-gradient(135deg, #f1f0ff 0%, var(--color-accent-secondary) 100%)',
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                }}
              >
                RAG Eval
              </span>
              <span className="badge badge-purple text-[10px] px-1.5 py-0.5">
                v1.0
              </span>
            </div>
            <span
              className="text-[11px] font-medium"
              style={{ color: 'var(--color-text-tertiary)' }}
            >
              Hallucination Suite
            </span>
          </div>
        </div>

        {/* ── Navigation Links ── */}
        <nav className="p-3 space-y-1">
          <div
            className="px-3 py-2 text-[10px] font-semibold uppercase tracking-wider"
            style={{ color: 'var(--color-text-muted)' }}
          >
            Navigation
          </div>

          {navItems.map((item) => {
            const Icon = item.icon;
            return (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.to === '/'}
                className={({ isActive }) =>
                  `flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all duration-150 ${
                    isActive
                      ? 'border-l-2 text-white font-semibold shadow-sm'
                      : 'hover:text-white'
                  }`
                }
                style={({ isActive }) => ({
                  backgroundColor: isActive
                    ? 'var(--color-bg-elevated)'
                    : 'transparent',
                  borderLeftColor: isActive
                    ? 'var(--color-accent-primary)'
                    : 'transparent',
                  color: isActive
                    ? 'var(--color-text-primary)'
                    : 'var(--color-text-secondary)',
                })}
              >
                {({ isActive }) => (
                  <>
                    <Icon
                      className={`w-4 h-4 transition-colors ${
                        isActive ? 'text-[var(--color-accent-primary)]' : ''
                      }`}
                    />
                    <span className="flex-1 truncate">{item.name}</span>
                    {item.badge && (
                      <span className="badge badge-teal text-[10px] px-1.5 py-0.2">
                        {item.badge}
                      </span>
                    )}
                  </>
                )}
              </NavLink>
            );
          })}
        </nav>
      </div>

      {/* ── Bottom Section ── */}
      <div
        className="p-3 border-t space-y-2"
        style={{ borderColor: 'var(--color-border)' }}
      >
        {/* System Status Indicator */}
        <div
          className="flex items-center justify-between px-3 py-2 rounded-lg text-xs"
          style={{
            backgroundColor: 'var(--color-bg-tertiary)',
            border: '1px solid var(--color-border-subtle)',
          }}
        >
          <div className="flex items-center gap-2">
            <span className="relative flex h-2 w-2">
              <span
                className="animate-ping absolute inline-flex h-full w-full rounded-full opacity-75"
                style={{ backgroundColor: 'var(--color-emerald)' }}
              />
              <span
                className="relative inline-flex rounded-full h-2 w-2"
                style={{ backgroundColor: 'var(--color-emerald)' }}
              />
            </span>
            <span
              className="font-medium text-[11px]"
              style={{ color: 'var(--color-text-secondary)' }}
            >
              System Ready
            </span>
          </div>
          <ShieldCheck
            className="w-3.5 h-3.5"
            style={{ color: 'var(--color-emerald)' }}
          />
        </div>

        {/* Theme Toggle Button */}
        <button
          type="button"
          onClick={toggleTheme}
          className="btn btn-ghost w-full justify-between text-xs py-2 px-3 rounded-lg border hover:border-[var(--color-border-active)] transition-all"
          style={{
            borderColor: 'var(--color-border)',
            color: 'var(--color-text-secondary)',
          }}
          title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
        >
          <span className="flex items-center gap-2">
            {theme === 'dark' ? (
              <Moon className="w-4 h-4 text-purple-400" />
            ) : (
              <Sun className="w-4 h-4 text-amber-400" />
            )}
            <span>{theme === 'dark' ? 'Dark Mode' : 'Light Mode'}</span>
          </span>
          <span
            className="text-[10px] px-1.5 py-0.5 rounded font-mono"
            style={{
              backgroundColor: 'var(--color-bg-tertiary)',
              color: 'var(--color-text-tertiary)',
            }}
          >
            {theme.toUpperCase()}
          </span>
        </button>
      </div>
    </aside>
  );
};

export default Sidebar;
