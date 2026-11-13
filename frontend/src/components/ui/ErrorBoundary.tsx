import { Component } from 'react';
import type { ErrorInfo, ReactNode } from 'react';
import { AlertTriangle, RotateCcw, Home } from 'lucide-react';

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
  showDetails: boolean;
}

export class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false,
    error: null,
    errorInfo: null,
    showDetails: false,
  };

  public static getDerivedStateFromError(error: Error): Partial<State> {
    return { hasError: true, error };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('ErrorBoundary caught an error:', error, errorInfo);
    this.setState({ errorInfo });
  }

  private handleReset = () => {
    this.setState({ hasError: false, error: null, errorInfo: null, showDetails: false });
    window.location.reload();
  };

  public render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback;
      }

      return (
        <div className="min-h-[400px] flex items-center justify-center p-6">
          <div
            className="card max-w-lg w-full text-center p-8 space-y-5 border"
            style={{
              backgroundColor: 'var(--color-bg-card)',
              borderColor: 'rgba(248, 113, 113, 0.3)',
            }}
          >
            <div
              className="w-14 h-14 mx-auto rounded-2xl flex items-center justify-center border shadow-lg"
              style={{
                backgroundColor: 'rgba(248, 113, 113, 0.1)',
                borderColor: 'rgba(248, 113, 113, 0.25)',
                color: 'var(--color-rose)',
              }}
            >
              <AlertTriangle className="w-7 h-7" />
            </div>

            <div className="space-y-1.5">
              <h2
                className="text-lg font-bold"
                style={{ color: 'var(--color-text-primary)' }}
              >
                Something went wrong
              </h2>
              <p
                className="text-xs"
                style={{ color: 'var(--color-text-secondary)' }}
              >
                An error occurred while rendering this view. You can reload the page or navigate back to the dashboard.
              </p>
            </div>

            {this.state.error && (
              <div
                className="p-3 rounded-lg text-left text-xs font-mono border overflow-x-auto"
                style={{
                  backgroundColor: 'var(--color-bg-tertiary)',
                  borderColor: 'var(--color-border)',
                  color: 'var(--color-rose)',
                }}
              >
                {this.state.error.message || 'Unknown runtime error'}
              </div>
            )}

            <div className="flex items-center justify-center gap-3 pt-2">
              <button
                type="button"
                onClick={this.handleReset}
                className="btn btn-primary btn-sm flex items-center gap-2"
              >
                <RotateCcw className="w-3.5 h-3.5" />
                Reload View
              </button>
              <a
                href="/"
                className="btn btn-secondary btn-sm flex items-center gap-2"
              >
                <Home className="w-3.5 h-3.5" />
                Return to Overview
              </a>
            </div>

            {/* Collapsible stack trace */}
            <div className="pt-2">
              <button
                type="button"
                onClick={() => this.setState((s) => ({ showDetails: !s.showDetails }))}
                className="text-[11px] underline opacity-60 hover:opacity-100"
                style={{ color: 'var(--color-text-tertiary)' }}
              >
                {this.state.showDetails ? 'Hide technical details' : 'Show technical details'}
              </button>

              {this.state.showDetails && this.state.errorInfo && (
                <pre
                  className="mt-3 p-3 text-[10px] text-left overflow-auto max-h-48 rounded bg-black/40 font-mono text-gray-400 border border-gray-800"
                >
                  {this.state.errorInfo.componentStack}
                </pre>
              )}
            </div>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
