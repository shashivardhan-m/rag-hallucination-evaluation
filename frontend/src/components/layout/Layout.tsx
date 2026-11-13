import React from 'react';
import { Outlet } from 'react-router-dom';
import { Sidebar } from './Sidebar';
import { TopBar } from './TopBar';

export const Layout: React.FC = () => {
  return (
    <div className="app-layout min-h-screen flex w-full" style={{ backgroundColor: 'var(--color-bg-primary)' }}>
      {/* Fixed Sidebar */}
      <Sidebar />

      {/* Main Content Area */}
      <div className="main-content flex-1 flex flex-col min-h-screen">
        <TopBar />
        <main className="page-content flex-1 w-full mx-auto">
          <Outlet />
        </main>
      </div>
    </div>
  );
};

export default Layout;
