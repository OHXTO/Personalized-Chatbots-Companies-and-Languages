import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Dashboard from './Dashboard';
import LandingPage from './landing_page/LandingPage';
import { ROUTE_CONSTANTS } from './constants/RouteConstants';

function App() {
  const currentHost = window.location.hostname;
  
  // Check if we're on the dashboard subdomain
  const isDashboard = currentHost.includes('dashboard') || currentHost.includes('app');
  
  return (
    <div className="App">
      <Router>
        <Routes>
          {isDashboard ? (
            <Route path="/*" element={<Dashboard />} />
          ) : (
            <>
              <Route path={`${ROUTE_CONSTANTS.DASHBOARD}/*`} element={<Dashboard />} />
              <Route path="/*" element={<LandingPage />} />
            </>
          )}
        </Routes>
      </Router>
    </div>
  );
}

export default App;
