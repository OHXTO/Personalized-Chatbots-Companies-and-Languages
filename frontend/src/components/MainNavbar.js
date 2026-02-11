import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { ROUTE_CONSTANTS } from '../constants/RouteConstants';

function MainNavbar() {
  const location = useLocation();
  
  const navItems = [
    { path: ROUTE_CONSTANTS.DASHBOARD, label: 'Dashboard' },
    { path: ROUTE_CONSTANTS.COURSES, label: 'Courses' },
    { path: ROUTE_CONSTANTS.STUDENT_PROGRESS, label: 'Progress' },
    { path: ROUTE_CONSTANTS.ASSIGNMENTS, label: 'Assignments' },
  ];
  
  return (
    <nav className="bg-academy-primary shadow-lg">
      <div className="container mx-auto px-4">
        <div className="flex justify-between items-center py-4">
          <Link to={ROUTE_CONSTANTS.DASHBOARD} className="text-white text-xl font-bold">
            🤖 AI Academy
          </Link>
          
          <div className="flex space-x-6">
            {navItems.map((item) => (
              <Link
                key={item.path}
                to={item.path}
                className={`text-white hover:text-academy-secondary transition-colors $'{
                  location.pathname === item.path ? 'text-academy-secondary font-semibold' : ''
                }`}
              >
                {item.label}
              </Link>
            ))}
          </div>
          
          <div className="flex items-center space-x-4">
            <span className="text-white">Welcome, Student!</span>
            <button className="btn-secondary text-sm">
              Logout
            </button>
          </div>
        </div>
      </div>
    </nav>
  );
}

export default MainNavbar;