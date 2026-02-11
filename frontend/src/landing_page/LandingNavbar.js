import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { ROUTE_CONSTANTS } from '../constants/RouteConstants';

const navItems = [
  { path: ROUTE_CONSTANTS.HOME, label: 'Home' },
  { path: ROUTE_CONSTANTS.ABOUT, label: 'About' },
  { path: ROUTE_CONSTANTS.COURSES, label: 'Courses' },
  { path: ROUTE_CONSTANTS.CONTACT, label: 'Contact' },
];

function LandingNavbar() {
  const location = useLocation();

  return (
    <nav className="bg-white shadow-sm">
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between py-4">
          <Link to={ROUTE_CONSTANTS.HOME} className="text-2xl font-bold text-academy-primary">
            AI Academy
          </Link>
          <div className="flex items-center space-x-6">
            {navItems.map((item) => (
              <Link
                key={item.path}
                to={item.path}
                className={`text-gray-700 hover:text-academy-primary transition-colors ${
                  location.pathname === item.path ? 'text-academy-primary font-semibold' : ''
                }`}
              >
                {item.label}
              </Link>
            ))}
            <Link to={ROUTE_CONSTANTS.DASHBOARD} className="btn-secondary text-sm">
              Dashboard
            </Link>
          </div>
        </div>
      </div>
    </nav>
  );
}

export default LandingNavbar;
