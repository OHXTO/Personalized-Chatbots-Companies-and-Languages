import React from 'react';
import { Link } from 'react-router-dom';
import { ROUTE_CONSTANTS } from '../constants/RouteConstants';

function Footer() {
  return (
    <footer className="bg-gray-900 text-gray-200">
      <div className="container mx-auto px-4 py-10">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          <div>
            <h3 className="text-xl font-semibold mb-3">AI Academy</h3>
            <p className="text-gray-400">
              Learn, build, and launch your AI career with guided projects and expert instruction.
            </p>
          </div>
          <div>
            <h4 className="font-semibold mb-3">Explore</h4>
            <div className="flex flex-col space-y-2">
              <Link to={ROUTE_CONSTANTS.ABOUT} className="hover:text-white transition-colors">
                About
              </Link>
              <Link to={ROUTE_CONSTANTS.COURSES} className="hover:text-white transition-colors">
                Courses
              </Link>
              <Link to={ROUTE_CONSTANTS.CONTACT} className="hover:text-white transition-colors">
                Contact
              </Link>
            </div>
          </div>
          <div>
            <h4 className="font-semibold mb-3">Get in touch</h4>
            <p className="text-gray-400">support@aiacademy.example</p>
            <p className="text-gray-400">(555) 123-4567</p>
          </div>
        </div>
        <div className="border-t border-gray-800 mt-8 pt-6 text-sm text-gray-500">
          © 2024 AI Academy. All rights reserved.
        </div>
      </div>
    </footer>
  );
}

export default Footer;
