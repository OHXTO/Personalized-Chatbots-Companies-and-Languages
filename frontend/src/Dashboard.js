import React from 'react';
import { Routes, Route } from 'react-router-dom';
import MainNavbar from './components/MainNavbar';
import CourseDashboard from './components/CourseDashboard';
import CourseDetail from './components/CourseDetail';
import StudentProgress from './components/StudentProgress';
import { ROUTE_CONSTANTS } from './constants/RouteConstants';

function Dashboard() {
  return (
    <div className="min-h-screen bg-gray-50">
      <MainNavbar />
      <main className="container mx-auto px-4 py-8">
        <Routes>
          <Route path={ROUTE_CONSTANTS.DASHBOARD} element={<CourseDashboard />} />
          <Route path={ROUTE_CONSTANTS.COURSE_DETAIL} element={<CourseDetail />} />
          <Route path={ROUTE_CONSTANTS.STUDENT_PROGRESS} element={<StudentProgress />} />
        </Routes>
      </main>
    </div>
  );
}

export default Dashboard;