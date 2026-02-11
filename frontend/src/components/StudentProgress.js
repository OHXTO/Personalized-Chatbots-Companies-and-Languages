import React, { useState, useEffect } from 'react';
import { getStudentProgress } from '../http/api';

function StudentProgress() {
  const [progress, setProgress] = useState(null);
  const [loading, setLoading] = useState(true);
  
  // Mock student ID - in real app, this would come from auth
  const studentId = 2;

  useEffect(() => {
    const fetchProgress = async () => {
      try {
        const data = await getStudentProgress(studentId);
        setProgress(data);
      } catch (error) {
        console.error('Failed to fetch progress:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchProgress();
  }, [studentId]);

  if (loading) {
    return <div className="text-center py-8">Loading progress...</div>;
  }

  if (!progress || !progress.courses) {
    return <div className="text-center py-8">No progress data available</div>;
  }

  return (
    <div className="max-w-4xl mx-auto">
      <h1 className="text-3xl font-bold mb-8">Your Learning Progress</h1>
      
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
        <div className="card text-center">
          <h3 className="text-lg font-semibold mb-2">Enrolled Courses</h3>
          <p className="text-3xl font-bold text-academy-primary">{progress.total_courses}</p>
        </div>
        <div className="card text-center">
          <h3 className="text-lg font-semibold mb-2">Average Progress</h3>
          <p className="text-3xl font-bold text-academy-secondary">
            {Math.round(progress.average_progress)}%
          </p>
        </div>
        <div className="card text-center">
          <h3 className="text-lg font-semibold mb-2">Total Time</h3>
          <p className="text-3xl font-bold text-ai-green">
            {progress.courses.reduce((total, course) => total + course.time_spent_hours, 0)}h
          </p>
        </div>
      </div>

      <div className="space-y-6">
        {progress.courses.map((course) => (
          <div key={course.course_id} className="card">
            <div className="flex justify-between items-start mb-4">
              <div>
                <h3 className="text-xl font-semibold mb-2">{course.course_title}</h3>
                <p className="text-gray-600">
                  Enrolled: {new Date(course.enrollment_date).toLocaleDateString()}
                </p>
              </div>
              <div className="text-right">
                <p className="text-2xl font-bold text-academy-primary">
                  {Math.round(course.progress_percentage)}%
                </p>
                <p className="text-sm text-gray-500">Complete</p>
              </div>
            </div>
            
            <div className="mb-4">
              <div className="w-full bg-gray-200 rounded-full h-3">
                <div 
                  className="bg-academy-primary h-3 rounded-full transition-all duration-300" 
                  style={{ width: `${course.progress_percentage}%` }}
                ></div>
              </div>
            </div>
            
            <div className="grid grid-cols-3 gap-4 text-sm">
              <div>
                <span className="text-gray-500">Lessons:</span>
                <p className="font-semibold">
                  {course.lessons_completed} / {course.total_lessons}
                </p>
              </div>
              <div>
                <span className="text-gray-500">Time Spent:</span>
                <p className="font-semibold">{course.time_spent_hours}h</p>
              </div>
              <div>
                <span className="text-gray-500">Status:</span>
                <p className="font-semibold">
                  {course.progress_percentage >= 100 ? 'Completed' : 'In Progress'}
                </p>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default StudentProgress;