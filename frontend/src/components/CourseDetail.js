import React, { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import { getCourse } from '../http/api';

function CourseDetail() {
  const { id } = useParams();
  const [course, setCourse] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchCourse = async () => {
      try {
        const data = await getCourse(id);
        setCourse(data);
      } catch (error) {
        console.error('Failed to fetch course:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchCourse();
  }, [id]);

  if (loading) {
    return <div className="text-center py-8">Loading course...</div>;
  }

  if (!course) {
    return <div className="text-center py-8">Course not found</div>;
  }

  return (
    <div className="max-w-4xl mx-auto">
      <div className="mb-8">
        <h1 className="text-4xl font-bold text-gray-900 mb-4">{course.title}</h1>
        <p className="text-xl text-gray-600">{course.description}</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        <div className="lg:col-span-2">
          <div className="card">
            <h2 className="text-2xl font-semibold mb-4">Course Content</h2>
            <div className="space-y-4">
              {course.lessons?.map((lesson, index) => (
                <div key={lesson.id} className="border-l-4 border-academy-primary pl-4">
                  <h3 className="font-semibold text-lg">
                    Lesson {index + 1}: {lesson.title}
                  </h3>
                  <p className="text-gray-600 mb-2">{lesson.description}</p>
                  <div className="flex items-center space-x-4 text-sm text-gray-500">
                    <span>⏱️ {lesson.duration} minutes</span>
                    <span>📚 {lesson.type}</span>
                    {lesson.completed && <span className="text-green-600">✅ Completed</span>}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="lg:col-span-1">
          <div className="card">
            <h3 className="text-xl font-semibold mb-4">Course Progress</h3>
            <div className="mb-4">
              <div className="flex justify-between text-sm text-gray-500 mb-1">
                <span>Overall Progress</span>
                <span>{course.progress}%</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-3">
                <div 
                  className="bg-academy-primary h-3 rounded-full" 
                  style={{ width: `${course.progress}%` }}
                ></div>
              </div>
            </div>
            
            <div className="space-y-3 text-sm">
              <div className="flex justify-between">
                <span>Lessons Completed:</span>
                <span className="font-semibold">{course.completedLessons}/{course.totalLessons}</span>
              </div>
              <div className="flex justify-between">
                <span>Time Spent:</span>
                <span className="font-semibold">{course.timeSpent} hours</span>
              </div>
              <div className="flex justify-between">
                <span>Grade:</span>
                <span className="font-semibold text-academy-primary">{course.grade || 'N/A'}</span>
              </div>
            </div>
            
            <button className="w-full btn-primary mt-6">
              Continue Course
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default CourseDetail;