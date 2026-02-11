import React from 'react';
import { Link } from 'react-router-dom';
import { ROUTE_CONSTANTS } from '../constants/RouteConstants';

const courses = [
  {
    title: 'Introduction to AI',
    description: 'Start with core AI concepts, search, and knowledge representation.',
    duration: '6 weeks',
    level: 'Beginner',
    path: ROUTE_CONSTANTS.INTRO_TO_AI,
  },
  {
    title: 'Machine Learning Foundations',
    description: 'Supervised and unsupervised methods with evaluation best practices.',
    duration: '8 weeks',
    level: 'Intermediate',
    path: ROUTE_CONSTANTS.MACHINE_LEARNING,
  },
  {
    title: 'Deep Learning & Neural Networks',
    description: 'Train networks for vision, text, and sequence modeling.',
    duration: '10 weeks',
    level: 'Advanced',
    path: ROUTE_CONSTANTS.DEEP_LEARNING,
  },
  {
    title: 'Natural Language Processing',
    description: 'Build NLP pipelines and transformer-based models.',
    duration: '8 weeks',
    level: 'Intermediate',
    path: ROUTE_CONSTANTS.NLP,
  },
];

function Courses() {
  return (
    <div className="py-16 bg-gray-50">
      <div className="container mx-auto px-4">
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold mb-4">Courses</h1>
          <p className="text-lg text-gray-600 max-w-2xl mx-auto">
            Choose a learning path and start building practical AI skills with
            real-world projects.
          </p>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {courses.map((course) => (
            <div key={course.title} className="card flex flex-col">
              <h2 className="text-xl font-semibold mb-2">{course.title}</h2>
              <p className="text-gray-600 mb-4">{course.description}</p>
              <div className="flex justify-between text-sm text-gray-500 mb-6">
                <span>{course.duration}</span>
                <span>{course.level}</span>
              </div>
              <Link to={course.path} className="btn-primary mt-auto text-center">
                View Course
              </Link>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export default Courses;
