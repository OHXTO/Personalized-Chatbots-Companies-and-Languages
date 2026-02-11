import React from 'react';
import { Link } from 'react-router-dom';
import { ROUTE_CONSTANTS } from '../constants/RouteConstants';

function HomePage() {
  return (
    <div>
      {/* Hero Section */}
      <section className="bg-gradient-to-r from-academy-primary to-ai-purple text-white py-20">
        <div className="container mx-auto px-4 text-center">
          <h1 className="text-5xl font-bold mb-6">
            Master AI & Machine Learning
          </h1>
          <p className="text-xl mb-8 max-w-2xl mx-auto">
            Join thousands of students learning cutting-edge AI technologies through hands-on projects, 
            expert instruction, and real-world applications.
          </p>
          <div className="space-x-4">
            <Link to={ROUTE_CONSTANTS.COURSES} className="btn-secondary text-lg px-8 py-3">
              Explore Courses
            </Link>
            <Link to="/dashboard" className="bg-white text-academy-primary hover:bg-gray-100 font-bold py-3 px-8 rounded text-lg transition-colors">
              Access Dashboard
            </Link>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-16 bg-gray-50">
        <div className="container mx-auto px-4">
          <h2 className="text-3xl font-bold text-center mb-12">Why Choose AI Academy?</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div className="text-center">
              <div className="text-4xl mb-4">🧠</div>
              <h3 className="text-xl font-semibold mb-2">Expert-Led Courses</h3>
              <p className="text-gray-600">Learn from industry experts with years of practical AI experience</p>
            </div>
            <div className="text-center">
              <div className="text-4xl mb-4">💻</div>
              <h3 className="text-xl font-semibold mb-2">Hands-On Projects</h3>
              <p className="text-gray-600">Build real AI applications and models you can add to your portfolio</p>
            </div>
            <div className="text-center">
              <div className="text-4xl mb-4">🎓</div>
              <h3 className="text-xl font-semibold mb-2">Career Support</h3>
              <p className="text-gray-600">Get job placement assistance and career guidance in the AI field</p>
            </div>
          </div>
        </div>
      </section>

      {/* Popular Courses */}
      <section className="py-16">
        <div className="container mx-auto px-4">
          <h2 className="text-3xl font-bold text-center mb-12">Popular Courses</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {[
              {
                title: "Introduction to AI",
                description: "Perfect for beginners. Learn AI fundamentals and basic concepts.",
                duration: "6 weeks",
                level: "Beginner"
              },
              {
                title: "Machine Learning Foundations",
                description: "Master supervised and unsupervised learning algorithms.",
                duration: "8 weeks", 
                level: "Intermediate"
              },
              {
                title: "Deep Learning & Neural Networks",
                description: "Build and train neural networks for complex problems.",
                duration: "10 weeks",
                level: "Advanced"
              }
            ].map((course, index) => (
              <div key={index} className="card hover:shadow-xl transition-shadow">
                <h3 className="text-xl font-semibold mb-2">{course.title}</h3>
                <p className="text-gray-600 mb-4">{course.description}</p>
                <div className="flex justify-between items-center text-sm text-gray-500 mb-4">
                  <span>📅 {course.duration}</span>
                  <span>📊 {course.level}</span>
                </div>
                <Link to={ROUTE_CONSTANTS.COURSES} className="btn-primary w-full text-center">
                  Learn More
                </Link>
              </div>
            ))}
          </div>
        </div>
      </section>
    </div>
  );
}

export default HomePage;