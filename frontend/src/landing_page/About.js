import React from 'react';

function About() {
  return (
    <div className="py-16 bg-white">
      <div className="container mx-auto px-4 max-w-4xl">
        <h1 className="text-4xl font-bold mb-6 text-center">About AI Academy</h1>
        <p className="text-lg text-gray-700 mb-6">
          AI Academy is a learning platform focused on practical, career-ready AI
          skills. We combine structured coursework with guided projects so you
          can build real models and deploy them with confidence.
        </p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          <div className="card">
            <h2 className="text-2xl font-semibold mb-3">Our Mission</h2>
            <p className="text-gray-600">
              Make advanced AI education accessible through clear instruction,
              mentorship, and hands-on labs.
            </p>
          </div>
          <div className="card">
            <h2 className="text-2xl font-semibold mb-3">What You Get</h2>
            <p className="text-gray-600">
              Structured learning paths, project feedback, and community support
              to help you land a role in AI.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default About;
