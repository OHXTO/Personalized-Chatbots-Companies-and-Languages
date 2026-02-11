import React from 'react';

function Contact() {
  return (
    <div className="py-16 bg-white">
      <div className="container mx-auto px-4 max-w-4xl">
        <h1 className="text-4xl font-bold mb-4 text-center">Contact Us</h1>
        <p className="text-lg text-gray-600 mb-10 text-center">
          Have questions about courses or partnerships? We would love to hear from you.
        </p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          <div className="card">
            <h2 className="text-2xl font-semibold mb-3">Reach Out</h2>
            <p className="text-gray-600 mb-4">
              Email us and our team will respond within 1-2 business days.
            </p>
            <div className="text-gray-700">
              <p className="mb-2">support@aiacademy.example</p>
              <p>(555) 123-4567</p>
            </div>
          </div>
          <div className="card">
            <h2 className="text-2xl font-semibold mb-4">Send a Message</h2>
            <form className="space-y-4">
              <input
                type="text"
                placeholder="Your name"
                className="w-full border border-gray-300 rounded px-3 py-2"
              />
              <input
                type="email"
                placeholder="Your email"
                className="w-full border border-gray-300 rounded px-3 py-2"
              />
              <textarea
                rows="4"
                placeholder="How can we help?"
                className="w-full border border-gray-300 rounded px-3 py-2"
              />
              <button type="button" className="btn-primary w-full">
                Send Message
              </button>
            </form>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Contact;
