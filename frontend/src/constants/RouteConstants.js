export const ROUTE_CONSTANTS = {
  // Landing Page Routes
  HOME: '/',
  ABOUT: '/about',
  COURSES: '/courses',
  CONTACT: '/contact',
  
  // Dashboard Routes
  DASHBOARD: '/dashboard',
  COURSE_DETAIL: '/course/:id',
  STUDENT_PROGRESS: '/progress',
  ASSIGNMENTS: '/assignments',
  GRADES: '/grades',
  
  // Course Specific Routes
  INTRO_TO_AI: '/courses/intro-to-ai',
  MACHINE_LEARNING: '/courses/machine-learning',
  DEEP_LEARNING: '/courses/deep-learning',
  NLP: '/courses/natural-language-processing',
};

export const API_ENDPOINTS = {
  BASE_URL: process.env.REACT_APP_API_ENDPOINT || 'http://localhost:5000',
  HEALTH: '/health',
  COURSES: '/api/courses',
  STUDENTS: '/api/students',
  PROGRESS: '/api/progress',
  PREDICT: '/api/predict',
};