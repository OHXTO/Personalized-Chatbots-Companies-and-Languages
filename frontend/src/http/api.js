import { API_ENDPOINTS } from '../constants/RouteConstants';

const API_BASE_URL = API_ENDPOINTS.BASE_URL;
const IS_DEV = process.env.NODE_ENV !== 'production';

const MOCK_COURSES = [
  {
    id: 1,
    title: 'Introduction to AI',
    description: 'Learn the fundamentals of Artificial Intelligence and core concepts.',
    progress: 42,
    lessons: 12,
  },
  {
    id: 2,
    title: 'Machine Learning Basics',
    description: 'Master supervised and unsupervised learning techniques.',
    progress: 18,
    lessons: 16,
  },
  {
    id: 3,
    title: 'Deep Learning & Neural Networks',
    description: 'Build and train neural networks for real-world problems.',
    progress: 0,
    lessons: 20,
  },
];

const MOCK_COURSE_DETAILS = {
  1: {
    id: 1,
    title: 'Introduction to AI',
    description: 'Learn the fundamentals of Artificial Intelligence and core concepts.',
    progress: 42,
    completedLessons: 5,
    totalLessons: 12,
    timeSpent: 6,
    grade: 'B+',
    lessons: [
      {
        id: 101,
        title: 'What is Artificial Intelligence?',
        description: 'Introduction to AI concepts and history.',
        duration: 45,
        type: 'video',
        completed: true,
      },
      {
        id: 102,
        title: 'Types of AI Systems',
        description: 'Understanding different AI paradigms.',
        duration: 30,
        type: 'reading',
        completed: true,
      },
      {
        id: 103,
        title: 'AI Ethics and Responsibility',
        description: 'Exploring ethical considerations in AI.',
        duration: 60,
        type: 'video',
        completed: false,
      },
    ],
  },
  2: {
    id: 2,
    title: 'Machine Learning Basics',
    description: 'Master supervised and unsupervised learning techniques.',
    progress: 18,
    completedLessons: 3,
    totalLessons: 16,
    timeSpent: 4,
    grade: null,
    lessons: [
      {
        id: 201,
        title: 'Supervised Learning Fundamentals',
        description: 'Introduction to supervised learning algorithms.',
        duration: 50,
        type: 'video',
        completed: true,
      },
      {
        id: 202,
        title: 'Linear Regression Deep Dive',
        description: 'Mathematical foundations of linear regression.',
        duration: 40,
        type: 'reading',
        completed: false,
      },
    ],
  },
  3: {
    id: 3,
    title: 'Deep Learning & Neural Networks',
    description: 'Build and train neural networks for real-world problems.',
    progress: 0,
    completedLessons: 0,
    totalLessons: 20,
    timeSpent: 0,
    grade: null,
    lessons: [
      {
        id: 301,
        title: 'Neural Network Foundations',
        description: 'Understand neurons, layers, and activation functions.',
        duration: 55,
        type: 'video',
        completed: false,
      },
    ],
  },
};

const MOCK_PROGRESS = {
  student_id: 2,
  courses: [
    {
      course_id: 1,
      course_title: 'Introduction to AI',
      progress_percentage: 42,
      enrollment_date: new Date().toISOString(),
      lessons_completed: 5,
      total_lessons: 12,
      time_spent_hours: 6,
    },
    {
      course_id: 2,
      course_title: 'Machine Learning Basics',
      progress_percentage: 18,
      enrollment_date: new Date().toISOString(),
      lessons_completed: 3,
      total_lessons: 16,
      time_spent_hours: 4,
    },
  ],
  total_courses: 2,
  average_progress: 30,
};

// Generic API request helper
async function apiRequest(endpoint, options = {}) {
  const url = `${API_BASE_URL}${endpoint}`;
  const config = {
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
    ...options,
  };

  try {
    const response = await fetch(url, config);
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('API request failed:', error);
    throw error;
  }
}

// Health check
export async function checkHealth() {
  return apiRequest('/health');
}

// Course APIs
export async function getCourses() {
  try {
    return await apiRequest('/api/courses');
  } catch (error) {
    if (IS_DEV) {
      return MOCK_COURSES;
    }
    throw error;
  }
}

export async function getCourse(courseId) {
  try {
    return await apiRequest(`/api/courses/${courseId}`);
  } catch (error) {
    if (IS_DEV) {
      return MOCK_COURSE_DETAILS[courseId] || null;
    }
    throw error;
  }
}

// Student APIs
export async function getStudentProgress(studentId) {
  try {
    return await apiRequest(`/api/students/${studentId}/progress`);
  } catch (error) {
    if (IS_DEV) {
      return { ...MOCK_PROGRESS, student_id: studentId };
    }
    throw error;
  }
}

// Prediction API (AI functionality)
export async function predict(inputData) {
  return apiRequest('/api/predict', {
    method: 'POST',
    body: JSON.stringify(inputData),
  });
}

// Course enrollment
export async function enrollInCourse(courseId, studentData) {
  return apiRequest(`/api/courses/${courseId}/enroll`, {
    method: 'POST',
    body: JSON.stringify(studentData),
  });
}
