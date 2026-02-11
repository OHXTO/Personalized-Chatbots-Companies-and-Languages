-- AI Academy Database Schema

-- Users table for student and instructor accounts
CREATE TABLE users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    first_name VARCHAR(100) NOT NULL,
    last_name VARCHAR(100) NOT NULL,
    role ENUM('student', 'instructor', 'admin') DEFAULT 'student',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- Courses table for AI/ML courses
CREATE TABLE courses (
    id INT AUTO_INCREMENT PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    instructor_id INT NOT NULL,
    difficulty_level ENUM('beginner', 'intermediate', 'advanced') DEFAULT 'beginner',
    duration_weeks INT DEFAULT 6,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (instructor_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Lessons table for individual course lessons
CREATE TABLE lessons (
    id INT AUTO_INCREMENT PRIMARY KEY,
    course_id INT NOT NULL,
    title VARCHAR(255) NOT NULL,
    content TEXT,
    lesson_order INT NOT NULL,
    duration_minutes INT DEFAULT 30,
    lesson_type ENUM('video', 'reading', 'quiz', 'project') DEFAULT 'video',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (course_id) REFERENCES courses(id) ON DELETE CASCADE
);

-- Enrollments table for student course enrollments
CREATE TABLE enrollments (
    id INT AUTO_INCREMENT PRIMARY KEY,
    student_id INT NOT NULL,
    course_id INT NOT NULL,
    enrollment_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completion_date TIMESTAMP NULL,
    progress_percentage DECIMAL(5,2) DEFAULT 0.00,
    FOREIGN KEY (student_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (course_id) REFERENCES courses(id) ON DELETE CASCADE,
    UNIQUE KEY unique_enrollment (student_id, course_id)
);

-- Progress tracking for individual lessons
CREATE TABLE lesson_progress (
    id INT AUTO_INCREMENT PRIMARY KEY,
    student_id INT NOT NULL,
    lesson_id INT NOT NULL,
    completed BOOLEAN DEFAULT FALSE,
    completion_date TIMESTAMP NULL,
    time_spent_minutes INT DEFAULT 0,
    FOREIGN KEY (student_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (lesson_id) REFERENCES lessons(id) ON DELETE CASCADE,
    UNIQUE KEY unique_lesson_progress (student_id, lesson_id)
);

-- AI/ML predictions table for student model outputs
CREATE TABLE predictions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    student_id INT NULL,
    input_text TEXT,
    model_name VARCHAR(100) DEFAULT 'default_model',
    prediction_label VARCHAR(64) NOT NULL,
    confidence_score FLOAT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (student_id) REFERENCES users(id) ON DELETE SET NULL
);

-- Assignments table for course assignments
CREATE TABLE assignments (
    id INT AUTO_INCREMENT PRIMARY KEY,
    course_id INT NOT NULL,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    due_date TIMESTAMP NULL,
    max_points INT DEFAULT 100,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (course_id) REFERENCES courses(id) ON DELETE CASCADE
);

-- Submissions table for assignment submissions
CREATE TABLE submissions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    assignment_id INT NOT NULL,
    student_id INT NOT NULL,
    submission_content TEXT,
    submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    grade DECIMAL(5,2) NULL,
    feedback TEXT NULL,
    FOREIGN KEY (assignment_id) REFERENCES assignments(id) ON DELETE CASCADE,
    FOREIGN KEY (student_id) REFERENCES users(id) ON DELETE CASCADE,
    UNIQUE KEY unique_submission (assignment_id, student_id)
);

-- Sample data insertion
INSERT INTO users (email, first_name, last_name, role) VALUES 
('john.doe@example.com', 'John', 'Doe', 'instructor'),
('jane.smith@example.com', 'Jane', 'Smith', 'student'),
('alice.wilson@example.com', 'Alice', 'Wilson', 'student');

INSERT INTO courses (title, description, instructor_id, difficulty_level, duration_weeks) VALUES 
('Introduction to AI', 'Learn the fundamentals of Artificial Intelligence', 1, 'beginner', 6),
('Machine Learning Basics', 'Master the core concepts of Machine Learning', 1, 'intermediate', 8),
('Deep Learning & Neural Networks', 'Advanced deep learning techniques', 1, 'advanced', 10);

INSERT INTO lessons (course_id, title, content, lesson_order, duration_minutes, lesson_type) VALUES 
(1, 'What is Artificial Intelligence?', 'Introduction to AI concepts and history', 1, 45, 'video'),
(1, 'Types of AI Systems', 'Understanding different AI paradigms', 2, 30, 'reading'),
(1, 'AI Ethics and Responsibility', 'Exploring ethical considerations in AI', 3, 60, 'video'),
(2, 'Supervised Learning Fundamentals', 'Introduction to supervised learning algorithms', 1, 50, 'video'),
(2, 'Linear Regression Deep Dive', 'Mathematical foundations of linear regression', 2, 40, 'reading'),
(2, 'Classification Algorithms', 'Decision trees, SVM, and more', 3, 55, 'video');

INSERT INTO enrollments (student_id, course_id, progress_percentage) VALUES 
(2, 1, 75.50),
(2, 2, 23.00),
(3, 1, 100.00),
(3, 2, 67.80);