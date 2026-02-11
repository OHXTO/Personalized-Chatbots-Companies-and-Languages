import mysql.connector
import os
from contextlib import contextmanager

# Database configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'user': os.getenv('DB_USER', 'root'),
    'password': os.getenv('DB_PASSWORD', 'password'),
    'database': os.getenv('DB_NAME', 'ai_academy'),
    'charset': 'utf8mb4',
    'autocommit': True
}

@contextmanager
def get_db_connection():
    """Context manager for database connections"""
    connection = None
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        yield connection
    except mysql.connector.Error as err:
        print(f"Database error: {err}")
        if connection:
            connection.rollback()
        raise
    finally:
        if connection and connection.is_connected():
            connection.close()

def get_db():
    """Simple database connection function"""
    return mysql.connector.connect(**DB_CONFIG)

# Course-related database functions
def get_all_courses():
    """Retrieve all courses with enrollment and lesson counts"""
    query = """
    SELECT 
        c.id, c.title, c.description, c.difficulty_level, c.duration_weeks,
        CONCAT(u.first_name, ' ', u.last_name) as instructor_name,
        COUNT(DISTINCT e.student_id) as enrolled_students,
        COUNT(DISTINCT l.id) as total_lessons
    FROM courses c
    LEFT JOIN users u ON c.instructor_id = u.id
    LEFT JOIN enrollments e ON c.id = e.course_id
    LEFT JOIN lessons l ON c.id = l.course_id
    GROUP BY c.id
    ORDER BY c.created_at DESC
    """
    
    with get_db_connection() as db:
        cursor = db.cursor(dictionary=True)
        cursor.execute(query)
        return cursor.fetchall()

def get_course_by_id(course_id):
    """Retrieve detailed course information by ID"""
    course_query = """
    SELECT 
        c.id, c.title, c.description, c.difficulty_level, c.duration_weeks,
        CONCAT(u.first_name, ' ', u.last_name) as instructor_name
    FROM courses c
    LEFT JOIN users u ON c.instructor_id = u.id
    WHERE c.id = %s
    """
    
    lessons_query = """
    SELECT id, title, content, lesson_order, duration_minutes, lesson_type
    FROM lessons 
    WHERE course_id = %s 
    ORDER BY lesson_order
    """
    
    with get_db_connection() as db:
        cursor = db.cursor(dictionary=True)
        
        # Get course details
        cursor.execute(course_query, (course_id,))
        course = cursor.fetchone()
        
        if course:
            # Get course lessons
            cursor.execute(lessons_query, (course_id,))
            course['lessons'] = cursor.fetchall()
        
        return course

def get_student_progress(student_id):
    """Get student's progress across all enrolled courses"""
    query = """
    SELECT 
        c.id as course_id, c.title as course_title,
        e.progress_percentage, e.enrollment_date,
        COUNT(lp.completed) as lessons_completed,
        COUNT(l.id) as total_lessons,
        SUM(lp.time_spent_minutes) as total_time_spent
    FROM enrollments e
    JOIN courses c ON e.course_id = c.id
    LEFT JOIN lessons l ON c.id = l.course_id
    LEFT JOIN lesson_progress lp ON l.id = lp.lesson_id AND lp.student_id = e.student_id AND lp.completed = TRUE
    WHERE e.student_id = %s
    GROUP BY c.id, e.id
    ORDER BY e.enrollment_date DESC
    """
    
    with get_db_connection() as db:
        cursor = db.cursor(dictionary=True)
        cursor.execute(query, (student_id,))
        return cursor.fetchall()

def insert_prediction(student_id, input_text, model_name, prediction_label, confidence_score):
    """Insert a new AI prediction into the database"""
    query = """
    INSERT INTO predictions (student_id, input_text, model_name, prediction_label, confidence_score)
    VALUES (%s, %s, %s, %s, %s)
    """
    
    with get_db_connection() as db:
        cursor = db.cursor()
        cursor.execute(query, (student_id, input_text, model_name, prediction_label, confidence_score))
        return cursor.lastrowid

def create_user(email, first_name, last_name, role='student'):
    """Create a new user account"""
    query = """
    INSERT INTO users (email, first_name, last_name, role)
    VALUES (%s, %s, %s, %s)
    """
    
    with get_db_connection() as db:
        cursor = db.cursor()
        cursor.execute(query, (email, first_name, last_name, role))
        return cursor.lastrowid

def enroll_student(student_id, course_id):
    """Enroll a student in a course"""
    query = """
    INSERT INTO enrollments (student_id, course_id)
    VALUES (%s, %s)
    ON DUPLICATE KEY UPDATE enrollment_date = CURRENT_TIMESTAMP
    """
    
    with get_db_connection() as db:
        cursor = db.cursor()
        cursor.execute(query, (student_id, course_id))
        return cursor.rowcount > 0