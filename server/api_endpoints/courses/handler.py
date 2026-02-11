from flask import jsonify
from database.db import get_all_courses, get_course_by_id

def CoursesHandler(request):
    """Handle requests for course listings"""
    try:
        courses = get_all_courses()
        
        # Format courses for frontend consumption
        formatted_courses = []
        for course in courses:
            formatted_courses.append({
                'id': course['id'],
                'title': course['title'],
                'description': course['description'],
                'instructor': course['instructor_name'],
                'difficulty': course['difficulty_level'],
                'duration': f"{course['duration_weeks']} weeks",
                'enrolled_students': course['enrolled_students'],
                'lessons': course['total_lessons'],
                'progress': 0  # This would be calculated per student
            })
        
        return jsonify(formatted_courses)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def CourseDetailHandler(request, course_id):
    """Handle requests for individual course details"""
    try:
        course = get_course_by_id(course_id)
        
        if not course:
            return jsonify({'error': 'Course not found'}), 404
        
        # Format course details for frontend
        formatted_course = {
            'id': course['id'],
            'title': course['title'],
            'description': course['description'],
            'instructor': course['instructor_name'],
            'difficulty': course['difficulty_level'],
            'duration': f"{course['duration_weeks']} weeks",
            'lessons': course.get('lessons', []),
            'progress': 0,  # This would be student-specific
            'completedLessons': 0,  # This would be calculated
            'totalLessons': len(course.get('lessons', [])),
            'timeSpent': 0,  # This would be calculated from lesson_progress
            'grade': None   # This would be calculated from assignments
        }
        
        return jsonify(formatted_course)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500