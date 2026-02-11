from flask import jsonify
from database.db import get_student_progress

def StudentProgressHandler(request, student_id):
    """Handle requests for student progress data"""
    try:
        progress_data = get_student_progress(student_id)
        
        # Format progress data for frontend consumption
        formatted_progress = []
        for course_progress in progress_data:
            formatted_progress.append({
                'course_id': course_progress['course_id'],
                'course_title': course_progress['course_title'],
                'progress_percentage': float(course_progress['progress_percentage'] or 0),
                'enrollment_date': course_progress['enrollment_date'].isoformat() if course_progress['enrollment_date'] else None,
                'lessons_completed': course_progress['lessons_completed'] or 0,
                'total_lessons': course_progress['total_lessons'] or 0,
                'time_spent_hours': round((course_progress['total_time_spent'] or 0) / 60, 1)
            })
        
        return jsonify({
            'student_id': student_id,
            'courses': formatted_progress,
            'total_courses': len(formatted_progress),
            'average_progress': sum(p['progress_percentage'] for p in formatted_progress) / len(formatted_progress) if formatted_progress else 0
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500