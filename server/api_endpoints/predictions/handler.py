from flask import jsonify
import random
from database.db import insert_prediction

def CreatePredictionHandler(request):
    """Handle AI/ML prediction requests"""
    try:
        payload = request.get_json(force=True)
        
        if not payload:
            return jsonify({'error': 'No input data provided'}), 400
        
        input_text = payload.get('text', '')
        model_name = payload.get('model', 'default_ai_model')
        student_id = payload.get('student_id')  # Optional
        
        if not input_text:
            return jsonify({'error': 'Text input is required'}), 400
        
        # Simulate AI prediction (replace with actual model inference)
        prediction_result = simulate_ai_prediction(input_text)
        
        # Store prediction in database
        try:
            prediction_id = insert_prediction(
                student_id, 
                input_text, 
                model_name,
                prediction_result['label'], 
                prediction_result['confidence']
            )
            prediction_result['prediction_id'] = prediction_id
        except Exception as db_error:
            print(f"Database error: {db_error}")
            # Continue without storing if DB fails
        
        return jsonify({
            'input': input_text,
            'model': model_name,
            'prediction': prediction_result,
            'timestamp': payload.get('timestamp'),
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def simulate_ai_prediction(input_text):
    """
    Simulate an AI prediction for demonstration purposes.
    In production, this would call your trained model.
    """
    
    # Example: Simple sentiment analysis simulation
    positive_keywords = ['good', 'excellent', 'amazing', 'love', 'great', 'fantastic', 'wonderful']
    negative_keywords = ['bad', 'terrible', 'hate', 'awful', 'horrible', 'disappointing']
    
    input_lower = input_text.lower()
    
    positive_score = sum(1 for word in positive_keywords if word in input_lower)
    negative_score = sum(1 for word in negative_keywords if word in input_lower)
    
    if positive_score > negative_score:
        label = 'positive'
        confidence = min(0.9, 0.6 + (positive_score * 0.1))
    elif negative_score > positive_score:
        label = 'negative' 
        confidence = min(0.9, 0.6 + (negative_score * 0.1))
    else:
        label = 'neutral'
        confidence = random.uniform(0.5, 0.8)
    
    return {
        'label': label,
        'confidence': round(confidence, 2),
        'explanation': f"Detected {positive_score} positive and {negative_score} negative indicators"
    }