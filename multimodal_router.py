from xray_analysis import analyze_xray
from wound_detection import analyze_wound
from medbot_rag import medical_qa  # This is safe now

def process_image(image_path, image_type):
    """Route image to appropriate analysis module"""
    if image_type == 'xray':
        analysis = analyze_xray(image_path)
        query = f"Explanation of X-ray finding: {analysis['condition']}"
        analysis['explanation'] = medical_qa(query)
        return analysis
    
    elif image_type == 'wound':
        analysis = analyze_wound(image_path)
        query = f"First aid for {analysis['wound_type']} wound with {analysis['severity']} severity"
        analysis['first_aid'] = medical_qa(query)
        return analysis
    
    else:
        raise ValueError(f"Unknown image type: {image_type}")