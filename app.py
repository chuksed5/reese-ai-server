"""
Reese AI - Python Fashion Stylist Server
Uses BLIP model for image captioning
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import base64
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

app = Flask(__name__)
CORS(app)  # Enable CORS for mobile app

# Load BLIP model (happens once at startup)
print("Loading BLIP model...")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
print("Model loaded successfully!")

@app.route('/')
def home():
    return jsonify({"status": "Reese AI Server is running!"})

@app.route('/api/stylist', methods=['POST', 'OPTIONS'])
def analyze_outfit():
    # Handle CORS preflight
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({"error": "No image provided"}), 400
        
        image_data = data['image']
        occasion = data.get('occasion', 'casual')
        
        # Remove data URL prefix if present
        if 'base64,' in image_data:
            image_data = image_data.split('base64,')[1]
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Generate caption using BLIP
        inputs = processor(image, return_tensors="pt")
        
        with torch.no_grad():
            out = model.generate(**inputs, max_length=50)
        
        description = processor.decode(out[0], skip_special_tokens=True)
        
        # Generate fashion advice
        advice = generate_advice(description, occasion)
        
        return jsonify({
            "response": advice,
            "imageDescription": description
        })
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

def generate_advice(description, occasion):
    """Generate fashion advice based on image description and occasion"""
    desc = description.lower()
    
    rules = {
        'job_interview': {
            'keywords': ['suit', 'tie', 'formal', 'blazer', 'shirt', 'dress'],
            'formal': [
                "Excellent professional look! You're interview-ready.",
                "Perfect business attire! Very appropriate.",
                "Great formal outfit! Make sure everything is pressed.",
            ],
            'casual': [
                "For an interview, try more formal attire like a suit.",
                "Consider a blazer or dress shirt for a professional look.",
                "This looks casual. Business formal would be better.",
            ]
        },
        'date': {
            'keywords': ['stylish', 'dress', 'nice', 'elegant'],
            'formal': [
                "You look fantastic! Perfect for a date.",
                "Great style! Very attractive outfit.",
                "Looking sharp! You'll make a great impression.",
            ],
            'casual': [
                "Nice look! Maybe add an accessory to elevate it.",
                "Consider dressing up a bit more to show effort.",
                "Looking good! A nice watch or shoes would complete this.",
            ]
        },
        'party': {
            'always': [
                "Fun outfit! You're ready for a great time!",
                "Perfect party look! Very festive.",
                "Great style! You'll fit right in.",
            ]
        },
        'casual': {
            'always': [
                "Relaxed and comfortable! Perfect for the day.",
                "Nice casual style! You look great.",
                "Comfortable and well put-together!",
            ]
        },
        'formal': {
            'keywords': ['suit', 'dress', 'formal', 'elegant', 'tie'],
            'formal': [
                "Elegant! Perfect formal attire.",
                "Excellent formal look! Very sophisticated.",
                "Stunning! Appropriate for any formal event.",
            ],
            'casual': [
                "For a formal event, try more elegant attire.",
                "This looks casual. Consider a suit or formal dress.",
                "More formal wear would be appropriate.",
            ]
        },
        'gym': {
            'keywords': ['athletic', 'sport', 'sneakers', 'shorts'],
            'formal': [
                "Perfect workout gear! Ready to train.",
                "Great athletic wear! Comfortable and functional.",
                "Excellent gym outfit!",
            ],
            'casual': [
                "For the gym, athletic wear would be better.",
                "Consider workout clothes for mobility.",
                "Gym attire would be more comfortable.",
            ]
        }
    }
    
    occasion_rule = rules.get(occasion, rules['casual'])
    
    # Check for always-good advice
    if 'always' in occasion_rule:
        import random
        return random.choice(occasion_rule['always'])
    
    # Check if outfit matches occasion
    is_formal = any(keyword in desc for keyword in occasion_rule['keywords'])
    advice_list = occasion_rule['formal'] if is_formal else occasion_rule['casual']
    
    import random
    return random.choice(advice_list)

if __name__ == '__main__':
    # Railway sets PORT env variable
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
