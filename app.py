from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from sklearn.cluster import KMeans
import json
import os

app = Flask(__name__)
CORS(app)

# Load floor plan metadata (make sure house_plans_metadata.json is in backend folder)
with open('house_plans_metadata.json', 'r') as f:
    FLOOR_PLANS_DATA = json.load(f)

class FloorPlanGenerator:
    def __init__(self):
        self.plans = FLOOR_PLANS_DATA['housePlans']
    
    def analyze_plot_image(self, image):
        """Analyze plot image using OpenCV"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            plot_area = cv2.contourArea(largest_contour)
            x, y, w, h = cv2.boundingRect(largest_contour)
            return {
                'plot_area': plot_area,
                'width': w,
                'height': h,
                'aspect_ratio': w / h if h > 0 else 1.0
            }
        return {'plot_area': 0, 'width': 0, 'height': 0, 'aspect_ratio': 1.0}
    
    def find_similar_plans(self, requirements, plot_analysis):
        target_rooms = requirements.get('total_rooms', 3)
        target_area = requirements.get('total_area', 35.0)
        scored_plans = []
        for plan in self.plans:
            score = 0
            room_diff = abs(plan['total_rooms'] - target_rooms)
            score += max(0, 10 - room_diff * 2)
            area_diff = abs(plan['total_area'] - target_area)
            score += max(0, 10 - area_diff)
            plan_room_types = {room['room_name'] for room in plan['rooms']}
            required_room_types = set(requirements.get('required_rooms', ['Living Room', 'Bedroom', 'Kitchen']))
            matching_rooms = len(plan_room_types.intersection(required_room_types))
            score += matching_rooms * 3
            scored_plans.append((plan, score))
        scored_plans.sort(key=lambda x: x[1], reverse=True)
        return [plan for plan, score in scored_plans[:5]]
    
    def generate_floor_plan(self, plot_analysis, requirements, similar_plans):
        base_plan = similar_plans[0] if similar_plans else self.plans[0]
        new_plan = {
            'generated': True,
            'base_plan': base_plan['filename'],
            'plot_dimensions': {
                'width': plot_analysis['width'],
                'height': plot_analysis['height'],
                'area': plot_analysis['plot_area']
            },
            'rooms': [],
            'total_rooms': requirements.get('total_rooms', 3),
            'total_area': requirements.get('total_area', base_plan['total_area']),
            'requirements': requirements
        }
        required_rooms = requirements.get('required_rooms', ['Living Room', 'Bedroom', 'Kitchen'])
        scale_factor = min(1.2, max(0.8, plot_analysis['aspect_ratio']))
        for room_name in required_rooms:
            base_room = None
            for room in base_plan['rooms']:
                if room['room_name'] == room_name:
                    base_room = room
                    break
            if not base_room:
                defaults = {
                    'Living Room': {'length': 500, 'width': 400},
                    'Bedroom': {'length': 350, 'width': 300},
                    'Kitchen': {'length': 300, 'width': 250},
                    'Bathroom': {'length': 200, 'width': 150},
                    'Dining Room': {'length': 350, 'width': 300}
                }
                base_room = defaults.get(room_name, {'length': 300, 'width': 250})
            new_room = {
                'room_name': room_name,
                'length': int(base_room['length'] * scale_factor),
                'width': int(base_room['width'] * scale_factor),
                'unit': 'cm'
            }
            new_plan['rooms'].append(new_room)
        return new_plan
    
    def create_floor_plan_svg(self, floor_plan, width=800, height=600):
        rooms = floor_plan['rooms']
        svg_content = f'<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n'
        svg_content += f'<svg width=\"{width}\" height=\"{height}\" xmlns=\"http://www.w3.org/2000/svg\">\n'
        svg_content += f'    <rect x=\"0\" y=\"0\" width=\"{width}\" height=\"{height}\" fill=\"#f8f9fa\" stroke=\"#dee2e6\" stroke-width=\"2\"/>\n'
        svg_content += f'    <text x=\"10\" y=\"25\" font-family=\"Arial\" font-size=\"16\" fill=\"#333\">Generated Floor Plan</text>\n'
        
        y_offset = 50
        colors = ['#e3f2fd', '#f3e5f5', '#e8f5e8', '#fff3e0', '#fce4ec']
        
        for i, room in enumerate(rooms):
            room_width = min(200, room['length'] // 2)
            room_height = min(120, room['width'] // 2)
            x = 50 + (i % 3) * 250
            y = y_offset + (i // 3) * 150
            color = colors[i % len(colors)]
            svg_content += f'    <rect x=\"{x}\" y=\"{y}\" width=\"{room_width}\" height=\"{room_height}\" fill=\"{color}\" stroke=\"#333\" stroke-width=\"1\"/>\n'
            svg_content += f'    <text x=\"{x + 10}\" y=\"{y + 20}\" font-family=\"Arial\" font-size=\"12\" fill=\"#333\">{room["room_name"]}</text>\n'
            svg_content += f'    <text x=\"{x + 10}\" y=\"{y + 35}\" font-family=\"Arial\" font-size=\"10\" fill=\"#666\">{room["length"]}x{room["width"]} cm</text>\n'
        
        svg_content += '</svg>'
        return svg_content

generator = FloorPlanGenerator()

@app.route('/')
def index():
    return jsonify({"message": "AI Floor Plan Designer API", "version": "1.0"})

@app.route('/analyze_and_generate', methods=['POST'])
def analyze_and_generate():
    try:
        if 'plot_image' not in request.files:
            return jsonify({"error": "No plot image uploaded"}), 400
        
        image_file = request.files['plot_image']
        requirements = {
            'total_rooms': int(request.form.get('total_rooms', 3)),
            'total_area': float(request.form.get('total_area', 35.0)),
            'required_rooms': request.form.get('required_rooms', 'Living Room,Bedroom,Kitchen').split(','),
            'vastu_compliant': request.form.get('vastu_compliant', 'false').lower() == 'true',
            'plot_orientation': request.form.get('plot_orientation', 'north')
        }
        image_bytes = image_file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({"error": "Invalid image format"}), 400
        plot_analysis = generator.analyze_plot_image(image)
        similar_plans = generator.find_similar_plans(requirements, plot_analysis)
        generated_plan = generator.generate_floor_plan(plot_analysis, requirements, similar_plans)
        svg_content = generator.create_floor_plan_svg(generated_plan)
        
        return jsonify({
            "success": True,
            "plot_analysis": plot_analysis,
            "generated_plan": generated_plan,
            "svg_plan": svg_content,
            "similar_plans_count": len(similar_plans)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get_sample_plans')
def get_sample_plans():
    sample_plans = generator.plans[:10]
    return jsonify({"plans": sample_plans})

@app.route('/search_plans')
def search_plans():
    rooms = request.args.get('rooms', type=int)
    area_min = request.args.get('area_min', type=float)
    area_max = request.args.get('area_max', type=float)
    filtered_plans = []
    for plan in generator.plans:
        if rooms and plan['total_rooms'] != rooms:
            continue
        if area_min and plan['total_area'] < area_min:
            continue
        if area_max and plan['total_area'] > area_max:
            continue
        filtered_plans.append(plan)
    return jsonify({"plans": filtered_plans[:20]})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
