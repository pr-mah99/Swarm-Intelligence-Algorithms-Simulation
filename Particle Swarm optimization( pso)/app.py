import numpy as np
import json
from flask import Flask, render_template, request, jsonify
import random
import math

app = Flask(__name__)

# تعريف معلمات PSO
class PSOParams:
    def __init__(self):
        self.num_particles = 20
        self.max_iterations = 100
        self.dimensions = 2
        self.c1 = 2.0  # معامل الاقتناع الشخصي
        self.c2 = 2.0  # معامل الاقتناع الاجتماعي
        self.w = 0.7   # معامل الوزن الذاتي
        self.bounds = [-5, 5]  # نطاق البحث

# دالة الهدف (سنستخدم دالة Rosenbrock كمثال)
def objective_function(x):
    if len(x.shape) == 1:
        x = x.reshape(1, -1)
    
    result = np.zeros(x.shape[0])
    
    for i in range(x.shape[1]-1):
        result += 100 * (x[:, i+1] - x[:, i]**2)**2 + (1 - x[:, i])**2
        
    return result

# تهيئة الجسيمات
def initialize_particles(params):
    particles = []
    
    for i in range(params.num_particles):
        # موقع عشوائي في نطاق البحث
        position = np.random.uniform(params.bounds[0], params.bounds[1], params.dimensions)
        
        # سرعة عشوائية
        velocity = np.random.uniform(-1, 1, params.dimensions)
        
        # أفضل موقع شخصي (يبدأ بالموقع الحالي)
        personal_best_position = position.copy()
        personal_best_value = objective_function(position)[0]
        
        particles.append({
            'id': i,
            'position': position.tolist(),
            'velocity': velocity.tolist(),
            'personal_best_position': personal_best_position.tolist(),
            'personal_best_value': float(personal_best_value),
            'current_value': float(personal_best_value)
        })
    
    return particles

# تحديث الجسيمات
def update_particles(particles, global_best_position, params):
    for particle in particles:
        position = np.array(particle['position'])
        velocity = np.array(particle['velocity'])
        personal_best_position = np.array(particle['personal_best_position'])
        
        # عوامل عشوائية
        r1 = np.random.random(params.dimensions)
        r2 = np.random.random(params.dimensions)
        
        # تحديث السرعة
        cognitive_component = params.c1 * r1 * (personal_best_position - position)
        social_component = params.c2 * r2 * (global_best_position - position)
        
        new_velocity = params.w * velocity + cognitive_component + social_component
        
        # تحديث الموقع
        new_position = position + new_velocity
        
        # تقييد الموقع للبقاء ضمن النطاق
        new_position = np.clip(new_position, params.bounds[0], params.bounds[1])
        
        # تقييم الموقع الجديد
        new_value = objective_function(new_position)[0]
        
        # تحديث أفضل موقع شخصي إذا كان الموقع الجديد أفضل
        if new_value < particle['personal_best_value']:
            particle['personal_best_position'] = new_position.tolist()
            particle['personal_best_value'] = float(new_value)
        
        # تحديث الموقع والسرعة والقيمة الحالية
        particle['position'] = new_position.tolist()
        particle['velocity'] = new_velocity.tolist()
        particle['current_value'] = float(new_value)
    
    return particles

# تشغيل PSO
def run_pso(params):
    # تهيئة الجسيمات
    particles = initialize_particles(params)
    
    # تهيئة أفضل موقع عالمي
    global_best_position = None
    global_best_value = float('inf')
    
    # العثور على أفضل موقع عالمي أولي
    for particle in particles:
        if particle['personal_best_value'] < global_best_value:
            global_best_value = particle['personal_best_value']
            global_best_position = np.array(particle['personal_best_position'])
    
    # سجل التاريخ
    history = []
    
    # تسجيل الحالة الأولية
    iteration_data = {
        'iteration': 0,
        'particles': particles.copy(),
        'global_best_position': global_best_position.tolist(),
        'global_best_value': float(global_best_value)
    }
    history.append(iteration_data)
    
    # تكرار الخوارزمية
    for iteration in range(1, params.max_iterations + 1):
        # تحديث الجسيمات
        particles = update_particles(particles, global_best_position, params)
        
        # تحديث أفضل موقع عالمي
        for particle in particles:
            if particle['personal_best_value'] < global_best_value:
                global_best_value = particle['personal_best_value']
                global_best_position = np.array(particle['personal_best_position'])
        
        # تسجيل البيانات لهذا التكرار
        iteration_data = {
            'iteration': iteration,
            'particles': particles.copy(),
            'global_best_position': global_best_position.tolist(),
            'global_best_value': float(global_best_value)
        }
        history.append(iteration_data)
    
    return history

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run_simulation', methods=['POST'])
def run_simulation():
    data = request.get_json()
    
    params = PSOParams()
    if data:
        if 'num_particles' in data:
            params.num_particles = int(data['num_particles'])
        if 'max_iterations' in data:
            params.max_iterations = int(data['max_iterations'])
        if 'dimensions' in data:
            params.dimensions = int(data['dimensions'])
        if 'c1' in data:
            params.c1 = float(data['c1'])
        if 'c2' in data:
            params.c2 = float(data['c2'])
        if 'w' in data:
            params.w = float(data['w'])
    
    history = run_pso(params)
    
    return jsonify(history)

if __name__ == "__main__":
    app.run(debug=True)