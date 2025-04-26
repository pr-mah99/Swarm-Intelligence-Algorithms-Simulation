from flask import Flask, render_template, jsonify
import numpy as np
import random
import math
import json

app = Flask(__name__)

# معلمات الخوارزمية
class BeeSwarmParams:
    def __init__(self):
        self.num_bees = 50         # عدد النحل الإجمالي
        self.num_scout_bees = 10   # عدد نحل الاستكشاف
        self.elite_sites = 5       # عدد المواقع النخبة
        self.best_sites = 8        # عدد أفضل المواقع (بما في ذلك النخبة)
        self.recruited_elite = 15  # عدد النحل المجند للمواقع النخبة
        self.recruited_best = 10   # عدد النحل المجند للمواقع الأفضل المتبقية
        self.shrink_factor = 0.8   # معامل تقليص نطاق البحث
        self.iterations = 100      # عدد التكرارات
        self.min_range = -5.0      # الحد الأدنى لنطاق البحث
        self.max_range = 5.0       # الحد الأقصى لنطاق البحث
        self.dimensions = 2        # أبعاد المشكلة

# دالة التقييم: دالة سفير (Sphere function) للاختبار
def objective_function(x):
    return np.sum(x**2)  # مجموع مربعات المكونات

# كائن BSO الرئيسي
class BeeSwarmOptimization:
    def __init__(self, params):
        self.params = params
        self.best_solution = None
        self.best_fitness = float('inf')
        self.all_solutions = []
        self.iteration_data = []
        
    def initialize(self):
        # توليد مواقع عشوائية أولية للنحل
        self.scout_bees = []
        for _ in range(self.params.num_scout_bees):
            position = self.params.min_range + np.random.random(self.params.dimensions) * (self.params.max_range - self.params.min_range)
            fitness = objective_function(position)
            self.scout_bees.append({
                'position': position.tolist(),
                'fitness': float(fitness)
            })
            
            # تحديث أفضل حل إذا وجد
            if fitness < self.best_fitness:
                self.best_fitness = fitness
                self.best_solution = position.tolist()
        
        # ترتيب النحل حسب اللياقة (من الأفضل إلى الأسوأ)
        self.scout_bees.sort(key=lambda x: x['fitness'])
        self.all_solutions = [bee['position'] for bee in self.scout_bees]
        
        # تسجيل بيانات التكرار الأول
        self.iteration_data.append({
            'iteration': 0,
            'solutions': self.all_solutions,
            'best_solution': self.best_solution,
            'best_fitness': self.best_fitness,
            'patches': self.get_patches()
        })
    
    def get_patches(self):
        patches = []
        
        # إنشاء رقع للمواقع النخبة
        for i in range(min(self.params.elite_sites, len(self.scout_bees))):
            patches.append({
                'center': self.scout_bees[i]['position'],
                'type': 'elite',
                'bees': self.params.recruited_elite
            })
        
        # إنشاء رقع للمواقع الأفضل المتبقية
        for i in range(self.params.elite_sites, min(self.params.best_sites, len(self.scout_bees))):
            patches.append({
                'center': self.scout_bees[i]['position'],
                'type': 'best',
                'bees': self.params.recruited_best
            })
        
        return patches
    
    def search_neighborhood(self, center, patch_size, num_bees):
        neighborhood_bees = []
        best_local_position = None
        best_local_fitness = float('inf')
        
        center_array = np.array(center)
        
        # إرسال النحل المجند للبحث حول المركز
        for _ in range(num_bees):
            # إنشاء موقع جديد ضمن الرقعة
            offset = (np.random.random(self.params.dimensions) * 2 - 1) * patch_size
            new_position = center_array + offset
            
            # التأكد من بقاء الموقع داخل حدود البحث
            new_position = np.clip(new_position, self.params.min_range, self.params.max_range)
            
            # حساب اللياقة
            fitness = objective_function(new_position)
            
            neighborhood_bees.append({
                'position': new_position.tolist(),
                'fitness': float(fitness)
            })
            
            # تحديث أفضل حل محلي
            if fitness < best_local_fitness:
                best_local_fitness = fitness
                best_local_position = new_position.tolist()
            
            # تحديث أفضل حل عام
            if fitness < self.best_fitness:
                self.best_fitness = fitness
                self.best_solution = new_position.tolist()
        
        return best_local_position, best_local_fitness, [bee['position'] for bee in neighborhood_bees]
    
    def run_iteration(self, iteration):
        patch_size = (self.params.max_range - self.params.min_range) * (self.params.shrink_factor ** iteration) / 10
        
        new_solutions = []
        all_local_solutions = []
        
        # البحث حول المواقع النخبة
        for i in range(min(self.params.elite_sites, len(self.scout_bees))):
            best_local, best_fitness, local_solutions = self.search_neighborhood(
                self.scout_bees[i]['position'],
                patch_size,
                self.params.recruited_elite
            )
            if best_local:
                new_solutions.append({
                    'position': best_local,
                    'fitness': best_fitness
                })
            all_local_solutions.extend(local_solutions)
        
        # البحث حول المواقع الأفضل المتبقية
        for i in range(self.params.elite_sites, min(self.params.best_sites, len(self.scout_bees))):
            best_local, best_fitness, local_solutions = self.search_neighborhood(
                self.scout_bees[i]['position'],
                patch_size,
                self.params.recruited_best
            )
            if best_local:
                new_solutions.append({
                    'position': best_local,
                    'fitness': best_fitness
                })
            all_local_solutions.extend(local_solutions)
        
        # استكشاف جديد لباقي النحل
        remaining_bees = self.params.num_scout_bees - self.params.best_sites
        for _ in range(remaining_bees):
            position = self.params.min_range + np.random.random(self.params.dimensions) * (self.params.max_range - self.params.min_range)
            fitness = objective_function(position)
            new_solutions.append({
                'position': position.tolist(),
                'fitness': float(fitness)
            })
            all_local_solutions.append(position.tolist())
            
            # تحديث أفضل حل إذا وجد
            if fitness < self.best_fitness:
                self.best_fitness = fitness
                self.best_solution = position.tolist()
        
        # تحديث قائمة النحل الاستكشافي
        self.scout_bees = new_solutions
        self.scout_bees.sort(key=lambda x: x['fitness'])
        
        # تحديث قائمة جميع الحلول لهذا التكرار
        self.all_solutions = [bee['position'] for bee in self.scout_bees]
        self.all_solutions.extend(all_local_solutions)
        
        # تسجيل بيانات هذا التكرار
        self.iteration_data.append({
            'iteration': iteration + 1,
            'solutions': self.all_solutions,
            'best_solution': self.best_solution,
            'best_fitness': self.best_fitness,
            'patches': self.get_patches()
        })
    
    def optimize(self):
        self.initialize()
        
        for i in range(self.params.iterations):
            self.run_iteration(i)
        
        return self.best_solution, self.best_fitness, self.iteration_data

# مسارات التطبيق
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/optimize')
def optimize():
    params = BeeSwarmParams()
    bso = BeeSwarmOptimization(params)
    best_solution, best_fitness, iteration_data = bso.optimize()
    
    result = {
        'best_solution': best_solution,
        'best_fitness': best_fitness,
        'iterations': iteration_data
    }
    
    return jsonify(result)

# إنشاء قالب HTML
@app.route('/templates/index.html')
def get_index_template():
    html = """

    """
    return html

if __name__ == '__main__':
    app.run(debug=True)