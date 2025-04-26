from flask import Flask, render_template, jsonify
import numpy as np
import math
import json

app = Flask(__name__)

class FireflyOptimization:
    def __init__(self, n_fireflies=20, max_iterations=50, alpha=0.5, beta_0=1.0, gamma=1.0, dimensions=2, 
                 bounds=[-10, 10], objective_function=None):
        """
        تهيئة خوارزمية تحسين سرب اليراعات
        
        المعاملات:
        n_fireflies: عدد اليراعات في السرب
        max_iterations: الحد الأقصى لعدد التكرارات
        alpha: معامل العشوائية (0-1)
        beta_0: جاذبية أساسية
        gamma: معامل امتصاص الضوء
        dimensions: عدد أبعاد مساحة البحث
        bounds: حدود مساحة البحث [الحد الأدنى، الحد الأقصى]
        objective_function: دالة الهدف المراد تحسينها
        """
        self.n_fireflies = n_fireflies
        self.max_iterations = max_iterations
        self.alpha = alpha
        self.beta_0 = beta_0
        self.gamma = gamma
        self.dimensions = dimensions
        self.bounds = bounds
        
        # تعريف دالة الهدف الافتراضية (دالة sphere) إذا لم يتم توفير دالة
        if objective_function is None:
            self.objective_function = lambda x: sum(x**2)
        else:
            self.objective_function = objective_function
            
        # تهيئة السرب
        self.fireflies = np.random.uniform(self.bounds[0], self.bounds[1], 
                                          (self.n_fireflies, self.dimensions))
        
        # حساب شدة الضوء الأولية (عكس قيمة دالة الهدف)
        self.light_intensity = np.array([self.objective_function(firefly) for firefly in self.fireflies])
        
        # حفظ أفضل حل محقق
        self.best_solution = None
        self.best_fitness = float('inf')
        
        # حفظ تاريخ التحسين والمواقع
        self.history = []
        self.save_current_state(0)
    
    def move_fireflies(self):
        """تحريك اليراعات وفقًا لقواعد الخوارزمية"""
        for i in range(self.n_fireflies):
            for j in range(self.n_fireflies):
                # إذا كان اليراع j أكثر سطوعًا من اليراع i
                if self.light_intensity[j] < self.light_intensity[i]:
                    # حساب المسافة بين اليراعين
                    r = np.linalg.norm(self.fireflies[i] - self.fireflies[j])
                    
                    # حساب معامل الجاذبية
                    beta = self.beta_0 * np.exp(-self.gamma * r**2)
                    
                    # تحديث موقع اليراع i
                    random_vector = self.alpha * (np.random.uniform(-0.5, 0.5, self.dimensions))
                    self.fireflies[i] += beta * (self.fireflies[j] - self.fireflies[i]) + random_vector
                    
                    # ضمان البقاء ضمن مجال البحث
                    self.fireflies[i] = np.clip(self.fireflies[i], self.bounds[0], self.bounds[1])
                    
                    # إعادة حساب شدة الضوء
                    self.light_intensity[i] = self.objective_function(self.fireflies[i])
    
    def save_current_state(self, iteration):
        """حفظ الحالة الحالية للسرب والبيانات ذات الصلة"""
        # العثور على أفضل حل في التكرار الحالي
        current_best_idx = np.argmin(self.light_intensity)
        current_best_solution = self.fireflies[current_best_idx].tolist()
        current_best_fitness = float(self.light_intensity[current_best_idx])
        
        # تحديث أفضل حل عام إذا لزم الأمر
        if current_best_fitness < self.best_fitness:
            self.best_fitness = current_best_fitness
            self.best_solution = current_best_solution
        
        # حفظ البيانات
        firefly_positions = self.fireflies.tolist()
        light_intensities = [-float(intensity) for intensity in self.light_intensity]  # سالب للتحويل إلى قيم موجبة للعرض
        
        iteration_data = {
            'iteration': iteration,
            'firefly_positions': firefly_positions,
            'light_intensities': light_intensities,
            'current_best': {
                'position': current_best_solution,
                'fitness': current_best_fitness
            },
            'global_best': {
                'position': self.best_solution,
                'fitness': self.best_fitness
            }
        }
        
        self.history.append(iteration_data)
    
    def optimize(self):
        """تنفيذ عملية التحسين"""
        for iteration in range(1, self.max_iterations + 1):
            self.move_fireflies()
            self.save_current_state(iteration)
        
        return self.best_solution, self.best_fitness, self.history

# تعريف بعض دوال الاختبار الشائعة للتحسين
def sphere_function(x):
    """دالة Sphere (x1^2 + x2^2 + ... + xn^2)"""
    return np.sum(x**2)

def rosenbrock_function(x):
    """دالة Rosenbrock"""
    return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

def rastrigin_function(x):
    """دالة Rastrigin"""
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

def ackley_function(x):
    """دالة Ackley"""
    a, b, c = 20, 0.2, 2 * np.pi
    sum1 = -a * np.exp(-b * np.sqrt(np.mean(x**2)))
    sum2 = -np.exp(np.mean(np.cos(c * x)))
    return sum1 + sum2 + a + np.exp(1)

@app.route('/')
def index():
    """الصفحة الرئيسية للتطبيق"""
    return render_template('index.html')

@app.route('/optimize/<function_name>')
def run_optimization(function_name):
    """تشغيل خوارزمية التحسين للدالة المحددة"""
    # تحديد دالة الهدف
    functions = {
        'sphere': sphere_function,
        'rosenbrock': rosenbrock_function,
        'rastrigin': rastrigin_function,
        'ackley': ackley_function
    }
    
    objective_function = functions.get(function_name, sphere_function)
    
    # تهيئة وتشغيل خوارزمية تحسين سرب اليراعات
    fso = FireflyOptimization(
        n_fireflies=20,
        max_iterations=30,
        alpha=0.5,
        beta_0=1.0,
        gamma=0.1,
        dimensions=2,
        bounds=[-5, 5],
        objective_function=objective_function
    )
    
    best_solution, best_fitness, history = fso.optimize()
    
    return jsonify({
        'best_solution': best_solution,
        'best_fitness': best_fitness,
        'history': history
    })

# قالب HTML للصفحة الرئيسية
@app.route('/templates/index.html')
def get_index_template():
    return """

    """

if __name__ == '__main__':
    app.run(debug=True)