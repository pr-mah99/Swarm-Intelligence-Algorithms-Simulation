# مكتبات مطلوبة
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import io
import base64
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

class AntColonyOptimization:
    def __init__(self, distances, n_ants, n_iterations, decay, alpha=1, beta=2):
        """
        تهيئة خوارزمية مستعمرة النمل
        
        المعلمات:
            distances: مصفوفة المسافات بين العقد
            n_ants: عدد النمل
            n_iterations: عدد التكرارات
            decay: معدل تبخر الفيرومون
            alpha: أهمية الفيرومون
            beta: أهمية المسافة
        """
        self.distances = distances
        self.pheromone = np.ones(distances.shape) / len(distances)
        self.all_inds = range(len(distances))
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        
        # للتتبع والتوضيح
        self.best_path = None
        self.best_path_length = float('inf')
        self.iteration_history = []
        self.ant_paths = []

    def run(self):
        """تشغيل الخوارزمية"""
        self.iteration_history = []
        
        for iteration in range(self.n_iterations):
            self.ant_paths = []
            all_paths = []
            all_path_lengths = []
            
            # كل نملة تبني مسارها
            for ant in range(self.n_ants):
                path = self.generate_path(0)  # نبدأ من العقدة 0
                path_length = self.calculate_path_length(path)
                all_paths.append(path)
                all_path_lengths.append(path_length)
                self.ant_paths.append({"path": path.copy(), "length": path_length})
                
                # تحديث أفضل مسار
                if path_length < self.best_path_length:
                    self.best_path = path
                    self.best_path_length = path_length
            
            # تحديث الفيرومون
            self.update_pheromone(all_paths, all_path_lengths)
            
            # تخزين معلومات هذا التكرار
            iteration_info = {
                "iteration": iteration + 1,
                "best_path": self.best_path.copy() if self.best_path is not None else None,
                "best_path_length": self.best_path_length,
                "pheromone_matrix": self.pheromone.copy(),
                "ant_paths": self.ant_paths.copy()
            }
            self.iteration_history.append(iteration_info)
            
            # تقليل الفيرومون (التبخر)
            self.pheromone = self.pheromone * self.decay
            
        return self.best_path, self.best_path_length, self.iteration_history

    def generate_path(self, start):
        """توليد مسار للنملة من العقدة البداية"""
        path = [start]
        visited = {start}
        
        while len(visited) < len(self.distances):
            current = path[-1]
            unvisited = set(self.all_inds) - visited
            
            # احتمالات الانتقال للعقد التالية
            probabilities = self.calculate_probabilities(current, unvisited)
            
            # اختيار العقدة التالية بناءً على الاحتمالات
            next_node = np.random.choice(list(unvisited), p=probabilities)
            path.append(next_node)
            visited.add(next_node)
            
        return np.array(path)

    def calculate_probabilities(self, current, unvisited):
        """حساب احتمالات الانتقال بناءً على كمية الفيرومون والمسافة"""
        unvisited_list = list(unvisited)
        
        # معادلة احتمالية اختيار المسار
        pheromone = np.array([self.pheromone[current][j] for j in unvisited_list])
        distance = np.array([1.0 / self.distances[current][j] for j in unvisited_list])
        
        # التأثير المرجح للفيرومون والمسافة
        probabilities = (pheromone ** self.alpha) * (distance ** self.beta)
        
        # تطبيع القيم لتكون احتمالات (المجموع = 1)
        probabilities = probabilities / probabilities.sum()
        
        return probabilities

    def calculate_path_length(self, path):
        """حساب الطول الكلي للمسار"""
        total_length = 0
        for i in range(len(path) - 1):
            total_length += self.distances[path[i]][path[i + 1]]
        
        # إضافة المسافة من العقدة الأخيرة إلى العقدة الأولى (للعودة إلى البداية)
        total_length += self.distances[path[-1]][path[0]]
        
        return total_length

    def update_pheromone(self, all_paths, all_path_lengths):
        """تحديث مستويات الفيرومون بعد اكتمال جميع مسارات النمل"""
        for path, path_length in zip(all_paths, all_path_lengths):
            # كمية الفيرومون تكون أكبر للمسارات الأقصر
            pheromone_amount = 1.0 / path_length
            
            # إضافة الفيرومون على كل حافة في المسار
            for i in range(len(path) - 1):
                self.pheromone[path[i]][path[i + 1]] += pheromone_amount
                self.pheromone[path[i + 1]][path[i]] += pheromone_amount
                
            # إضافة الفيرومون على الحافة من العقدة الأخيرة إلى العقدة الأولى
            self.pheromone[path[-1]][path[0]] += pheromone_amount
            self.pheromone[path[0]][path[-1]] += pheromone_amount

    def plot_graph(self, iteration_idx=None):
        """رسم الرسم البياني للمشكلة والحل"""
        G = nx.Graph()
        
        # إضافة العقد
        for i in range(len(self.distances)):
            G.add_node(i, pos=(np.random.random(), np.random.random()))
        
        # إضافة الحواف مع المسافات
        for i in range(len(self.distances)):
            for j in range(i + 1, len(self.distances)):
                G.add_edge(i, j, weight=self.distances[i][j], pheromone=self.pheromone[i][j])
        
        # تحديد موقع العقد
        pos = nx.spring_layout(G, seed=42)
        
        plt.figure(figsize=(12, 8))
        
        # رسم الحواف مع سمك يعتمد على الفيرومون
        edge_width = [G[u][v]['pheromone'] * 5 for u, v in G.edges()]
        nx.draw_networkx_edges(G, pos, width=edge_width, alpha=0.7, edge_color='g')
        
        # إظهار المسار الأفضل إذا تم تحديده
        if iteration_idx is not None and iteration_idx < len(self.iteration_history):
            current_best_path = self.iteration_history[iteration_idx]["best_path"]
            if current_best_path is not None:
                best_path_edges = [(current_best_path[i], current_best_path[i + 1]) for i in range(len(current_best_path) - 1)]
                best_path_edges.append((current_best_path[-1], current_best_path[0]))
                nx.draw_networkx_edges(G, pos, edgelist=best_path_edges, width=3, edge_color='r')
        
        # رسم العقد
        nx.draw_networkx_nodes(G, pos, node_size=500, node_color='b', alpha=0.8)
        
        # إضافة تسميات العقد
        nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')
        
        plt.title("Ant Colony: Repetition " + str(iteration_idx + 1 if iteration_idx is not None else 'النهائي'))
        plt.axis('off')
        
        # تحويل الرسم إلى صورة Base64 لعرضها في الويب
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        graph_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        return 'data:image/png;base64,{}'.format(graph_url)

# إنشاء مسألة TSP عشوائية
def create_random_tsp(n_cities):
    # توليد مواقع عشوائية للمدن
    points = np.random.rand(n_cities, 2)
    
    # حساب مصفوفة المسافات
    distances = np.zeros((n_cities, n_cities))
    for i in range(n_cities):
        for j in range(i + 1, n_cities):
            distances[i][j] = np.sqrt(np.sum((points[i] - points[j])**2))
            distances[j][i] = distances[i][j]  # المصفوفة متماثلة
    
    return distances, points

# إعداد تطبيق Flask وتعريف المسارات
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run_aco', methods=['POST'])
def run_aco():
    # استلام معلمات الخوارزمية من النموذج
    data = request.get_json()
    n_cities = int(data.get('n_cities', 10))
    n_ants = int(data.get('n_ants', 10))
    n_iterations = int(data.get('n_iterations', 20))
    decay = float(data.get('decay', 0.5))
    alpha = float(data.get('alpha', 1))
    beta = float(data.get('beta', 2))
    
    # توليد مسألة TSP عشوائية
    distances, points = create_random_tsp(n_cities)
    
    # إنشاء وتشغيل خوارزمية مستعمرة النمل
    aco = AntColonyOptimization(distances, n_ants, n_iterations, decay, alpha, beta)
    best_path, best_path_length, iteration_history = aco.run()
    
    # تحضير النتائج لكل تكرار
    results = []
    for i, iter_info in enumerate(iteration_history):
        graph_image = aco.plot_graph(i)
        results.append({
            'iteration': iter_info['iteration'],
            'best_path': iter_info['best_path'].tolist() if iter_info['best_path'] is not None else None,
            'best_path_length': float(iter_info['best_path_length']),
            'graph_image': graph_image
        })
    
    return jsonify({
        'results': results,
        'final_best_path': best_path.tolist() if best_path is not None else None,
        'final_best_path_length': float(best_path_length)
    })


if __name__ == '__main__':
    app.run(debug=True)