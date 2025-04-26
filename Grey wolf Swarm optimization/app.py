from flask import Flask, render_template, jsonify
import numpy as np
import random
import math

app = Flask(__name__)

class GreyWolfOptimizer:
    def __init__(self, objective_function, dim=2, pop_size=30, max_iter=100, search_domain=(-10, 10)):
        # Initialize parameters
        self.objective_function = objective_function
        self.dim = dim
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.search_domain = search_domain
        
        # Initialize population
        self.population = np.random.uniform(
            low=search_domain[0], 
            high=search_domain[1], 
            size=(pop_size, dim)
        )
        
        # Initialize alpha, beta, and delta positions
        self.alpha_pos = np.zeros(dim)
        self.alpha_score = float('inf')
        
        self.beta_pos = np.zeros(dim)
        self.beta_score = float('inf')
        
        self.delta_pos = np.zeros(dim)
        self.delta_score = float('inf')
        
        # History for animation
        self.history = []
        self.best_scores_history = []
        
    def optimize(self):
        for iteration in range(self.max_iter):
            # Save current state to history
            current_state = {
                'iteration': iteration,
                'wolves': self.population.copy().tolist(),
                'alpha': self.alpha_pos.tolist(),
                'beta': self.beta_pos.tolist(),
                'delta': self.delta_pos.tolist()
            }
            self.history.append(current_state)
            
            # Update a, decreases linearly from 2 to 0
            a = 2 - iteration * (2 / self.max_iter)
            
            # Update each wolf position
            for i in range(self.pop_size):
                # Calculate fitness
                fitness = self.objective_function(self.population[i])
                
                # Update alpha, beta, delta
                if fitness < self.alpha_score:
                    self.delta_score = self.beta_score
                    self.delta_pos = self.beta_pos.copy()
                    
                    self.beta_score = self.alpha_score
                    self.beta_pos = self.alpha_pos.copy()
                    
                    self.alpha_score = fitness
                    self.alpha_pos = self.population[i].copy()
                    
                elif fitness < self.beta_score:
                    self.delta_score = self.beta_score
                    self.delta_pos = self.beta_pos.copy()
                    
                    self.beta_score = fitness
                    self.beta_pos = self.population[i].copy()
                    
                elif fitness < self.delta_score:
                    self.delta_score = fitness
                    self.delta_pos = self.population[i].copy()
            
            self.best_scores_history.append(self.alpha_score)
            
            # Update positions
            for i in range(self.pop_size):
                # Position update formula for each wolf
                
                # Alpha
                r1 = np.random.random(self.dim)
                r2 = np.random.random(self.dim)
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                
                D_alpha = np.abs(C1 * self.alpha_pos - self.population[i])
                X1 = self.alpha_pos - A1 * D_alpha
                
                # Beta
                r1 = np.random.random(self.dim)
                r2 = np.random.random(self.dim)
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                
                D_beta = np.abs(C2 * self.beta_pos - self.population[i])
                X2 = self.beta_pos - A2 * D_beta
                
                # Delta
                r1 = np.random.random(self.dim)
                r2 = np.random.random(self.dim)
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                
                D_delta = np.abs(C3 * self.delta_pos - self.population[i])
                X3 = self.delta_pos - A3 * D_delta
                
                # Update position
                self.population[i] = (X1 + X2 + X3) / 3
                
                # Ensure wolves stay within bounds
                self.population[i] = np.clip(
                    self.population[i], 
                    self.search_domain[0], 
                    self.search_domain[1]
                )
        
        # Final state
        current_state = {
            'iteration': self.max_iter,
            'wolves': self.population.copy().tolist(),
            'alpha': self.alpha_pos.tolist(),
            'beta': self.beta_pos.tolist(),
            'delta': self.delta_pos.tolist()
        }
        self.history.append(current_state)
        
        return self.alpha_pos, self.alpha_score, self.history, self.best_scores_history

# Test functions for optimization
def sphere_function(x):
    """Sphere function - a simple optimization test function"""
    return np.sum(x**2)

def rastrigin_function(x):
    """Rastrigin function - a more complex multimodal test function"""
    A = 10
    n = len(x)
    return A * n + sum([(xi**2 - A * np.cos(2 * np.pi * xi)) for xi in x])

def himmelblau_function(x):
    """Himmelblau's function - a multimodal test function with 4 identical local minima"""
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/optimize/<function_name>')
def optimize(function_name):
    function_map = {
        'sphere': sphere_function,
        'rastrigin': rastrigin_function,
        'himmelblau': himmelblau_function
    }
    
    objective_function = function_map.get(function_name, sphere_function)
    
    # Initialize optimizer
    gwo = GreyWolfOptimizer(
        objective_function=objective_function,
        dim=2,
        pop_size=30,
        max_iter=50,
        search_domain=(-5, 5)
    )
    
    # Run optimization
    best_pos, best_score, history, best_scores = gwo.optimize()
    
    # Prepare response
    response = {
        'best_position': best_pos.tolist(),
        'best_score': float(best_score),
        'history': history,
        'best_scores': best_scores
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)