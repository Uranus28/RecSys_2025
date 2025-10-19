import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

class ALS:
    def __init__(self, n_factors=10, alpha=40, regularization=0.1, iterations=15):
        """
        Инициализация ALS модели
        
        Parameters:
        - n_factors: количество латентных факторов
        - alpha: параметр уверенности для неявной обратной связи
        - regularization: коэффициент регуляризации (lambda)
        - iterations: количество итераций обучения
        """
        self.n_factors = n_factors
        self.alpha = alpha
        self.regularization = regularization
        self.iterations = iterations
        self.user_factors = None
        self.item_factors = None
        self.loss_history = []
    
    def fit(self, ratings):
        """
        Обучение ALS модели
        
        Parameters:
        - ratings: scipy sparse matrix в формате (users × items)
        """
        n_users, n_items = ratings.shape
        
        # Инициализация матриц факторов
        self.user_factors = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.item_factors = np.random.normal(0, 0.1, (n_items, self.n_factors))
        
        print("Начинаем обучение ALS...")
        
        for iteration in range(self.iterations):
            # Шаг 1: Фиксируем item factors, обновляем user factors
            for u in range(n_users):
                # Получаем индексы и значения оценок пользователя
                user_ratings = ratings[u].toarray().flatten()
                rated_items = np.where(user_ratings > 0)[0]
                
                if len(rated_items) > 0:
                    # Вычисляем уверенности для неявной обратной связи
                    confidence = 1 + self.alpha * user_ratings[rated_items]
                    
                    # Матрица item factors для оцененных items
                    Y = self.item_factors[rated_items]
                    
                    # Вычисляем матрицу A и вектор b
                    C = np.diag(confidence)
                    A = Y.T @ C @ Y + self.regularization * np.eye(self.n_factors)
                    b = Y.T @ confidence
                    
                    # Решаем линейную систему
                    self.user_factors[u] = np.linalg.solve(A, b)
            print("halfDone------------")
            # Шаг 2: Фиксируем user factors, обновляем item factors
            for i in range(n_items):
                # Получаем индексы и значения оценок для item
                item_ratings = ratings[:, i].toarray().flatten()
                rating_users = np.where(item_ratings > 0)[0]
                
                if len(rating_users) > 0:
                    # Вычисляем уверенности
                    confidence = 1 + self.alpha * item_ratings[rating_users]
                    
                    # Матрица user factors для оценивших пользователей
                    X = self.user_factors[rating_users]
                    
                    # Вычисляем матрицу A и вектор b
                    C = np.diag(confidence)
                    A = X.T @ C @ X + self.regularization * np.eye(self.n_factors)
                    b = X.T @ confidence
                    
                    # Решаем линейную систему
                    self.item_factors[i] = np.linalg.solve(A, b)
            print("calcLoss")
            # Вычисляем loss (пока отключил)
            # loss = self._compute_loss(ratings)
            # self.loss_history.append(loss)
            # print(f"Итерация {iteration + 1}/{self.iterations}, Loss: {loss:.4f}")
            print(f"Итерация {iteration + 1}/{self.iterations}")
    
    def _compute_loss(self, ratings):
        """Вычисление функции потерь"""
        loss = 0
        n_users, n_items = ratings.shape
        
        for u in range(n_users):
            if u % 1000 == 0:
              print(u) 
            for i in range(n_items):
                if ratings[u, i] > 0:
                    prediction = self.user_factors[u] @ self.item_factors[i]
                    confidence = 1 + self.alpha * ratings[u, i]
                    error = confidence - prediction
                    loss += error ** 2
        
        # Добавляем регуляризацию
        reg_loss = self.regularization * (
            np.sum(self.user_factors ** 2) + np.sum(self.item_factors ** 2)
        )
        
        return loss + reg_loss
    
    def predict(self, user_id, item_id):
        """Предсказание оценки"""
        return self.user_factors[user_id] @ self.item_factors[item_id]
    
    def recommend(self, user_id, n_recommendations=10):
        """Рекомендации для пользователя"""
        user_vector = self.user_factors[user_id]
        scores = user_vector @ self.item_factors.T
        top_items = np.argsort(scores)[::-1][:n_recommendations]
        return top_items, scores[top_items]
    
    def plot_loss(self):
        """Визуализация истории потерь"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.loss_history, marker='o')
        plt.title('История обучения ALS')
        plt.xlabel('Итерация')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.show()

    


     


