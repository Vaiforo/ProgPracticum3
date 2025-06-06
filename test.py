# Линейная регрессия с L2-регуляризацией (Ridge) и градиентным спуском
class RidgeRegressionGD:
    def __init__(self, alpha=0.0001, lr=0.001, n_iter=1000):
        self.alpha = alpha
        self.lr = lr
        self.n_iter = n_iter
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iter):
            y_pred = np.dot(X, self.weights) + self.bias
            error = y_pred - y

            dw = (2 / n_samples) * (np.dot(X.T, error) + self.alpha * self.weights)
            db = (2 / n_samples) * np.sum(error)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias


# Обучение
model = RidgeRegressionGD(alpha=0.0001, lr=0.00001, n_iter=50000)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# Метрики
sklearn_metrick_printer(y_test, y_pred)
