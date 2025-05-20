import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)
m_samples = 1_000_000

# Генерация признаков
age = np.random.normal(35, 10, m_samples).astype(np.float32)
income = np.random.normal(60_000, 20_000, m_samples).astype(np.float32)
credit_rating = np.random.normal(1000, 100, m_samples).astype(np.float32)
credit_count = np.random.poisson(2, m_samples).astype(np.int8)
employment_years = np.random.randint(0, 40, m_samples).astype(np.int8)
children_count = np.abs(np.round(np.random.normal(1.5, 1.5, m_samples))).astype(np.int8)
marital_status = np.random.choice([0, 1, 2], size=m_samples, p=[0.5, 0.3, 0.2]).astype(np.int8)
has_mortgage = np.random.choice([0, 1], size=m_samples, p=[0.7, 0.3]).astype(np.int8)
education_level = np.random.choice([0, 1, 2], size=m_samples, p=[0.4, 0.4, 0.2]).astype(np.int8)
X_noise = np.random.randn(m_samples, 16).astype(np.float32)
X_main = np.column_stack([
    age, income, credit_rating, credit_count, marital_status,
    employment_years, has_mortgage, education_level, children_count
]).astype(np.float32)
numerical_index = [0, 1, 2, 3, 5, 8]
categorical_index = [4, 6, 7]
X_numerical = X_main[:, numerical_index]
X_categorical = X_main[:, categorical_index]
scaler = StandardScaler()
X_numerical_scaled = scaler.fit_transform(X_numerical).astype(np.float32)
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_categorical_encoded = encoder.fit_transform(X_categorical).astype(np.float32)
X_preprocessed = np.concatenate([
    X_numerical_scaled, X_categorical_encoded, X_noise
], axis=1).astype(np.float32)

weights = np.array([0.3, -0.9, -0.4, 0.5, 0.2, 0.1, 0.2, -0.1, 0.3, 0.4, -0.2, 0.1, 0.15, -0.05], dtype=np.float32)
to_real_y = X_preprocessed[:, :14] @ weights + np.random.normal(0, 0.05, size=m_samples).astype(np.float32)
y = (to_real_y > np.percentile(to_real_y, 70)).astype(np.int8)

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)

def sigmoid(z):
    z = np.clip(z, -500, 500) 
    pos_mask = z >= 0
    neg_mask = ~pos_mask
    result = np.zeros_like(z, dtype=np.float64)
    exp_neg = np.exp(-np.abs(z))
    result[pos_mask] = 1 / (1 + exp_neg[pos_mask])
    result[neg_mask] = exp_neg[neg_mask] / (1 + exp_neg[neg_mask])
    result = np.clip(result, 1e-15, 1-1e-15)
    return result

def binary_crossentropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1-1e-15)
    if np.any(y_pred <= 0) or np.any(y_pred >= 1):
        print("Error: y_pred out of bounds after clipping!")
        print("y_pred min/max:", np.min(y_pred), np.max(y_pred))
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    if np.isnan(loss):
        print("Error: Loss is NaN!")
        print("y_pred min/max:", np.min(y_pred), np.max(y_pred))
    return loss

def accuracy(y_true, y_pred):
    return np.mean(y_true == (y_pred > 0.5).astype(np.int8))

def train_logistic_regression_sgd(X_train, y_train, X_test, y_test, learning_rate=0.01, lambda_reg=0.001, epochs=20):
    n_features = X_train.shape[1]
    w = np.random.RandomState(42).normal(0, 0.01, n_features).astype(np.float32)
    b = 0.0
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    
    for epoch in range(epochs):
        indices = np.random.permutation(len(X_train))
        epoch_loss = 0.0
        count = 0
        for i in indices:
            x_i = X_train[i]
            y_i = y_train[i]
            z = np.dot(x_i, w) + b
            y_pred = sigmoid(z)
            error = y_pred - y_i
            dw = error * x_i + 2 * lambda_reg * w
            db = error
            w -= learning_rate * dw
            b -= learning_rate * db
            if i % 1000 == 0:
                epoch_loss += binary_crossentropy(np.array([y_i]), np.array([y_pred]))
                count += 1
        
        avg_train_loss = epoch_loss / count if count > 0 else 0
        train_pred = sigmoid(np.dot(X_train, w) + b)
        train_acc = accuracy(y_train, train_pred)
        
        z_test = np.dot(X_test, w) + b
        test_pred = sigmoid(z_test)
        avg_test_loss = binary_crossentropy(y_test, test_pred)
        test_acc = accuracy(y_test, test_pred)
        
        train_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}, "
              f"Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
    
    return w, b, train_losses, test_losses, train_accuracies, test_accuracies

# Обучение модели
w, b, train_losses, test_losses, train_accuracies, test_accuracies = train_logistic_regression_sgd(
    X_train, y_train, X_test, y_test, learning_rate=0.01, lambda_reg=0.001, epochs=20)

# Оценка на тестовых данных
z_test = np.dot(X_test, w) + b
y_pred = (sigmoid(z_test) > 0.5).astype(np.int8)
test_accuracy = np.mean(y_pred == y_test)
print(f"\nФинальная точность на тестовых данных: {test_accuracy:.4f}")

# Визуализация
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.plot(train_losses, color='royalblue', linewidth=2, marker='o', markersize=8, label='Train Loss')
plt.plot(test_losses, color='orange', linewidth=2, marker='s', markersize=8, label='Test Loss')
plt.title('Динамика функции потерь', fontsize=14)
plt.xlabel('Номер эпохи', fontsize=12)
plt.ylabel('Binary Cross-Entropy', fontsize=12)
plt.xticks(range(len(train_losses)), range(1, len(train_losses)+1))
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(train_accuracies, color='royalblue', linewidth=2, marker='o', markersize=8, label='Train Accuracy')
plt.plot(test_accuracies, color='orange', linewidth=2, marker='s', markersize=8, label='Test Accuracy')
plt.title('Динамика точности', fontsize=14)
plt.xlabel('Номер эпохи', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.xticks(range(len(train_accuracies)), range(1, len(train_accuracies)+1))
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

plt.subplot(2, 2, 3)
test_prob = sigmoid(z_test)
plt.hist(test_prob[y_test == 0], bins=50, alpha=0.5, color='red', label='Class 0')
plt.hist(test_prob[y_test == 1], bins=50, alpha=0.5, color='green', label='Class 1')
plt.title('Распределение предсказанных вероятностей', fontsize=14)
plt.xlabel('Предсказанная вероятность', fontsize=12)
plt.ylabel('Частота', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

plt.subplot(2, 2, 4)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Матрица ошибок', fontsize=14)
plt.xlabel('Предсказанный класс', fontsize=12)
plt.ylabel('Истинный класс', fontsize=12)

plt.tight_layout()
plt.savefig('training_visualizations.png')
plt.show()

fpr, tpr, _ = roc_curve(y_test, test_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='royalblue', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', alpha=0.7)
plt.title('ROC-кривая', fontsize=14)
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.savefig('roc_curve.png')
plt.show()