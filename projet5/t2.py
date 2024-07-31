import numpy as np
import matplotlib.pyplot as plt

def load_mnist_data(file_path) -> tuple: # tuple contenant l'image et le one-hot

    print("file_path : ", file_path)
    data = np.loadtxt(file_path, delimiter=',', skiprows=1)
    
    #... 
    #labels = None
    labels = data[:, 0].astype(int)
    num_classes = 10
    labels_one_hot = np.eye(num_classes)[labels] # Qu'est-ce qu'un encodage one-hot ?
    
    #images = None
    #images = images / 255.0
    images = data[:, 1:] / 255.0

    #print("images : ", images)
    #print("labels one hot : ", labels_one_hot)

    return images, labels_one_hot

class NeuralNetwork:
    def __init__(self, input_size: int = 784, hidden_size: int = 784, output_size: int = 10):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)

    def tanh(self, x):
        return np.tanh(x)

    def softmax(self, x: np.ndarray, epsilon=1e-12) -> np.ndarray:
        # to avoid large exponentials and possible overflows:
        # Shift each row of x by subtracting its max value.
        x_shifted = x - np.max(x, axis=1, keepdims=True)
        # Calculate the softmax with the shifted values.
        exp_x_shifted = np.exp(x_shifted)
        sum_exp_x_shifted = np.sum(exp_x_shifted, axis=1, keepdims=True)

        # Calculate softmax and prevent division by zero.
        softmax_output = np.divide(
            exp_x_shifted, np.maximum(sum_exp_x_shifted, epsilon)
        )
        return softmax_output
        # exps = np.exp(x - np.max(x, axis=1, keepdims=True))  # Stabilité numérique
        # return exps / np.sum(exps, axis=1, keepdims=True)

    def mse_loss(self, y_true : np.ndarray, y_pred : np.ndarray) -> float:
        #return np.mean((y_true - y_pred) ** 2)
        return np.divide(np.sum((np.subtract(y_true, y_pred) ** 2)), y_true.shape[0])

    def forward(self, X) -> np.ndarray:
        self.hidden_input = np.dot(X, self.weights_input_hidden)
        self.hidden_output = self.tanh(self.hidden_input)
        self.output_input = np.dot(self.hidden_output, self.weights_hidden_output)
        self.model_output = self.softmax(self.output_input)

    def backward(self, X, y_one_hot, learning_rate=0.01) -> float:
        loss = self.mse_loss(y_one_hot, self.model_output)
        
        output_error = self.model_output - y_one_hot
        hidden_error = np.dot(output_error, self.weights_hidden_output.T) * (1 - self.hidden_output ** 2)

        self.weights_hidden_output -= learning_rate * np.dot(self.hidden_output.T, output_error)
        self.weights_input_hidden -= learning_rate * np.dot(X.T, hidden_error)

        return loss

    def train(self, X, y_one_hot, epochs=10, learning_rate=0.03):
        for epoch in range(epochs):
            self.forward(X)
            loss = self.backward(X, y_one_hot, learning_rate)
            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Loss: {loss}')

    def predict(self, X):
        self.forward(X)
        return np.argmax(self.model_output, axis=1)
    
    def visualize_prediction(self, X, y_true, index):
        input_data = X[index, :].reshape(28, 28)

        predicted_label = self.predict(X[index:index + 1])[0]

        plt.imshow(input_data, cmap='gray')
        plt.title(f"Prediction: {predicted_label}, True Label: {np.argmax(y_true[index])}")
        plt.show()
        
    def confusion_matrix(self, X, y_true):
        y_pred = self.predict(X)

        num_classes = 10
        cm = np.zeros((num_classes, num_classes), dtype=int)

        for true_label, pred_label in zip(np.argmax(y_true, axis=1), y_pred):
            cm[true_label, pred_label] += 1

        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(num_classes)
        plt.xticks(tick_marks, range(num_classes))
        plt.yticks(tick_marks, range(num_classes))
        plt.xlabel('Predicted')
        plt.ylabel('True')

        for i in range(num_classes):
            for j in range(num_classes):
                plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='black')

        plt.show()

if __name__ == "__main__":
    X, y = load_mnist_data('train.csv')
    
    # Vérification des données chargées
    if X is None or y is None:
        raise ValueError("Les données n'ont pas été correctement chargées.")
    
    input_size = X.shape[1]
    hidden_size = 64  # Par exemple, vous pouvez ajuster cette valeur
    output_size = 10
    e = 30  # Nombre d'époques d'entraînement
    mu = 0.01  # Taux d'apprentissage

    nn = NeuralNetwork(input_size, hidden_size, output_size)
    nn.train(X, y, epochs=e, learning_rate=mu)

    X_test, y_test = load_mnist_data('test.csv')
    nn.confusion_matrix(X_test, y_test)
    nn.visualize_prediction(X_test, y_test, 10)
