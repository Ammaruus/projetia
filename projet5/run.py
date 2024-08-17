import numpy as np
import matplotlib.pyplot as plt


def load_mnist_data(file_path) -> tuple: # tuple contenant l'image et le one-hot
    print("file_path : ", file_path)
    data = np.loadtxt(file_path, delimiter=',', skiprows=1)

    labels = data[:, 0].astype(int)
    num_classes = 10
    labels_one_hot = np.eye(num_classes)[labels] # Qu'est-ce qu'un encodage one-hot ?
    images = data[:, 1:] / 255.0
    return images, labels_one_hot


class NeuralNetwork:
    def __init__(self,
                  input_size: int,
                  hidden_size: int,
                  output_size: int,
                  hidden_activation_function: str ="tanh",
                  output_activation_function: str = "softmax"):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        self.hidden_activation_function = hidden_activation_function
        self.output_activation_function = output_activation_function

    def set_loss_factor_exponent(self):
        """
        Calculates the loss factor exponent.
        """
        batch_rate_log = np.log10(self.batch_rate)
        hidden_size_log = np.log10(self.hidden_size)
        self.loss_factor_exponent = (abs(batch_rate_log) + 1) / (hidden_size_log) + 1



    def tanh(self, x : np.ndarray) -> np.ndarray: 
        return np.tanh(x)

    def rectified_linear_unit(self, x: np.ndarray) -> np.ndarray:
        """
        ReLU activation function
        """
        return np.maximum(0, x)
    
    def rectified_linear_unit_leaky(self, x):
        """
        LeakyReLU activation function
        """
        return np.maximum(0.01 * x, x)

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
    
    def get_output_error(
        self,
        y_true: np.ndarray,  # true labels
        y_pred: np.ndarray,  # predicted labels
    ) -> np.ndarray:
        """
        Calculates the output error.
        Returns:
        ndarray: Output error.
        """
        # Output layer error is the difference between predicted and true values
        # y_one_hot.shape[0] is the number of rows in y_one_hot
        # output_error = np.substract(y_one_hot, self.model_output) / y_one_hot.shape[0]
        output_error = np.divide(np.subtract(y_true, y_pred), y_true.shape[0])
        return output_error

    def forward(self, X: np.ndarray) -> np.ndarray:  # input data
        """
        Performs the forward pass
        of the neural network.
        """
        # Input to Hidden Layer
        # Weighted sum of inputs
        self.hidden_input = np.dot(X, self.weights_input_hidden)
        # Apply input-hidden activation function
        if self.hidden_activation_function == "tanh":
            self.hidden_output = self.tanh(self.hidden_input)
        elif self.hidden_activation_function == "ReLU":
            self.hidden_output = self.rectified_linear_unit(self.hidden_input)
        elif self.hidden_activation_function == "LeakyReLU":
            self.hidden_output = self.rectified_linear_unit_leaky(self.hidden_input)
        else:
            raise ValueError(
                f"Unknown activation function: {self.hidden_activation_function}"
            )
        # Weighted sum of hidden outputs
        self.output_input = np.dot(self.hidden_output, self.weights_hidden_output)
        # Apply hidden-output activation function
        if self.output_activation_function == "softmax":
            self.model_output = self.softmax(self.output_input)
        return self.model_output

    def backward(
        self,
        X: np.ndarray,  # input data
        y_one_hot: np.ndarray,  # one-hot encoded labels
        learning_rate=0.01,
    ) -> float:
        """
        Performs the backward pass
        (backpropagation) and
        updates the weights.
        Returns:
        float: Computed loss.
        """
        # Calculate the Mean Squared Error loss
        self.loss = self.mse_loss(y_one_hot, self.model_output)
        # Rétropropagation
        output_error = self.get_output_error(y_one_hot, self.model_output)
        # Calculate hidden layer error (backpropagated error)
        # hidden_output_ =  1 - self.hidden_output**2
        # L’erreur de la couche intermédiaire est donnée par 𝑒ℎ = (𝑒_𝑜 × 𝑊_𝑜^𝑇 ) ∗ 𝑦ℎ ∗ (1 − 𝑦ℎ)
        weights_hidden_output_transpose = np.transpose(self.weights_hidden_output)

        hidden_error = (
            np.dot(output_error, weights_hidden_output_transpose)
            * self.hidden_output
            * (1 - self.hidden_output)
        )
        # gradient is the derivative of the loss function (MSE) and serves to update the weights

        # Calculating gradient for weights between input and hidden layer
        x_transpose = np.transpose(X)
        d_weights_input_hidden = np.dot(x_transpose, hidden_error)
        # Calculate gradient for weights between hidden and output layer
        d_weights_hidden_output = np.dot(self.hidden_output.T, output_error)
        # Update the weights with the derivatives (gradient descent)
        # 𝑊ℎ = 𝑊ℎ − 𝜇(𝑥𝑇 × 𝑒ℎ)
        self.weights_input_hidden -= learning_rate * d_weights_input_hidden
        # 𝑊𝑜 = 𝑊𝑜 − 𝜇(𝑦𝑜 × 𝑒𝑜)
        self.weights_hidden_output -= learning_rate * d_weights_hidden_output
        self.fit = (
            np.sum(np.argmax(y_one_hot, axis=1) == np.argmax(self.model_output, axis=1))
            / y_one_hot.shape[0]
        )
        return self.loss


    def train(self,
               X:np.ndarray,
                y_one_hot: np.ndarray,
                epochs=10,
                learning_rate=0.03,
                batch_rate=0.0005):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_rate = batch_rate
        self.set_loss_factor_exponent()
        best_fit = 0.0
        best_loss = float("inf")
        for epoch in range(epochs):
            permutation = np.random.permutation(X.shape[0])
            x_shuffled = X[permutation]
            y_shuffled = y_one_hot[permutation]
            loss = 0.0
            batch_size = int(batch_rate * X.shape[0])
            best_batch_loss = float("inf")
            for i in range(0, X.shape[0], batch_size):
                x_batch = x_shuffled[i : i + batch_size]
                y_batch = y_shuffled[i : i + batch_size]

                # Forward and backward pass for the batch
                self.forward(x_batch)
                batch_loss = self.backward(x_batch, y_batch, learning_rate)
                # print(f"batch_loss {ii}: {batch_loss}")
                if batch_loss < best_batch_loss:
                    best_batch_loss = batch_loss

                loss += batch_loss / len(range(0, X.shape[0], batch_size))
            
            
                if epoch % 10 == 0:
                    print(f"Epoch {epoch}, Loss: {loss}")
                if best_fit < self.fit:
                    best_fit = self.fit
                if loss < best_loss:
                    best_loss = loss
        print(f"Best fit: {best_fit}, Best loss: {best_loss}")

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
    hidden_size = 784
    output_size = 10
    e = 50 # Nombre d'époches d'entraînement
    mu = 0.03  # Taux d'apprentissage
    batch_rates = 0.0005
    loss_factor_exponents = 3

    nn = NeuralNetwork(input_size, hidden_size, output_size)
    nn.train(X, y, epochs=e, learning_rate=mu)

    X_test, y_test = load_mnist_data('test.csv')
    nn.confusion_matrix(X_test, y_test)
    nn.visualize_prediction(X_test, y_test, 100)
