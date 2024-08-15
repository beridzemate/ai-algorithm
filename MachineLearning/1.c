double sigmoid(double z) {
    return 1 / (1 + exp(-z));
}

void logistic_regression(double x[], double y[], int size, double *weights, double learning_rate, int epochs) {
    for (int i = 0; i < epochs; i++) {
        for (int j = 0; j < size; j++) {
            double predicted = sigmoid(weights[0] + weights[1] * x[j]);
            double error = y[j] - predicted;
            weights[0] += learning_rate * error * predicted * (1 - predicted);
            weights[1] += learning_rate * error * predicted * (1 - predicted) * x[j];
        }
    }
}

