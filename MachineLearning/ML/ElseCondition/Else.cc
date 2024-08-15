#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

// Function to compute Mean Squared Error
double mean_squared_error(const std::vector<double>& y_true, const std::vector<double>& y_pred) {
    double sum = 0.0;
    size_t n = y_true.size();
    for (size_t i = 0; i < n; ++i) {
        double error = y_true[i] - y_pred[i];
        sum += error * error;
    }
    return sum / n;
}

// Linear Regression class
class LinearRegression {
private:
    double slope;
    double intercept;

public:
    LinearRegression() : slope(0), intercept(0) {}

    // Function to fit the model
    void fit(const std::vector<double>& x, const std::vector<double>& y) {
        if (x.size() != y.size() || x.empty()) {
            std::cerr << "Error: Mismatch in input data size or empty data." << std::endl;
            return;
        }

        double x_mean = 0, y_mean = 0, numerator = 0, denominator = 0;

        size_t n = x.size();
        for (size_t i = 0; i < n; ++i) {
            x_mean += x[i];
            y_mean += y[i];
        }
        x_mean /= n;
        y_mean /= n;

        for (size_t i = 0; i < n; ++i) {
            numerator += (x[i] - x_mean) * (y[i] - y_mean);
            denominator += (x[i] - x_mean) * (x[i] - x_mean);
        }

        if (denominator == 0) {
            std::cerr << "Error: Denominator is zero, cannot compute slope." << std::endl;
            return;
        }

        slope = numerator / denominator;
        intercept = y_mean - slope * x_mean;
    }

    // Function to predict output
    double predict(double x) const {
        return slope * x + intercept;
    }

    // Function to print model parameters
    void print() const {
        std::cout << "Linear Regression - Slope: " << slope << ", Intercept: " << intercept << std::endl;
    }
};

// Euclidean distance function
double euclidean_distance(const std::vector<double>& a, const std::vector<double>& b) {
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

// K-Nearest Neighbors class
class KNearestNeighbors {
private:
    std::vector<std::vector<double>> X_train;
    std::vector<int> y_train;
    int k;

public:
    KNearestNeighbors(int k) : k(k) {}

    // Function to fit the model
    void fit(const std::vector<std::vector<double>>& X, const std::vector<int>& y) {
        if (X.size() != y.size() || X.empty() || y.empty()) {
            std::cerr << "Error: Mismatch in training data size or empty data." << std::endl;
            return;
        }
        X_train = X;
        y_train = y;
    }

    // Function to predict the class of a new sample
    int predict(const std::vector<double>& sample) {
        if (X_train.empty() || y_train.empty()) {
            std::cerr << "Error: Model has not been trained." << std::endl;
            return -1; // Indicate an error
        }

        std::vector<std::pair<double, int>> distances;
        for (size_t i = 0; i < X_train.size(); ++i) {
            double dist = euclidean_distance(sample, X_train[i]);
            distances.push_back(std::make_pair(dist, y_train[i]));
        }

        std::sort(distances.begin(), distances.end());

        std::vector<int> votes(2, 0); // Assuming binary classification (0 and 1)
        for (int i = 0; i < k; ++i) {
            if (i >= distances.size()) break; // Avoid out-of-bounds access
            int label = distances[i].second;
            votes[label]++;
        }

        if (votes[0] == votes[1]) {
            std::cerr << "Warning: Tie detected in voting, defaulting to class 0." << std::endl;
            return 0;
        }

        return (votes[0] > votes[1]) ? 0 : 1;
    }
};

int main() {
    // Linear Regression Example
    std::vector<double> X_lr = {1, 2, 3, 4, 5}; // Independent variable
    std::vector<double> y_lr = {2, 4, 6, 8, 10}; // Dependent variable

    LinearRegression model_lr;
    model_lr.fit(X_lr, y_lr);

    std::vector<double> y_pred_lr;
    for (double x : X_lr) {
        y_pred_lr.push_back(model_lr.predict(x));
    }

    double mse = mean_squared_error(y_lr, y_pred_lr);
    std::cout << "Linear Regression - Mean Squared Error: " << mse << std::endl;
    model_lr.print();

    // K-Nearest Neighbors Example
    std::vector<std::vector<double>> X_knn = {{1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 6}};
    std::vector<int> y_knn = {0, 0, 1, 1, 1}; // Binary classes (0 and 1)

    KNearestNeighbors model_knn(3); // 3 neighbors
    model_knn.fit(X_knn, y_knn);

    std::vector<double> sample = {3, 4};
    int prediction = model_knn.predict(sample);
    if (prediction != -1) { // Check for error
        std::cout << "K-Nearest Neighbors - Prediction for sample {3, 4}: " << prediction << std::endl;
    } else {
        std::cerr << "Error: Prediction failed." << std::endl;
    }

    return 0;
}
