#include <iostream>
#include <vector>
#include <cmath>

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

        slope = numerator / denominator;
        intercept = y_mean - slope * x_mean;
    }

    // Function to predict output
    double predict(double x) const {
        return slope * x + intercept;
    }

    // Function to print model parameters
    void print() const {
        std::cout << "Slope: " << slope << ", Intercept: " << intercept << std::endl;
    }
};

int main() {
    // Example data
    std::vector<double> X = {1, 2, 3, 4, 5}; // Independent variable
    std::vector<double> y = {2, 4, 6, 8, 10}; // Dependent variable

    // Create and train the model
    LinearRegression model;
    model.fit(X, y);

    // Make predictions
    std::vector<double> y_pred;
    for (double x : X) {
        y_pred.push_back(model.predict(x));
    }

    // Compute and display Mean Squared Error
    double mse = mean_squared_error(y, y_pred);
    std::cout << "Mean Squared Error: " << mse << std::endl;

    // Print model parameters
    model.print();

    return 0;
}
