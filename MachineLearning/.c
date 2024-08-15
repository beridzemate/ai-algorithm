#include <stdio.h>


double mean(double arr[], int size) {
    double sum = 0.0;
    for (int i = 0; i < size; i++) {
        sum += arr[i];
    }
    return sum / size;
}


double variance(double arr[], int size, double mean) {
    double sum = 0.0;
    for (int i = 0; i < size; i++) {
        sum += (arr[i] - mean) * (arr[i] - mean);
    }
    return sum / size;
}


double covariance(double arr1[], double arr2[], int size, double mean1, double mean2) {
    double sum = 0.0;
    for (int i = 0; i < size; i++) {
        sum += (arr1[i] - mean1) * (arr2[i] - mean2);
    }
    return sum / size;
}

void linear_regression(double x[], double y[], int size, double *slope, double *intercept) {
    double mean_x = mean(x, size);
    double mean_y = mean(y, size);

    double var_x = variance(x, size, mean_x);
    double cov_xy = covariance(x, y, size, mean_x, mean_y);

    *slope = cov_xy / var_x;
    *intercept = mean_y - (*slope * mean_x);
}

int main() {
 
    double x[] = {1, 2, 3, 4, 5};
    double y[] = {2, 4, 5, 4, 5};
    int size = 5;

    double slope, intercept;
    linear_regression(x, y, size, &slope, &intercept);

    printf("Linear Regression Model: y = %.2fx + %.2f\n", slope, intercept);

    return 0;
}
