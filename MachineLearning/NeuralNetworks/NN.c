#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define INPUT_SIZE 2
#define HIDDEN_SIZE 4
#define OUTPUT_SIZE 1
#define TRAINING_SET_SIZE 4
#define LEARNING_RATE 0.1
#define EPOCHS 10000

// Activation function: Sigmoid
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Derivative of the sigmoid function
double sigmoid_derivative(double x) {
    return x * (1.0 - x);
}

// Initialize weights with random values
void initialize_weights(double *weights, int size) {
    srand(time(NULL));
    for (int i = 0; i < size; i++) {
        weights[i] = (double)rand() / RAND_MAX;
    }
}

// Define the neural network structure
typedef struct {
    double input[INPUT_SIZE];
    double hidden[HIDDEN_SIZE];
    double output[OUTPUT_SIZE];
    double weights_input_hidden[INPUT_SIZE * HIDDEN_SIZE];
    double weights_hidden_output[HIDDEN_SIZE * OUTPUT_SIZE];
    double bias_hidden[HIDDEN_SIZE];
    double bias_output[OUTPUT_SIZE];
} NeuralNetwork;

// Forward pass function
void forward_pass(NeuralNetwork *nn) {
    // Compute hidden layer values
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        nn->hidden[i] = nn->bias_hidden[i];
        for (int j = 0; j < INPUT_SIZE; j++) {
            nn->hidden[i] += nn->input[j] * nn->weights_input_hidden[j * HIDDEN_SIZE + i];
        }
        nn->hidden[i] = sigmoid(nn->hidden[i]);
    }

    // Compute output layer values
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        nn->output[i] = nn->bias_output[i];
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            nn->output[i] += nn->hidden[j] * nn->weights_hidden_output[j * OUTPUT_SIZE + i];
        }
        nn->output[i] = sigmoid(nn->output[i]);
    }
}

// Backpropagation function
void backpropagate(NeuralNetwork *nn, double *target) {
    // Compute output layer error
    double output_error[OUTPUT_SIZE];
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        output_error[i] = target[i] - nn->output[i];
    }

    // Update weights between hidden and output layers
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            double gradient = output_error[j] * sigmoid_derivative(nn->output[j]);
            nn->weights_hidden_output[i * OUTPUT_SIZE + j] += LEARNING_RATE * gradient * nn->hidden[i];
        }
    }

    // Update bias of the output layer
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        nn->bias_output[i] += LEARNING_RATE * output_error[i] * sigmoid_derivative(nn->output[i]);
    }

    // Compute hidden layer error
    double hidden_error[HIDDEN_SIZE];
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        hidden_error[i] = 0;
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            hidden_error[i] += output_error[j] * sigmoid_derivative(nn->output[j]) * nn->weights_hidden_output[i * OUTPUT_SIZE + j];
        }
    }

    // Update weights between input and hidden layers
    for (int i = 0; i < INPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            double gradient = hidden_error[j] * sigmoid_derivative(nn->hidden[j]);
            nn->weights_input_hidden[i * HIDDEN_SIZE + j] += LEARNING_RATE * gradient * nn->input[i];
        }
    }

    // Update bias of the hidden layer
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        nn->bias_hidden[i] += LEARNING_RATE * hidden_error[i] * sigmoid_derivative(nn->hidden[i]);
    }
}

// Training function
void train(NeuralNetwork *nn, double training_inputs[TRAINING_SET_SIZE][INPUT_SIZE], double training_outputs[TRAINING_SET_SIZE][OUTPUT_SIZE]) {
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        for (int i = 0; i < TRAINING_SET_SIZE; i++) {
            // Set input values
            for (int j = 0; j < INPUT_SIZE; j++) {
                nn->input[j] = training_inputs[i][j];
            }

            // Perform forward pass
            forward_pass(nn);

            // Perform backpropagation
            backpropagate(nn, training_outputs[i]);
        }
    }
}

// Print output of the neural network
void print_output(NeuralNetwork *nn) {
    printf("Output: ");
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        printf("%f ", nn->output[i]);
    }
    printf("\n");
}

int main() {
    NeuralNetwork nn;

    // Initialize weights and biases
    initialize_weights(nn.weights_input_hidden, INPUT_SIZE * HIDDEN_SIZE);
    initialize_weights(nn.weights_hidden_output, HIDDEN_SIZE * OUTPUT_SIZE);
    initialize_weights(nn.bias_hidden, HIDDEN_SIZE);
    initialize_weights(nn.bias_output, OUTPUT_SIZE);

    // Training data
    double training_inputs[TRAINING_SET_SIZE][INPUT_SIZE] = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}
    };

    double training_outputs[TRAINING_SET_SIZE][OUTPUT_SIZE] = {
        {0},
        {1},
        {1},
        {0}
    };

    // Train the neural network
    train(&nn, training_inputs, training_outputs);

    // Test the neural network with the training data
    for (int i = 0; i < TRAINING_SET_SIZE; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            nn.input[j] = training_inputs[i][j];
        }
        forward_pass(&nn);
        printf("Input: ");
        for (int j = 0; j < INPUT_SIZE; j++) {
            printf("%f ", nn.input[j]);
        }
        printf("Output: ");
        print_output(&nn);
    }

    return 0;
}
