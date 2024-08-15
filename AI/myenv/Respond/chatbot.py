# Import necessary libraries
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# --- scikit-learn Model ---
print("Starting scikit-learn model...")

# Load Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train the model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Evaluate the model
print("scikit-learn Model Evaluation:")
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print('Classification Report:')
print(classification_report(y_test, y_pred))

# --- TensorFlow Model ---
print("Starting TensorFlow model...")

# Load and preprocess MNIST data
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

# Build the TensorFlow model
tf_model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
tf_model.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])

# Train the model
tf_model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.1)

# Evaluate the model
loss, accuracy = tf_model.evaluate(X_test, y_test)
print("TensorFlow Model Evaluation:")
print(f'Loss: {loss:.4f}')
print(f'Accuracy: {accuracy:.4f}')

# --- PyTorch Model ---
print("Starting PyTorch model...")

# Define the neural network model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Load and preprocess MNIST data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_data = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_data = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# Initialize model, loss function, and optimizer
pytorch_model = SimpleNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(pytorch_model.parameters(), lr=0.001)

# Training loop
for epoch in range(5):
    pytorch_model.train()
    for batch in train_loader:
        images, labels = batch
        optimizer.zero_grad()
        outputs = pytorch_model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch+1} completed')

# Evaluation
pytorch_model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch in test_loader:
        images, labels = batch
        outputs = pytorch_model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f'PyTorch Model Accuracy: {accuracy:.4f}')

# --- Chatbot Integration ---
print("Starting chatbot...")

model_name = 'gpt2'
gpt2_model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

def get_chatbot_response(prompt):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    output = gpt2_model.generate(input_ids, max_length=50, num_return_sequences=1)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# Example interaction with the chatbot
user_input = "Tell me a joke about AI."
response = get_chatbot_response(user_input)
print("Chatbot Response:")
print(response)

# Example interaction with the chatbot
user_input = "Tell me a joke about AI."
response = get_chatbot_response(user_input)
print("Chatbot Response:")
print(response)

# --- Math Functions ---
print("\n--- Math Functions ---")

# Basic operations
print(f"Square root of 16: {math.sqrt(16)}")
print(f"Factorial of 5: {math.factorial(5)}")
print(f"Sin of 30 degrees: {math.sin(math.radians(30))}")
print(f"Cos of 60 degrees: {math.cos(math.radians(60))}")

# Using numpy for more advanced functions
array = np.array([1, 2, 3, 4, 5])
print(f"\nNumpy array: {array}")
print(f"Mean of array: {np.mean(array)}")
print(f"Standard deviation of array: {np.std(array)}")
print(f"Logarithm (base 10) of 100: {np.log10(100)}")
print(f"Exponential of 2: {np.exp(2)}")

# --- Code Problem Solving ---
def solve_code_problem(problem_statement):
    print("\n--- Solving Code Problem ---")
    print(f"Problem Statement: {problem_statement}")
    try:
        # Simple execution environment for solving problems
        exec(problem_statement)
    except Exception as e:
        print(f"Error while solving problem: {e}")

# Example code problem
problem_code = """
def add_numbers(a, b):
    return a + b

result = add_numbers(5, 3)
print(f"Result of adding numbers: {result}")
"""
solve_code_problem(problem_code)

# --- Code Generation ---
def generate_code(prompt):
    print("\n--- Generating Code ---")
    response = openai.Completion.create(
        engine="text-davinci-003",  # or use "gpt-3.5-turbo" if available
        prompt=prompt,
        max_tokens=150
    )
    code = response.choices[0].text.strip()
    print("Generated Code:")
    print(code)

# Example code generation prompt
generation_prompt = "Write a Python function that calculates the factorial of a number."
generate_code(generation_prompt)
