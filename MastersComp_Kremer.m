%% Problem 1: Linear Fit
%% Jake Kremer

clc; clear; close all;

% Seed for reproducibility
rng(42);

% Generate data
x_generated = linspace(0, 5, 10)';
n_true = 0.06;
a_true = 0.25;
m_true = 0.57;
b_true = 0.11;
noise = 0.001 * randn(size(x_generated)); % Gaussian noise
y_generated = n_true * exp(-a_true * (m_true * x_generated + b_true).^2) + noise;

% Hyperparameters
epochs = 130;
learning_rate = 0.1;

% Initialize parameters
m = rand(); % Initial guess for slope
b = rand(); % Initial guess for intercept

% Gradient descent loop
N = length(x_generated);
for epoch = 1:epochs
    % Predicted values
    y_pred = m * x_generated + b;
    
    % Gradients
    dL_dm = -(2/N) * sum(x_generated .* (y_generated - y_pred));
    dL_db = -(2/N) * sum(y_generated - y_pred);
    
    % Parameter updates
    m = m - learning_rate * dL_dm;
    b = b - learning_rate * dL_db;

    % Compute loss (Mean Squared Error)
    loss = (1/N) * sum((y_generated - y_pred).^2);

    % (Optional) Print loss every 1000 epochs
    if mod(epoch, 10) == 0
        fprintf('Epoch %d, Loss: %.6f\n', epoch, loss);
    end
end

% Display optimized parameters
fprintf('Optimized m: %.4f\n', m);
fprintf('Optimized b: %.4f\n', b);

% Plot results
figure;
plot(x_generated, y_generated, 'bo-', 'DisplayName', 'Actual Data');
hold on;
plot(x_generated, m * x_generated + b, 'r-', 'DisplayName', 'Predicted Data');
legend();
xlabel('x');
ylabel('y');
title('Linear Fit Using Gradient Descent');
grid on;
