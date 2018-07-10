X = inputs
y = target
theta = coefficient(s)
num_iters = number of gradient steps

m = length(y) % number of training examples

% computing cost for a linear regression

costJ = (1/(2*m))*(X*theta -y)' * (X*theta -y)

% normalize features
%    X_norm is normalized X where mean of each feature is 0 and stddev is 1
mu = mean(X)
sigma = std(X)
X_norm = (X - (ones(length(X), 1) * mu)) ./ (ones(length(X), 1) * sigma)

% gradient descent
%   save one step to J_history, save new theta

for iter = 1:num_iters
    theta = theta - alpha * (1/m) * (((X*theta) - y)' * X)'
    
    J_history(iter) = computeCost(X, y, theta)
end

% normal equation

theta = pinv(X'*X)*X'*y
