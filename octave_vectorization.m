% use the built in libraries to get faster and more efficent code

% h(x) = sum from j=0 to n of theta_j * x_j
    % can be expressed as theta^T*x

% unvectorized h(x)
prediction = 0.0;
for j = 1:n+1,
    prediction = prediction + theta(j) * x(j)
end;

% vectorized, simpler and runs more efficiently
prediction = theta' * x;


# in C++
double prediction = 0.0;
for (int j = 0; j <= n; j++) {
    prediction += theta[j] * x[j];
}

double prediction = theta.transpose() * x;

% vectorizing gradient descent
    % compressing each computational step into one function
% treat theta as a vector
% theta = theta - alpha * delta
    % alpha being the learning rate
    % delta being a n+1 dimensional vector representing the gradient step formula
        % the summation of (h(x^i) - y^i) represeted as vector multiplied by x, obviously also a vector 


% suppose three vectors u,v,w
% unvectorized is 
for j = 1:3,
    u(j) = 2*v(j) + 5*w(j);
end;
% vectorized is 
u = 2*v + 5*w

% vectorization can also help linear regressions with 1000's or 10's of 1000's of features