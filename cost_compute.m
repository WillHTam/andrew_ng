% given theta is a two-dim vector with theta1 and theta2
% and J(theta) is (theta1 - 5)^2 + (theta2 - 5^2)
% the minimum of the cost function is t1 and t2 = 5
% and expressed by this function:

function [jVal, gradient] = costFunction(theta)

jVal = (theta(1) - 5)^2 + (theta(2) - 5)^2;

gradient = zeros(2,1);

gradient(1) = 2*(theta(1)) - 5;
gradient(2) = 2*(theta(2)) - 5;

% options data structure
% 'Gradobj on' means that a gradient will be provided to the algorithm
% Maximum iterations is 100
options = optimset('GradObj', 'on', 'MaxIter', '100')

% provide initial guess for theta
initialTheta = zeros(2,1);

% fminunc is 'function minimum unconstrained'
% @ is pointer to costFunction above
[optTheta, functionVal, exitFlag] = fminunc(@costFunction, initialTheta, options);