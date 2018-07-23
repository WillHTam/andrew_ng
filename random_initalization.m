% Symmetry breaking
% Initialize each Theta to a random value in [-Epsilon, Epsilon]
% Note: not the same as the epsilon defined for gradient checking
% Here just want a continuous value between 0 and 1

% If the dimensions of 
% Theta1 is 10x11, T2 is 10x11 and T3 is 1x11

Theta1 = rand(10,11)*(2*INIT_EPSILON) - INIT_EPSILON;

Theta2 = rand(10,11)*(2*INIT_EPSILON) - INIT_EPSILON;

Theta3 = rand(1,11)*(2*INIT_EPSILON) - INIT_EPSILON;
