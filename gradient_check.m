% numerically computing the approximation of all partial derivatives
for i = 1;n,
    thetaPlus = theta;
    thetaPlus(i) = thetaPlus(i) + EPSILON;
    thetaMinus = theta;
    thetaMinus(i) = thetaMinus(i) - EPSILON;
    gradApprox(i) = (J(thetaPlus) - J(thetaMinus)) / (2*EPSILON);
end;

% in the neural network, use this to compute the partial derivative of
% the cost function in respect to the parameter
% Take the gradient gotten from back prop - DVec
% and check that gradApprox (somewhat)= DVec
% if these two numbers are similar, then be confident that the derivatives are being 
% calculated correctly. 


