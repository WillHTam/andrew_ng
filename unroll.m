# want to put each of these matrices into a vector
Theta1 = ones(10 ,11);
Theta2 = 2*ones(10,11);
Theta3 = 3*ones(1,11);

thetaVec = [ Theta1(:); Theta2(:); Theta3(:)];

size(thetaVec)

# getting back original matrices
# ones
reshape(thetaVec(1:110), 10,11)
# twos
reshape(thetaVec(111:220), 10,11)
# threes
reshape(thetaVec(221:231), 1, 11)