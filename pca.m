% dimensionality reduction from n dim to k dim 

% m - number of training examples
% x - input variable
% z - representation of X as projected on the lower dimensional surface
% u - vectors that express the lower dimensional surface

% Preprocessing - Feature Scaling & Mean Normalization
%   mu_j = (1/m)*sum(1 to m)*x_j^i)
%   replace each x_j^i with x_j - mu_j
%   replace x_j^i with (xji - muj) / stddev of x

% Compute covariance matrix
Sigma = (1/m) * X' * X;
% X is nx1 vec,and its transpose is 1xn, so Sigma is nxn dim

% Compute Eigenvectors of Sigma
[U, S, V] = svd(Sigma); 
% svd calls octave's singular value decomposition
%   same as calling eig() but more numerically stable
% U will be an nxn matrix of the U's we want

% take first K columns of u
Ureduce = U(:, 1:k);
% since we are reducing from n dim to k dim, take first k columns to get u1...uK
% doing the slice will form Ureduce of nxK dim

% finally get the final reduced input
z = Ureduce' * X;
% the resultant z has K rows

% Reconstruction from Compressed representation
% given that z = Ureduce' * X;
Xapprox = Ureduce * z
% Only an approximation however
