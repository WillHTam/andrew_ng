A = [1 2; 3 4; 5 6];
B = [11 12; 14 14; 15 16];
C = [1 1; 2 2];

% matrix multiplication
disp('A * C')
A*C

% > period usually denotes element-wise operations
% element wise multiplication
disp('A .* B')
A .* B

% element wise squaring
disp('A .^ 2')
A .^ 2

% element wise reciprocal
disp('1 ./ v')
v = [1; 2; 3];
1 ./ v

% element wise log
disp('log(v)')
log(v)

% element wise base e exponentiation
disp('e^x')
exp(v)

% absolute value
disp('abs(v)')
abs(v)

% negative
-v 
% equivalent to -1*v

% increment all by one
disp('incremented v by 1')
v + ones(length(v),1)
% actually could just do
v + 1

% transpose A
disp('transposed A')
A'

% get max
disp('get max')
a = [1 15 2 0.5]
val = max(a)

% get max and the index of the maximum
disp('get max, index')
[val, ind] = max(a)

% element wise comparison
% returns 'list' of boolean 0/1 
disp('match')
a < 3 % 1 0 1 1

% find elements which match operator
disp('find')
find(a < 3)

% magic
% returns magic square matrix, where columns, rows, and diagonals sum up to same thing
% good for generating matrix of whole numbers
disp('magic 3')
A = magic(3)

% return rows and columns that return true in find
disp('r c of A>=7')
[r,c] = find(A >= 7)
disp('the first element of r matches with the first of c')
disp(r(1))
disp(c(1))

% sum up a
sum(a)

% reduce a with x*y
prod(a)

% floor to round down to nearest integer
% 0.5 to 0
floor(a)

% round up
% 0.5 to 1
floor(a)