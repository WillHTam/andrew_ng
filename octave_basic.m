% change Octave prompt
PS1('>> ');

% semicolon suppresses output
a = 'no';
b = 'yes';
a;
b

% comma chain commands
% semicolons would do the same thing, but not print out the assignments
a=1, b=2, c=3

% simple math
6*8;
2^3; % exponent

% comparison, ans of 1 if true 0 is not
1 == 1;
1 ~= 2;

% logical operators
1 && 1;
1 || 0;
xor(1,0);

% assignment
pie = pi;
string = 'miaou';
com = (3>=1);

% printing
disp(pie)
disp(sprintf('2 decimals: %0.2f', pie))

% change output decimal places by inputting either of following 2 commands
format long 
format short

% matrix
A = [1 2; 3 4; 5 6];
% same as
B = [1 2;
3 4;
5 6];

% vector
V = [1; 2; 3];

% Sets first element to 1, increments by 0.1 until 2
% creates fat vector
v = 1:0.1:2;
% 1x6 of whole numbers from 1 to 6
x = 1:6;

% generate 1x3 matrix of zero's
zeros(1,3)

% generate 2x3 matrix of one's
ones(2,3);

% to generate 2x3 of two's
2*ones(2,3);

% 1x3 matrix of random numbers
rand(1,3);

% 1x3 matrix of gaussian/normal random numbers
randn(1,3);

w = -6 + sqrt(10)*(randn(1,10000))
hist(w)

% generate 4x4 identity matrix
eye(4);

% get help
help eye

