% add a dir to octave search path with
% addpath('~/Downloads/function_folder')

v = zeros(10,1);

% for loop
for i=1:length(v),
    v(i) = 2^i;
end;
v

% while loop
i=1;
while i <= 5,
    v(i) = 100;
    i = i+1;
end;
v

% break and continue also available
i = 1;
while true,
    v(i) = 999;
    i = i+1;
    if i == 6,
        break;
    end;
end;
v

% if/else
v(1) = 2;
if v(1) == 1,
    disp('The value is one');
elseif v(1) == 2,
    disp('The value is two');
else,
    disp('meow!');
end;

disp('use square function from squareThisNumber.m file')
squareThisNumber(2)

disp('using cost function')
X = [1 1; 1 2; 1 3]
y = [1; 2; 3;]
theta = [0; 0];
j = costFunctionJ(X,y,theta)
