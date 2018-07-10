A = magic(10);
x = ones(10,1);

v = zeros(10, 1);
for i = 1:10
  for j = 1:10
    v(i) = v(i) + A(i, j) * x(j);
  end
end

disp('v')
v

disp('1')
x*A

disp('2')
Ax

disp('3')
x' * A

disp('4')
sum(A * x)