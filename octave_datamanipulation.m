A = [1 2; 3 4; 5 6;];

% get element
A(3,2); % gets A32 element of the A matrix

% get second row of A
A(2,:);

% get second column of A
A(:,2);

% get everything from first and third rows of A, all columns
A([1 3], :)

% can also be used for assignment
% assign second column of A to these values
A(:,2) = [10; 11; 12]

% append column vector to the right 
A = [A, [100; 101; 102]]

% put all elements of A into a single column vector
A(:); % resultant 9x1 vector with all elements from A

% concatenating two matrices side by side
A = [1 2; 3 4; 5 6;];
B = [7 8; 9 10; 11 12;];

C = [A B]
% same as C = [A, B]

% concatenating vertically
D = [A; B]