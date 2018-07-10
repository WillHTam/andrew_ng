% get size of matrix
A = [1 2; 3 4; 5 6]
size(A);

% size returns a 1x2 matrix
sz = size(A) % sz is now a 1x2 matrix of [3, 2]

% get dimension of rows
size(A,1);
% columns
size(A,2);

% get size of longest dimension, mostly for vectors
v = [1 2 3 4]
length(v); %4
length(A); %3

% command bash commands available
ls
pwd
cd '/path in quotes'

%load files
load filename.ext
% equivalent to load('filename.ext')
% => will be loaded in a variable called 'filename'

% show currently available variables in local scope
who;

% show variables AND their sizes
whos; 
% Size is the size of the matrix
% Class are the types 
    % double is double position; floating point numbers
    % logical for booleans
    % char for strings

% get rid of variable
clear variable

% get first ten elements of vector Y
v = Y(1:10);

% save variable
save hello.mat v; % saves v to hello.mat, MATLAB format
save hello.txt v -ascii % saves to readable format in .txt file

% clear all variables
clear



