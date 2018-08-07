% On a 24-bit color represntation, each pixel is represented as three 8-bit unsigned integers (from 0 to 255) specifying RGB intensity. Known as RGB encoding. 

% Preprocessing will therefore involve reducing to 16 colors from thousands. Representing each pixel will therefore only require storage of the index and the RG values for the colors

A = imread('bird_small.png');
% Creates a 3 dim matrix A
% first two indices are pixel position, and last index is RG or B

disp( A(50, 33, 3) ); % displays blue intensity at row 50, col 33
