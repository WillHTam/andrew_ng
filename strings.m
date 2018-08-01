str = 'cat';

str(1); % returns c

strcmp(str, 'cat'); % returns 1
strcmp(str, 'dog'); % returns 0

animalList = cell();
idx = strmatch(str, animalList, 'exact'); % returns with index of string in cell-array

animalList{idx} % returns 'cat'

