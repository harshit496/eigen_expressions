function result = read_gray(filename)

temp = double(imread(filename));

if (size(temp, 3) == 1)
    result = temp;
else
    result = 0.3*temp(:,:,1) + 0.59*temp(:,:,2) + 0.11*temp(:,:,3);
end
