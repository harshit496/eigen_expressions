function [all_average, all_eigenvectors] = eigenexpressions(normalize_flag)
expressions = ["ANGRY";"FEAR";"DISGUST";"HAPPY";"NEUTRAL";"SAD";"SURPRISE"];
% d = 5;
all_eigenvectors = zeros(2704,2704,7);
all_average = zeros(2704,1,7);

for e = 1:length(expressions)
    train = 'jaffe/train/';
    train = strcat(train,expressions(e));
    trainFiles = dir(fullfile(train,'*.jpg'));
    vectors = zeros(2704,length(trainFiles));
    for i = 1:length(trainFiles)
        baseFileName = trainFiles(i).name;
        fullFileName = fullfile(train, baseFileName);
        [filepath,name,ext] = fileparts(fullFileName);
        image = read_gray(fullFileName);
        if normalize_flag == 1
            image = (image - mean(image(:))) / std(double(image(:)));
        end
        image = imresize(image, 0.2);
        vectors(:,i) = image(:);
    end
    [average, eigenvectors, eigenvalues] = compute_pca(vectors);
%     eigenvectors = eigenvectors(:,1:d);
    all_eigenvectors(:,:,e) = eigenvectors;
    all_average(:,:,e) = average;
    
end
end

function [average, eigenvectors, eigenvalues] = compute_pca(vectors)
number = size(vectors, 2);
% note that we are transposing twice
average = [mean(vectors')]';
centered_vectors = zeros(size(vectors));
for index = 1:number
    
    centered_vectors(:, index) = double(vectors(:, index)) - average;
end
covariance_matrix = centered_vectors * centered_vectors';
[eigenvectors eigenvalues] = eig( covariance_matrix);
eigenvalues = diag(eigenvalues);
[eigenvalues, indices] = sort(eigenvalues, 'descend');
eigenvectors = eigenvectors(:, indices);
end

