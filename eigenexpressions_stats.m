function d_vs_accuracy = eigenexpressions_stats(d_values);
expressions = ["ANGRY";"FEAR";"DISGUST";"HAPPY";"NEUTRAL";"SAD";"SURPRISE"];

total_testfiles = 42;  normalize_flag = 0;
d_vs_accuracy = zeros(length(d_values),2);
[all_average, all_eigenvectors] = eigenexpressions(normalize_flag);

for m = 1:length(d_values)
    d = d_values(m);
    to_display = ['No of principal components: ',num2str(d)];
    disp(to_display);
    confusion_matrix = zeros(7,7);
    %testing PCA with test files
    for j = 1:length(expressions)
        test = 'jaffe/test/';
        test = strcat(test,expressions(j));
        testFiles = dir(fullfile(test,'*.jpg'));
        %     vectors = zeros(2704,length(testFiles));
        for i = 1:length(testFiles)
            baseFileName = testFiles(i).name;
            fullFileName = fullfile(test, baseFileName);
            [filepath,name,ext] = fileparts(fullFileName);
            %         name
            vector = (read_gray(fullFileName));
            if normalize_flag == 1
                vector = (vector - mean(vector(:))) / std(double(vector(:)));
            end
            vector = imresize(vector, 0.2);
            
            distance = zeros(7,1);
            
            for k = 1:length(expressions)
                eigenvectors = all_eigenvectors(:,:,k);
                eigenvectors = eigenvectors(:,1:d);
                average =  all_average(:,:,k);
                
                %subract average
                centered = vector(:) - average(:);
                
                %projection
                projection = eigenvectors' * centered;
                projection = projection(:);
                
                %centered results
                centered_result = eigenvectors * projection;
                
                %back projection
                result = centered_result + average(:);
                
                distance(k,:) = norm(result-vector(:));
            end
            indexes = find(min(distance) == distance);
            %         expressions(indexes)
            confusion_matrix(j,indexes) = confusion_matrix(j,indexes) + 1;
            
            
        end
    end
    tp = diag(confusion_matrix);
    accuracy = sum(tp)/total_testfiles;
    confusion_matrix
    accuracy
    d_vs_accuracy(m,:) = [d,accuracy];
    
end
end