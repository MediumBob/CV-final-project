% CS4337.001 - Computer Vision - Dr. Metsis
% Group 3 Final Project
% Members: Robert Elizondo, Kevin Garcia Lopez, Jacob Lopez

%% - preprocessing

clear all; 
close all;
directories;
addpath(data_directory);
addpath(code_directory);
addpath(training_directory);
s = filesep;
load classifiers.mat
addpath([training_directory s 'test_face_photos'])

face_horizontal = 100;
face_vertical = 100;
face_size = [100 100];

%% check accuracy for cropped faces
% path = strcat(training_directory, s, "test_cropped_faces");
% ds = imageDatastore(path);
% imgs = readall(ds);
% % 
% hit = 0;
% miss = 0;
% falsePos = 0;
% falseNeg = 0;
% threshold = 1;
% for i=1:size(imgs,1)
%     temp_img = imgs{i};
%     result = apply_classifier_aux(temp_img, boosted_classifier, weak_classifiers, [100 100]);
%     sorted = sort(result(:));
%     [r,c] = find(sorted(length(sorted)) == result);
%     max = result(r(1),c(1));
%     if max > threshold && labels(i,1) == -1
%         falsePos = falsePos +1;
%     elseif max < threshold && labels(i,1) == 1
%         falseNeg = falseNeg +1;
%     end
% end
% cropped_faces_accuracy = ((size(imgs,1) - falsePos - falseNeg)/size(imgs,1)) * 100;

%% check accuracy for non-faces
% path = strcat(training_directory, s, "test_nonfaces");
% ds = imageDatastore(path);
% imgs = readall(ds);
% % 
% hit = 0;
% miss = 0;
% falsePos = 0;
% falseNeg = 0;
% threshold = 1;
% for i=1:size(imgs,1)
%     temp_img = imgs{i};
%     result = apply_classifier_aux(temp_img, boosted_classifier, weak_classifiers, [100 100]);
%     sorted = sort(result(:));
%     [r,c] = find(sorted(length(sorted)) == result);
%     max = result(r(1),c(1));
%     if max > threshold && labels(i,1) == -1
%         falsePos = falsePos +1;
%     elseif max < threshold && labels(i,1) == 1
%         falseNeg = falseNeg +1;
%     end
% end
% nonfaces_accuracy = ((size(imgs,1) - falsePos - falseNeg)/size(imgs,1)) * 100;

%% check accuracy for non-faces
% path = strcat(training_directory, s, "test_face_photos");
% ds = imageDatastore(path);
% imgs = readall(ds);
% % 
% hit = 0;
% miss = 0;
% falsePos = 0;
% falseNeg = 0;
% threshold = 100;
% for i=1:size(imgs,1)
%     temp_img = imgs{i};
%     result = apply_classifier_aux(temp_img, boosted_classifier, weak_classifiers, [100 100]);
%     sorted = sort(result(:));
%     [r,c] = find(sorted(length(sorted)) == result);
%     max = result(r(1),c(1));
%     if max > threshold && labels(i,1) == -1
%         falsePos = falsePos +1;
%     elseif max < threshold && labels(i,1) == 1
%         falseNeg = falseNeg +1;
%     end
%     boxed_img = draw_rectangle2(temp_img, r(1), c(1), face_vertical, face_horizontal);
%     imshow(boxed_img);
% end
% 
% faces_accuracy = ((size(imgs,1) - falsePos - falseNeg)/size(imgs,1)) * 100;
% 
% 

%%

path = strcat(training_directory, s, "test_face_photos");
ds = imageDatastore(path);
imgs = readall(ds);
% init faces for testing dataset
test_faces = zeros(face_vertical,face_horizontal,size(imgs,1));
% convert from cell to matrix
falsePosCount = 0;
falseNegCount = 0;
for i=1:size(imgs,1)
    temp_img = imgs{i};
    tic; result = boosted_detector_demo(temp_img, 1:0.5:3, boosted_classifier, weak_classifiers, [100 100], 2); toc
    prediction = boosted_predict(temp_img, boosted_classifier, weak_classifiers, 5);
    if prediction > 0 && labels(i,1) == -1
        falsePosCount = falsePosCount + 1;
    elseif prediction < 0 && labels(i,1) == 1
        falseNegCount = falseNegCount + 1;
    end
    figure(i); imshow(result, []);
end


