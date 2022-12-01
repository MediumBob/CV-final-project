% CS4337.001 - Computer Vision - Dr. Metsis
% Group 3 Final Project
% Members: Robert Elizondo, Kevin Garcia Lopez, Jacob Lopez

%% - preprocessing

s = filesep;
directories;
addpath(code_directory)

%% load the training dataset - if it does not exist, create it

if (~isfile('trainingdataset.mat')) 
   testingdata;
else 
    load trainingdataset.mat
end

%% get weak classifiers

number = 1000;
face_horizontal = 100;
face_vertical = 100;
face_size = [100 100];
weak_classifiers = cell(1, number);
for i = 1:number
    weak_classifiers{i} = generate_classifier(face_vertical, face_horizontal);
end

%% precompute responses of all training examples on all weak classifiers
example_number = size(faces, 3) + size(nonfaces, 3);
labels = zeros(example_number, 1);
labels (1:size(faces, 3)) = 1;
labels((size(faces, 3)+1):example_number) = -1;
examples = zeros(face_vertical, face_horizontal, example_number);
examples (:, :, 1:size(faces, 3)) = face_integrals;
examples(:, :, (size(faces, 3)+1):example_number) = nonface_integrals;

classifier_number = numel(weak_classifiers);

responses =  zeros(classifier_number, example_number);

for example = 1:example_number
    integral = examples(:, :, example);
    for feature = 1:classifier_number
        classifier = weak_classifiers {feature};
        responses(feature, example) = eval_weak_classifier(classifier, integral);
    end
    disp(example);
end

%% get weights

weights = ones(example_number, 1) / example_number;
cl = random_number(1, 1000)
[error, thr, alpha] = weighted_error(responses, labels, weights, cl)
weights = ones(example_number, 1) / example_number;
% next line takes about 8.5 seconds.
tic; [index error threshold] = find_best_classifier(responses, labels, weights); toc
%disp([index error]);
boosted_classifier = AdaBoost(responses, labels, 15);

save classifiers.mat boosted_classifier weak_classifiers labels

