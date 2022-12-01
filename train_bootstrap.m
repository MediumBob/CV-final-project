% CS4337.001 - Computer Vision - Dr. Metsis
% Group 3 Final Project
% Members: Robert Elizondo, Kevin Garcia Lopez, Jacob Lopez

%% - preprocessing
s = filesep;
directories;
addpath(code_directory)
addpath(data_directory)
addpath (code_directory, s, 'Code')

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

%% get a subset of the training examples

faces_subset = faces(:,:,1:761); % take a quarter of the total examples
num_nonfaces = size(nonfaces,3);
nonfaces_subset = nonfaces(:,:,1:(num_nonfaces/4));

%% precompute responses of the current training examples on all weak classifiers
example_number = size(faces_subset, 3) + size(nonfaces_subset, 3);
labels = zeros(example_number, 1);
labels (1:size(faces_subset, 3)) = 1;
labels((size(faces_subset, 3)+1):example_number) = -1;
examples = zeros(face_vertical, face_horizontal, example_number);
examples (:, :, 1:size(faces_subset, 3)) = face_integrals(:,:,1:761);
examples(:, :, (size(faces_subset, 3)+1):example_number) = nonface_integrals(:,:,1:975);

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

% big for loop here?

%% train the detector

boosted_classifier = AdaBoost(responses, labels, 15);

%% apply detector to all training images

% init matrix to store difficult examples
difficult_examples = zeros(face_vertical,face_horizontal,1000);

falsePosCount = 0;
falseNegCount = 0;

for i=1:size(imgs,1)
    temp_img = imgs{i};
    %tic; result = boosted_detector_demo(temp_img, 1:0.5:3, boosted_classifier, weak_classifiers, [100 100], 2); toc
    prediction = boosted_predict(temp_img, boosted_classifier, weak_classifiers, 15);
%% identify mistakes 
    if prediction > 0 && labels(i,1) == -1
        falsePosCount = falsePosCount + 1;
        % add whatever image was wrong into the difficult_exampels 
        difficult_examples(:,:,i) = temp_img;
        
    elseif prediction < 0 && labels(i,1) == 1
        falseNegCount = falseNegCount + 1;
        % add whatever image was wrong into the difficult_exampels 
    end
    %figure(i); imshow(result, []);
end


%% add mistakes to training examples



save classifiers.mat boot_boosted_classifier boot_weak_classifiers boot_labels

