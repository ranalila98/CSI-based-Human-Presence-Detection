clc; clear; close all;
addpath(genpath('npy-matlab-master'));

base_folder = 'preprocessed_data';
channels = {'channel_1', 'channel_36'};
rooms = {'a', 'b', 'c'};

% Load All Empty Data for Training
X_empty = [];

for ch = 1:length(channels)
    ch_folder = fullfile(base_folder, channels{ch});
    ch_num = extractAfter(channels{ch}, 'channel_');

    for r = 1:length(rooms)
        filename = sprintf('preprocessed_%s_%s_empty.npy', rooms{r}, ch_num);
        filepath = fullfile(ch_folder, filename);

        if exist(filepath, 'file') == 2
            data = readNPY(filepath);
            [Nt, Btw, NrxNtx, Nf] = size(data);
            reshaped = reshape(data, Nt, Btw * NrxNtx * Nf);
            X_empty = [X_empty; reshaped];
        else
            warning('Missing: %s', filepath);
        end
    end
end

fprintf('Total empty room samples: %d\n', size(X_empty, 1));

% Split Empty Data into Training (80%) and Testing (20%) Sets
cv = cvpartition(size(X_empty, 1), 'HoldOut', 0.2);
X_train = X_empty(training(cv), :);
X_test_empty = X_empty(test(cv), :);

fprintf('Training set: %d samples\n', size(X_train, 1));
fprintf('Testing set (empty rooms): %d samples\n', size(X_test_empty, 1));

% Train OC-SVM
rng(42);
SVMModel = fitcsvm(X_train, ones(size(X_train, 1), 1), ...
    'KernelFunction', 'rbf', ...
    'OutlierFraction', 0.05, ...
    'BoxConstraint', 1, ...
    'Standardize', true, ...
    'KernelScale', 'auto');

% Save Model
if ~exist('trained_models', 'dir')
    mkdir('trained_models');
end

save('trained_models/OCSVM_Model.mat', 'SVMModel');
disp('Saved: trained_models/OCSVM_Model.mat');

save('trained_models/TestData.mat', 'X_test_empty');
disp('Saved: trained_models/TestData.mat (20% empty data for testing)');


