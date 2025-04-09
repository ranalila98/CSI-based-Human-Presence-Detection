clc; clear; close all;
addpath(genpath('npy-matlab-master'));

% Load trained model and test data
load('trained_models/OCSVM_Model.mat', 'SVMModel');
load('trained_models/TestData.mat', 'X_test_empty');

base_folder = 'preprocessed_data';
channels = {'channel_1', 'channel_36'};
rooms = {'a', 'b', 'c'};
occupied_label = -1;

% Initialize variables to store test data and labels
X_test = [];
y_test = [];
channel_labels = {};
room_labels = {};

% Load Empty Room Data for Testing
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
            X_test = [X_test; reshaped];
            y_test = [y_test; ones(size(reshaped, 1), 1)];
            channel_labels = [channel_labels; repmat(channels(ch), size(reshaped, 1), 1)];
            room_labels = [room_labels; repmat(rooms(r), size(reshaped, 1), 1)];
        else
            warning('Missing: %s', filepath);
        end
    end
end

% Load Occupied Room Data for Testing
for ch = 1:length(channels)
    ch_folder = fullfile(base_folder, channels{ch});
    ch_num = extractAfter(channels{ch}, 'channel_');

    for r = 1:length(rooms)
        filename = sprintf('preprocessed_%s_%s_occupied.npy', rooms{r}, ch_num);
        filepath = fullfile(ch_folder, filename);

        if exist(filepath, 'file') == 2
            data = readNPY(filepath);
            [Nt, Btw, NrxNtx, Nf] = size(data);
            reshaped = reshape(data, Nt, Btw * NrxNtx * Nf);
            X_test = [X_test; reshaped];
            y_test = [y_test; occupied_label * ones(size(reshaped, 1), 1)];
            channel_labels = [channel_labels; repmat(channels(ch), size(reshaped, 1), 1)];
            room_labels = [room_labels; repmat(rooms(r), size(reshaped, 1), 1)];
        else
            warning('Missing: %s', filepath);
        end
    end
end

fprintf('Total testing samples: %d\n', length(y_test));

% Predict
[~, scores] = predict(SVMModel, X_test);
y_pred = double(scores >= 0);
y_pred(y_pred == 1) = 1;
y_pred(y_pred == 0) = -1;

% Apply Post-Processing: Majority Voting
k = 5;
y_pred_post = y_pred;
for i = k:length(y_pred)
    window = y_pred(i-k+1:i);
    if sum(window) > 0
        y_pred_post(i) = 1;
    elseif sum(window) < 0
        y_pred_post(i) = -1;
    end
end

% Initialize structures to store metrics
metrics_before = struct();
metrics_after = struct();

% Unique channels and rooms
unique_channels = unique(channel_labels);
unique_rooms = unique(room_labels);

% Function to compute metrics
compute_metrics = @(y_true, y_pred) struct(...
    'Accuracy', sum(y_true == y_pred) / length(y_true), ...
    'Precision', sum((y_pred == -1) & (y_true == -1)) / (sum((y_pred == -1) & (y_true == -1)) + sum((y_pred == -1) & (y_true == 1))), ...
    'Recall', sum((y_pred == -1) & (y_true == -1)) / (sum((y_pred == -1) & (y_true == -1)) + sum((y_pred == 1) & (y_true == -1))), ...
    'F1_Score', 2 * (sum((y_pred == -1) & (y_true == -1)) / (sum((y_pred == -1) & (y_true == -1)) + sum((y_pred == -1) & (y_true == 1)))) * ...
               (sum((y_pred == -1) & (y_true == -1)) / (sum((y_pred == -1) & (y_true == -1)) + sum((y_pred == 1) & (y_true == -1)))) / ...
               ((sum((y_pred == -1) & (y_true == -1)) / (sum((y_pred == -1) & (y_true == -1)) + sum((y_pred == -1) & (y_true == 1)))) + ...
               (sum((y_pred == -1) & (y_true == -1)) / (sum((y_pred == -1) & (y_true == -1)) + sum((y_pred == 1) & (y_true == -1))))));

% Compute metrics for each channel
for ch = 1:length(unique_channels)
    idx = strcmp(channel_labels, unique_channels{ch});
    y_true_ch = y_test(idx);
    y_pred_ch = y_pred(idx);
    y_pred_post_ch = y_pred_post(idx);

    metrics_before.(unique_channels{ch}) = compute_metrics(y_true_ch, y_pred_ch);
    metrics_after.(unique_channels{ch}) = compute_metrics(y_true_ch, y_pred_post_ch);
end

% Compute metrics for each room
for r = 1:length(unique_rooms)
    idx = strcmp(room_labels, unique_rooms{r});
    y_true_rm = y_test(idx);
    y_pred_rm = y_pred(idx);
    y_pred_post_rm = y_pred_post(idx);

    metrics_before.(unique_rooms{r}) = compute_metrics(y_true_rm, y_pred_rm);
    metrics_after.(unique_rooms{r}) = compute_metrics(y_true_rm, y_pred_post_rm);
end

% Compute average metrics
metrics_before.Average = compute_metrics(y_test, y_pred);
metrics_after.Average = compute_metrics(y_test, y_pred_post);

% Display Results
fprintf('\n===== Channel-wise Metrics =====\n');
for ch = 1:length(unique_channels)
    ch_name = unique_channels{ch};
    fprintf('\nChannel: %s\n', ch_name);
    disp(' Before Post-Processing:');
    disp(metrics_before.(ch_name));
    disp(' After Post-Processing:');
    disp(metrics_after.(ch_name));
end

fprintf('\n===== Room-wise Metrics =====\n');
for r = 1:length(unique_rooms)
    rm_name = unique_rooms{r};
    fprintf('\nRoom: %s\n', rm_name);
    disp(' Before Post-Processing:');
    disp(metrics_before.(rm_name));
    disp(' After Post-Processing:');
    disp(metrics_after.(rm_name));
end

fprintf('\n===== Average Metrics =====\n');
disp(' Before Post-Processing:');
disp(metrics_before.Average);
disp(' After Post-Processing:');
disp(metrics_after.Average);

plot_metrics(metrics_before, metrics_after, unique_channels, unique_rooms);
