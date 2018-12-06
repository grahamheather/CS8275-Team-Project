function [] = extract_file(in_filename, out_filename)
    % load data from file
    data_file = load(in_filename);
    [num_channels, num_samples] = size(data_file.datastart);
    data = cell(num_samples, num_channels);
    for sample = 1:num_samples
        for channel = 1:num_channels
            data{sample, channel} = data_file.data(data_file.datastart(channel, sample):data_file.dataend(channel, sample)); 
        end
    end
    
    % extract features
    features = {
        @(channel) mav(channel),
        @(channel) rms(channel),
        @(channel) zero_crossings(channel),
        @(channel) mean_dwt(channel, 1),
        @(channel) mean_dwt(channel, 2),
        @(channel) mean_dwt(channel, 3),
        @(channel) dwt_energy(channel, 1),
        @(channel) dwt_energy(channel, 2)
    };
    feature_data = extract_features(data, features, num_samples, num_channels);
    
    % output
    save(out_filename, 'feature_data', '-v7.3')
    
    exit
end

function feature_data = extract_features(data, features, num_samples, num_channels)
    size_features = size(features);
    num_features = size_features(1);
    num_segments = 8;
    num_channels = 6;
    
    % extract features
    feature_data = zeros(num_samples, num_features * num_channels * num_segments);
    for sample = 1:num_samples
        for channel = 1:num_channels
          current_channel = data(sample, channel);
          current_channel = current_channel{1};
          channel_length = size(current_channel, 2);
          segment_length = channel_length / num_segments;
          for segment = 1:num_segments
            segment_start = segment_length * (segment - 1) + 1;
            segment_end = segment_length * segment;
            current_segment = current_channel(segment_start:segment_end);
            for feature = 1:num_features
              % location in the feature vector for the sample
              feature_offset = (channel - 1) * num_segments * num_features + (segment - 1) * num_features + feature;
              % extract feature from segment
              feature_data(sample, feature_offset) = features{feature}(current_segment);
            end
          end
        end
    end
end

function out = mav(channel)
  out = mean(abs(channel));
end

function out = rms(channel)
  out = sqrt(mean((channel) .^ 2));
end

function out = zero_crossings(channel)
    out = size(find(channel(:).*circshift(channel(:),[-1,0])<=0), 1);
end

function out = get_dwt(channel)
    dwtmode('per','nodisplay');
    wname = 'db10';
    level = fix(log2(length(channel)));
    level = level - floor(level/2);
    [C,L] = wavedec(channel,level,wname);
    out = detcoef(C,L,1:level); % wc 
    
end

function out = mean_dwt(channel, index)
    wc = get_dwt(channel);

    for i = 1:length(wc)
        sum_ = sum(wc{i});
        mean_{i} = sum_/ length(wc{i});
    end
    
    mean_ = cell2mat(mean_);
    max_val = max(mean_);
    min_val = min(mean_);
    mean_val = mean(mean_);
    
    out_values = [max_val,min_val,mean_val];
    out = out_values(index);
end

function out = dwt_energy(channel, index)
    wc = get_dwt(channel);

    for i=1:length(wc)
     signal_Pow = wc{i}.^2;  % Compute the second power of signal points
     total_energy = sum(signal_Pow);
     dwtEnergy{i} = total_energy;
    end
    max_ = max(cell2mat(dwtEnergy));
    mean_energy = mean(cell2mat(dwtEnergy));

    out_values = [ max_ , mean_energy];
    out = out_values(index);
end