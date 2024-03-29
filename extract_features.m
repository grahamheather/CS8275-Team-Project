all_data = load("all_data_matlab.mat");

skin_supra = [4 5.3 4.3 5.6 4 3.8 4 3.2 4 4.5 4 3.5 5 5.6 5.2 4 5 4.8 4.2 7.3 5 7.3 4.2 4.8 2.2 5.6 6.83 5.8 4.67 3.67 5.25 3.67 5.83 6.67 4.17 5 5 4.6 8.8 6.83 7.166 7.66 4.667 7.3 6 7.33 8.16 7.6 7 14.3 4.3 11 4.17 6.67 8 4.17 2.83];
skin_infra = [4.5 5.6 2.6 6.6 4.3 3.8 2.6 3.8 3.3 5 3 3 4.5 5 4.8 3.5 4 3.2 3 8 4.7 6.3 3.2 3.2 1 6 6.83 6.83 3.83 3.5 5.25 3.16 6.3 6.17 5.17 5.3 5.33 3.6 7 4.3 6 4 4.333 7.6 7 4.83 5.16 5.3 5 11.3 5.3 13 2.8 5 9 2.67 2.5];

zero_crossings=@(v) size(find(v(:).*circshift(v(:),[-1,0])<=0), 1);

features = {
  @(channel, ss, si) mav(channel),
  @(channel, ss, si) rms(channel),
  @(channel, ss, si) zero_crossings(channel),
  @(channel, ss, si) mean_dwt(channel, 1),
  @(channel, ss, si) mean_dwt(channel, 2),
  @(channel, ss, si) mean_dwt(channel, 3),
  @(channel, ss, si) dwt_energy(channel, 1),
  @(channel, ss, si) dwt_energy(channel, 2),
  @(channel, ss, si) getDASDV(channel),
  @(channel, ss, si) getTM3(channel),
  @(channel, ss, si) getAAC(channel)
  };

features_all('featuresV2.mat', all_data, features, skin_supra, skin_infra);

function features_all(filename, data, features, skin_supra, skin_infra) 
  size_patients = size(data.patients);
  size_vowels = size(data.vowels);
  size_features = size(features);
  
  num_patients = size_patients(1);
  num_vowels = size_vowels(2);
  num_features = size_features(1);
  
  % we need the same number of segments for every channel and
  % every sample and every patient (need a constant length
  % feature vector)
  % From Nuerzati's work:
  %   segment window length = 250e-3
  %   length of entire channel = 2s
  %   2s / .25 = 8 segments
  num_segments = 8;
  num_channels = 6;
  
  % extract the features
  feature_data = cell(num_patients, num_vowels);
  for patient = 1:num_patients
    disp(patient)
    for vowel = 1:num_vowels
      size_cell = size(data.D{patient, vowel});
      num_samples = size_cell(1);
      feature_data{patient, vowel} = zeros(num_samples, num_features * num_channels * num_segments);
      for sample = 1:num_samples
        for channel = 1:num_channels
          current_channel = data.D{patient, vowel}(sample, channel);
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
              feature_data{patient, vowel}(sample, feature_offset) = features{feature}(current_segment, skin_supra(1, patient), skin_infra(1, patient));
            end
          end
        end
      end
    end
  end
  
  save(filename, 'feature_data', '-v7.3')
end

function out = mav(channel)
  out = mean(abs(channel));
end

function out = rms(channel)
  out = sqrt(mean((channel) .^ 2));
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


function DASDV = getDASDV(channel)
    %% Difference Absolute Standard Deviation
    N = length(channel);
    sum_total = 0;
    for i =1:N-1
         sum_total = sum_total + ( channel(i+1) - channel(i) ).^2;
    end
    DASDV = sqrt ( (1 / (N - 1)) * sum_total );
end

function TM3 = getTM3(channel)
    %% TM3 :: 3rd Temporal Moment
    N = length(channel);
    channelcubed = channel.^3;
    TM3 = (1/N)* sum(channelcubed);
end



function AAC = getAAC(channel)
    %% AAC :: Average Amplitude Change
    N = length(channel);
    s_total = 0;
    for i =1:N-1
         s_total = s_total + abs( channel(i+1) - channel(i) );
    end

    AAC = (1/N) * s_total;
end