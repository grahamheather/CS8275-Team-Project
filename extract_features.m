%all_data = load("all_data_matlab.mat");

skin_supra = [4 5.3 4.3 5.6 4 3.8 4 3.2 4 4.5 4 3.5 5 5.6 5.2 4 5 4.8 4.2 7.3 5 7.3 4.2 4.8 2.2 5.6 6.83 5.8 4.67 3.67 5.25 3.67 5.83 6.67 4.17 5 5 4.6 8.8 6.83 7.166 7.66 4.667 7.3 6 7.33 8.16 7.6 7 14.3 4.3 11 4.17 6.67 8 4.17 2.83];
skin_infra = [4.5 5.6 2.6 6.6 4.3 3.8 2.6 3.8 3.3 5 3 3 4.5 5 4.8 3.5 4 3.2 3 8 4.7 6.3 3.2 3.2 1 6 6.83 6.83 3.83 3.5 5.25 3.16 6.3 6.17 5.17 5.3 5.33 3.6 7 4.3 6 4 4.333 7.6 7 4.83 5.16 5.3 5 11.3 5.3 13 2.8 5 9 2.67 2.5];

zero_crossings=@(v) size(find(v(:).*circshift(v(:),[-1,0])<=0), 1);

features = {
  @(channel, ss, si) mav(channel),
  @(channel, ss, si) rms(channel),
  @(channel, ss, si) zero_crossings(channel),
  @(channel, ss, si) ssc_peaks(channel),
  @(channel, ss, si) ssc_valleys(channel),
  @(channel, ss, si) willison_amp(channel)
  };

features_all('featuresV5.mat', all_data, features, skin_supra, skin_infra);

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

function out = ssc_peaks(channel)
    dx = diff(channel);
    dx = [0 dx];          %shifting because diff has one less element
    dx1 = [dx(2:end) 0];  %shifting dx for 1 element

    out = size(find( dx>0 & dx1<0), 2);  %compare dx and dx1 to find peaks
end

function out = ssc_valleys(channel)
    dx = diff(channel);
    dx = [0 dx];          %shifting because diff has one less element
    dx1 = [dx(2:end) 0];  %shifting dx for 1 element

    out = size(find( dx<0 & dx1>0), 2);  %compare dx and dx1 to find valleys
end

function out = willison_amp(channel)
    count=0;
    N=size(channel,2);
    for ii=1:N-1
        if (channel(ii)-channel(ii+1)) >= 10e-3
            count=count+1;
        end
    end
    out = count;
end