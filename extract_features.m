all_data = load("all_data.mat");

global skin_supra = [4 5.3 4.3 5.6 4 3.8 4 3.2 4 4.5 4 3.5 5 5.6 5.2 4 5 4.8 4.2 7.3 5 7.3 4.2 4.8 2.2 5.6 6.83 5.8 4.67 3.67 5.25 3.67 5.83 6.67 4.17 5 5 4.6 8.8 6.83 7.166 7.66 4.667 7.3 6 7.33 8.16 7.6 7 14.3 4.3 11 4.17 6.67 8 4.17 2.83];
global skin_infra = [4.5 5.6 2.6 6.6 4.3 3.8 2.6 3.8 3.3 5 3 3 4.5 5 4.8 3.5 4 3.2 3 8 4.7 6.3 3.2 3.2 1 6 6.83 6.83 3.83 3.5 5.25 3.16 6.3 6.17 5.17 5.3 5.33 3.6 7 4.3 6 4 4.333 7.6 7 4.83 5.16 5.3 5 11.3 5.3 13 2.8 5 9 2.67 2.5];

function results = features_all(filename, data, features)
  global skin_supra
  global skin_infra
  
  num_patients = size(data.patients)(1);
  num_vowels = size(data.vowels)(2);
  num_features = size(features)(1);
  
  % extract the features
  feature_data = cell(num_patients, num_vowels);
  for patient = 1:num_patients
    for vowel = 1:num_vowels
      num_samples = size(data.D{patient, vowel})(1);
      feature_data{patient, vowel} = zeros(num_samples, num_features);
      for sample = 1:num_samples
        for feature = 1:num_features
          feature_data{patient, vowel}(sample, feature) = features{feature}(data.D{patient, vowel}, skin_supra(1, patient), skin_infra(1, patient));
        end
      end
    end
  end
  
  save(filename, 'feature_data', '-v7')
end

features_all('all_features.mat', all_data, {@(channels, ss, si) mean(channels(1){1})})