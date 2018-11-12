all_data = load("all_data.mat");


global classes = [0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 1 1 1 0 1 1 0 1 0 1 1 1 0 1 0 0];
global skin_supra = [4 5.3 4.3 5.6 4 3.8 4 3.2 4 4.5 4 3.5 5 5.6 5.2 4 5 4.8 4.2 7.3 5 7.3 4.2 4.8 2.2 5.6 6.83 5.8 4.67 3.67 5.25 3.67 5.83 6.67 4.17 5 5 4.6 8.8 6.83 7.166 7.66 4.667 7.3 6 7.33 8.16 7.6 7 14.3 4.3 11 4.17 6.67 8 4.17 2.83];
global skin_infra = [4.5 5.6 2.6 6.6 4.3 3.8 2.6 3.8 3.3 5 3 3 4.5 5 4.8 3.5 4 3.2 3 8 4.7 6.3 3.2 3.2 1 6 6.83 6.83 3.83 3.5 5.25 3.16 6.3 6.17 5.17 5.3 5.33 3.6 7 4.3 6 4 4.333 7.6 7 4.83 5.16 5.3 5 11.3 5.3 13 2.8 5 9 2.67 2.5];

function results = train_all(data, features, classifier, metrics)
  global classes
  global skin_supra
  global skin_infra
  
  num_patients = size(data.patients)(1);
  num_vowels = size(data.vowels)(2);
  num_metrics = size(metrics)(1);
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
    
  results = cell(num_patients)(1);
  for left_out = 1:num_patients
    % combine data
    iter_data = [];
    test_data = [];
    for i = 1:num_patients
      for vowel = 1:num_vowels
        if i != left_out
          iter_data = [iter_data; feature_data{i, vowel}];
        else
          test_data = [test_data; feature_data{i, vowel}];
        end
      end
    end
    iter_classes = [ classes(1, 1:left_out-1), classes(1, left_out+1:end) ];
    
    % train the classifier
    test_predicted = classifier(iter_data, iter_classes', test_data);
    tp = 0;
    tn = 0;
    fp = 0;
    fn = 0;
    for i = 1:size(test_predicted)(1)
      if classes(1, left_out) == 1
        if test_predicted(i) == 1
          tp = tp + 1;
        else
          fn = fn + 1;
        end
      else
        if test_predicted(i) == 1
          fp = fp + 1;
        else
          tn = tn + 1;
        end
      end
    end
    
    % calculate the provided metrics
    results{left_out} = cell(num_metrics)(1);
    for metric = 1:size(metrics)(1)
      results{left_out}{metric} = metrics{metric}(tp, tn, fp, fn);
    end
    
  end
end

function labels = svm_classifier(X, Y, test)
  SVMModel = fitcsvm(X, Y, 'KernelFunction', 'rbf', 'Standardize', true);
  [labels, scores] = predict(SVMModel, test);
end

results = train_all(all_data, {@(channels, ss, si) mean(channels(1){1})}, @svm_classifier, {@(tp, tn, fp, fn) tp; @(tp, tn, fp, fn) tn; @(tp, tn, fp, fn) fp; @(tp, tn, fp, fn) fn});
disp(results)