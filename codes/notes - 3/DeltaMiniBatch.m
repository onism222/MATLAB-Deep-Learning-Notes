function [weight] = DeltaMiniBatch(weight, data_input, correct_output)

alpha = 0.9; % learning rate
dwsum1 = zeros(3, 1); % initialize the sum of dalta weight of the first mini batch
dwsum2 = zeros(3, 1); % initialize the sum of dalta weight of the second mini batch
N = 4; % the number of data points in training data
M = 2; % the number of mini batch = 2 

for k = 1 : N/M % the first mini batch
    x = data_input(k, :)';
    d = correct_output(k);
    
    v = weight * x; % {1X3} * {3X1}
    y = sigmoid(v);
    
    e = d - y; % error = correct output - actual output 
    
    delta = y * (1-y) * e; % equation (8)
    
    dw = alpha * delta * x; % delta rule
    
    dwsum1 = dwsum1 + dw; % the sum of dalta weight of the first mini batch
    
end
dwavg2 = dwsum1/ (N/M); % average weight update(i.e., equation (11))

    weight(1) = weight(1) + dwavg2(1); % weight updating
    weight(2) = weight(2) + dwavg2(2); % weight updating
    weight(3) = weight(3) + dwavg2(3); % weight updating
    
 for k = (N/M) +1 : N % the second mini batch
    x = data_input(k, :)';
    d = correct_output(k);
    
    v = weight * x; % {1X3} * {3X1}
    y = sigmoid(v);
    
    e = d - y; % error = correct output - actual output 
    
    delta = y * (1-y) * e; % equation (8)
    
    dw = alpha * delta * x; % delta rule

    dwsum2 = dwsum2 + dw; % the sum of dalta weight of the second mini batch
    
end
dwavg2 = dwsum2 / (N/M); % average weight update(i.e., equation (11))

    weight(1) = weight(1) + dwavg2(1); % weight updating
    weight(2) = weight(2) + dwavg2(2); % weight updating
    weight(3) = weight(3) + dwavg2(3); % weight updating   
    
end