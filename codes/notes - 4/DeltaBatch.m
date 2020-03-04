function [weight] = DeltaBatch(weight, data_input, correct_output)

alpha = 0.9; % learning rate
dwsum = zeros(3, 1);
N = 4;

for k = 1 : N
    x = data_input(k, :)';
    d = correct_output(k);
    
    v = weight * x; % {1X3} * {3X1}
    y = sigmoid(v);
    
    e = d - y; % error = correct output - actual output 
    
    delta = y * (1-y) * e; % equation (8)
    
    dw = alpha * delta * x; % delta rule
    
    %adds the individual weight updates of the entire training data to dwsum 
    dwsum = dwsum + dw; 
    
end
dwavg = dwsum / N; % average weight update(i.e., equation (11))

    weight(1) = weight(1) + dwavg(1); % weight updating
    weight(2) = weight(2) + dwavg(2); % weight updating
    weight(3) = weight(3) + dwavg(3); % weight updating
end