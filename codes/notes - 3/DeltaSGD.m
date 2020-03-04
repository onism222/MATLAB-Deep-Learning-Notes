function [weight] = DeltaBatch(weight, data_input, correct_output)

alpha = 0.9; % learning rate
N = 4;

for k = 1 : N
    x = data_input(k, :)';
    d = correct_output(k);
    
    v = weight * x; % {1X3} * {3X1}
    y = sigmoid(v);
    
    e = d - y; % error = correct output - actual output 
    
    delta = y * (1-y) * e; % equation (8)
    
    dw = alpha * delta * x; % delta rule
    
    weight(1) = weight(1) + dw(1); % equation (9)
    weight(2) = weight(2) + dw(2); % equation (9)
    weight(3) = weight(3) + dw(3); % equation (9)
end
end