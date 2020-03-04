function [weight1, weight2] = BackPropXOR(weight1, weight2, data_input,...
, correct_output)

alpha = 0.9;
N = 4;

for k = 1 : N
    x = data_input(k, :)';
    d = correct_output(k);
    
    v1 = weight1 * x;
    y1 = sigmoid(v1);
    v = weight2 * y1;
    y = sigmoid(v);
    
    e = d - y; % calculate the error of the output to the correct output
    delta = y .* (1 - y) .* e;
    
    e1 = weight2' * delta; % calculate the error of the hidden layer
    delta1 = y1 .* (1 - y1) .* e1;
    
    % Adjust the weights
    dw1 = alpha * delta1 * x';
    weight1 = weight1 + dw1;
    
    dw2 = alpha * delta * y1';
    weight2 = weight2 + dw2;
end
end