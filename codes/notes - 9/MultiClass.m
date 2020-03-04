function [weight1, weight2] = MultiClass(weight1, weight2, data_input,...
correct_output)

alpha = 0.9;
N = 5;

for k = 1 : N
    x = reshape(data_input(:, :, k), 25, 1); %size(data_input) = [5, 5, 5]
    d = correct_output(k, :)';
    
    v1 = weight1 * x;
    y1 = sigmoid(v1);
    v = weight2 * y1;
    y = softmax(v);
    
    e = d - y; % calculate the error of the output to the correct output
    delta = e;
    
    e1 = weight2' * delta; % calculate the error of the hidden layer
    delta1 = y1 .* (1 - y1) .* e1;
    
    % Adjust the weights
    dw1 = alpha * delta1 * x';
    weight1 = weight1 + dw1;
    
    dw2 = alpha * delta * y1';
    weight2 = weight2 + dw2;
end
end