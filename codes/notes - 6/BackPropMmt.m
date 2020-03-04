function [weight1, weight2] = BackPropMmt(weight1, weight2, data_input,...
correct_output)

alpha = 0.9;
beta = 0.9;

mmt1 = zeros(size(weight1));
mmt2 = zeros(size(weight2));

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
    
    % Adjust the weights use momentum
    dw1 = alpha * delta1 * x';
    mmt1 = dw1 + beta * mmt1;
    weight1 = weight1 + mmt1;
    
    dw2 = alpha * delta * y1';
    mmt2 = dw2 + beta * mmt2;
    weight2 = weight2 + mmt2;
end
end