function [W1, W5, Wo] = MnistConv(W1, W5, Wo, X, D)

alpha = 0.01;
beta = 0.95;

momentum1 = zeros(size(W1));
momentum5 = zeros(size(W5));
momentumo = zeros(size(Wo));

N = length(D);

bsize = 100;
blist = 1 : bsize : (N-bsize+1);

% one epoch loop
%
for batch = 1 : length(blist)
    dW1 = zeros(size(W1));
    dW5 = zeros(size(W5));
    dWo = zeros(size(Wo));
    % mini-batch loop
    %
    begin = blist(batch);
    for k = begin : begin + bsize - 1
        % forward pass = inference
        %
        x = X(:, :, k); % Input, 28 x 28
        y1 = conv(x, W1); % Convolution, 20x20x20
        y2 = ReLU(y1); % 
        y3 = pool(y2); % Pool, 10x10x20
        y4 = reshape(y3, [ ], 1); % 2000
        v5 = W5 * y4; % ReLU, 360
        y5 = ReLU(v5);
        v = Wo * y5; % softmax, 10
        y = softmax(v); %
        
        % One-hot encoding
        %
        d = zeros(10, 1);
        d(sub2ind(size(d), D(k), 1)) = 1; % sub2ind函数是MATLAB中对矩阵索引号检索的函数
        % Backpropagation
        %
        e = d - y; % Output layer
        delta = e;
        
        e5 = Wo' * delta;  % Hidden(ReLU) layer
        delta5 = (y5 > 0) .* e5;
        
        e4 = W5' * delta5; % Pooling layer
        
        e3 = reshape(e4, size(y3));
        
        e2 = zeros(size(y2));
        W3 = ones(size(y2)) / (2 * 2);
        for c = 1 : 20
            e2(:, :, c) = kron(e3(:, :, c), ones([2, 2])) .* W3(:, :, c);
        end
        
        delta2 = (y2 > 0) .* e2; % ReLU layer
        
        delta1_x = zeros(size(W1)); % Convolutional layer
        for c = 1 : 20
            delta1_x(:, :, c) = conv2(x(:, :), rot90(delta2(:, :, c), 2), 'valid');
        end
        
        dW1 = dW1 + delta1_x;
        dW5 = dW5 + delta5 * y4';
        dWo = dWo + delta * y5';
        
    end
    % Update weights
    %
    dW1 = dW1 / bsize;
    dW5 = dW5 / bsize;
    dWo = dWo / bsize;
    
    momentum1 = alpha * dW1 + beta * momentum1;
    W1 = W1 + momentum1;
    
    momentum5 = alpha * dW5 + beta * momentum5;
    W5 = W5 + momentum5;
    
    momentumo = alpha * dWo + beta * momentumo;
    Wo = Wo + momentumo;
end

end
   
