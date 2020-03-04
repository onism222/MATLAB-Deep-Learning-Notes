clear

rng(3)
X = zeros(5, 5, 5);

X(:, :, 1) = [ 0 1 1 0 0;
                   0 0 1 0 0;
                   0 0 1 0 0;
                   0 0 1 0 0;
                   0 1 1 1 0
                 ];
X(:, :, 2) = [ 1 1 1 1 0;
                   0 0 0 0 1;
                   0 1 1 1 0;
                   1 0 0 0 0;
                   1 1 1 1 1
                 ];
X(:, :, 3) = [ 1 1 1 1 0;
                   0 0 0 0 1;
                   0 1 1 1 0;
                   0 0 0 0 1;
                   1 1 1 1 0
                ];
X(:, :, 4) = [ 0 0 0 1 0;
                   0 0 1 1 0;
                   0 1 0 1 0;
                   1 1 1 1 1;
                   0 0 0 1 0
                ];
X(:, :, 5) = [ 1 1 1 1 1;
                   1 0 0 0 0;
                   1 1 1 1 0;
                   0 0 0 0 1;
                   1 1 1 1 0
                ];
          D = [ 1 0 0 0 0;
                   0 1 0 0 0;
                   0 0 1 0 0;
                   0 0 0 1 0;
                   0 0 0 0 1
                 ];
weight1 = 2*rand(50, 25) - 1;
weight2 = 2*rand( 5, 50) - 1;

for epoch = 1:10000 % train
    [weight1, weight2] = MultiClass(weight1, weight2, X, D);
end
N = 5; % inference
for k = 1:N
    x = reshape(X(:, :, k), 25, 1);
    v1 = weight1*x;
    y1 = sigmoid(v1);
    v = weight2*y1;
    y = softmax(v)
end