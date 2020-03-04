clear

data_input = [ 0, 0, 1; 0, 1, 1; 1, 0, 1; 1, 1, 1]; % training data
correct_output = [0; 1; 1; 0]; % correct outputs(i.e., labels), note that it is different from previous [0; 0; 1; 1]
weight = 2 * rand(1, 3) - 1; % initializes the weights  with random real numbers between [-1, 1]

for epoch = 1 : 40000
    weight =  DeltaSGD(weight, data_input, correct_output)
end

% inference
N = 4;                        
for k = 1:N
    x = data_input(k, :)';
    v = weight * x;
    y = sigmoid(v)
end


                       