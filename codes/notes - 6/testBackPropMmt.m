clear 
data_input = [ 0, 0, 1; 0, 1, 1; 1, 0, 1; 1, 1, 1]; % training data
correct_output = [0; 1; 1; 0]; % correct outputs(i.e., labels)
weight1 = 2 * rand(4, 3) - 1; 
weight2 = 2 * rand(1, 4) - 1; 


for epoch = 1:10000 % train
[weight1, weight2] = BackPropMmt(weight1, weight2, data_input,...
correct_output);
end

N = 4; % inference
for k = 1 : N
x = data_input(k, :)';
v1 = weight1 * x;
y1 = sigmoid(v1);
v = weight2 * y1;
y = sigmoid(v)
end