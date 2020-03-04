clear

data_input = [ 0, 0, 1; 0, 1, 1; 1, 0, 1; 1, 1, 1 ]; % training data
correct_output = [ 0; 0; 1; 1 ]; % correct outputs(i.e., labels)
weight1 = 2 * rand(1, 3) - 1; % initializes the weights with random real numbers between [-1, 1]
weight2 = weight1;
weight3 = weight1;

E1 = zeros(1000, 1);
E2 = E1;
E3 = E1;



for epoch = 1 : 1000 % the number of epoch = 1000
    weight1 =  DeltaSGD(weight1, data_input, correct_output);
    weight2 =  DeltaBatch(weight2, data_input, correct_output);
    weight3 =  DeltaMiniBatch(weight3, data_input, correct_output);
    
    es1 = 0; es2 = 0; es3 = 0;
    N = 4;
    for k = 1 : N
        x = data_input(k, :)';
        d = correct_output(k);
        % SGD
        v1 = weight1 * x;
        y1 = sigmoid(v1);
        es1 = es1 + (d - y1)^2;
        % Batch
        v2 = weight2 * x;
        y2 = sigmoid(v2);
        es2 = es2 + (d - y2)^2;
        % Mini Batch
        v3 = weight3 * x;
        y3 = sigmoid(v3);
        es3 = es3 + (d - y3)^2;
    end
    E1(epoch) = es1 / N;
    E2(epoch) = es2 / N;
    E3(epoch) = es3 / N;
    
end

plot(E1, '--b', 'Linewidth',1);
hold on
plot(E2, '-.g', 'Linewidth',1);
hold on 
plot(E3, '-r', 'Linewidth',1);
xlabel('the number of Epoch')
ylabel('Average of Training error')
legend('SGD', 'Batch', 'Mini Batch')


                       