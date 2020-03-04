clear

data_input = [ 0, 0, 1; 0, 1, 1; 1, 0, 1; 1, 1, 1 ]; % training data
correct_output = [ 0; 0; 1; 1 ]; % correct outputs
weight11 = 2 * rand(4, 3) - 1; 
weight12 = 2 * rand(1, 4) - 1; 
weight21 = weight11;
weight22 = weight12;

E1 = zeros(1000, 1);
E2 = E1;

for epoch = 1 : 1000 % the number of epoch = 1000
    [weight11, weight12] =  BackPropCE(weight11, weight12, data_input, correct_output);
    [weight21, weight22] =  BackPropXOR(weight21, weight22, data_input, correct_output);

    es1 = 0; es2 = 0; 
    N = 4;
    for k = 1 : N
        x = data_input(k, :)';
        d = correct_output(k);
        % CE
        v1_CE = weight11 * x;
        y1_CE = sigmoid(v1_CE);
        v_CE = weight12 * y1_CE;
        y_CE = sigmoid(v_CE);
        es1 = es1 + (d - y_CE)^2;
        % XOR
        v1_XOR= weight21 * x;
        y1_XOR = sigmoid(v1_XOR);
        v_XOR = weight22 * y1_XOR;
        y_XOR = sigmoid(v_XOR);
        es2 = es2 + (d - y_XOR)^2;
    end
    E1(epoch) = es1 / N;
    E2(epoch) = es2 / N;
    
end

plot(E1, '--b', 'Linewidth',1);
hold on
plot(E2, '-.r', 'Linewidth',1);
xlabel('the number of Epoch')
ylabel('Average of Training error')
legend('Cross Entropy', 'Sum of Squared Error')