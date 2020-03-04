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
    [weight11, weight12] =  BackPropXOR(weight11, weight12, data_input, correct_output);
    [weight21, weight22] =  BackPropMmt(weight21, weight22, data_input, correct_output);

    es1 = 0; es2 = 0; 
    N = 4;
    for k = 1 : N
        x = data_input(k, :)';
        d = correct_output(k);
        % Simple(i.e., Mmt)
        v1_simple = weight11 * x;
        y1_simple = sigmoid(v1_simple);
        v_simple = weight12 * y1_simple;
        y_simple = sigmoid(v_simple)
        es1 = es1 + (d - y_simple)^2;
        % Mmt
        v1_mmt = weight21 * x;
        y1_mmt = sigmoid(v1_mmt);
        v_mmt = weight22 * y1_mmt;
        y_mmt = sigmoid(v_mmt)
        es2 = es2 + (d - y_mmt)^2;
    end
    E1(epoch) = es1 / N;
    E2(epoch) = es2 / N;
    
end

plot(E1, '--b', 'Linewidth',1);
hold on
plot(E2, '-.r', 'Linewidth',1);
xlabel('the number of Epoch')
ylabel('Average of Training error')
legend('Simple', 'Mmt')