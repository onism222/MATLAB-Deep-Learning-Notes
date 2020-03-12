clear

load('MnistConv.mat')

k = 2; % figure of the number 2
x = X( :, :, 2)
y1 = conv( x, W1 );
y2 = ReLU(y1);
y3 = pool(y2);
y4 = reshape( y3, [ ], 1);
v5 = W5 * y4;
y5 = ReLU(v5);
v = Wo * y5;
y = softmax(v);

figure;
display_network(x( : ));
title('Input Image')

convFilters = zeros( 9*9, 20 );
for i = 1 : 20
    filter = W1( :, :, i );
    convFilters( :, i ) = filter( : );
end
figure
display_network(convFilters);
title('Convolution Filters')

fList = zeros( 20 * 20, 20 );
for i = 1 : 20
    feature = y1( :, :, i );
    fList( :, i ) = feature( : );
end
figure
display_network(fList);
title('Features [Convolution]')

fList = zeros( 20 * 20, 20 );
for i = 1 : 20
    feature = y2( :, :, i );
    fList( :, i ) = feature( : );
end
figure
display_network(fList);
title('Features [Convolution + ReLU]')

fList = zeros( 10 * 10, 20 );
for i = 1 : 20
    feature = y3( :, :, i );
    fList( :, i ) = feature( : );
end
figure
display_network(fList);
title('Features [Convolution + ReLU + MeanPool]')






