function [ y ] = softmax( x )
ex = exp( x );
y = ex / sum( ex );
end