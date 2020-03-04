function y = conv(x, W)

[wrow, wcol, numFilters] = size(W); 
[xrow, xcol, ~] = size(x);                

yrow = xrow - wrow + 1;
ycol = xcol - wcol + 1;

y = zeros(yrow, ycol, numFilters);

for k = 1 : numFilters
    filter = W(:, :, k);
    filter = rot90(squeeze(filter), 2); 
    % squeeze:表示返回一个数组，其元素与输入数组 A 相同，但删除了长度为 1 的维度
    % 例如，如果 A 是 3×1×2 数组，则 squeeze(A)返回 3×2 矩阵
    % B=rot90(A,k)表示将矩阵A逆时针旋转(90×k)°以后返回B，k取负数时表示顺时针旋转
    y(:, :, k) = conv2(x, filter, 'valid');
end

end
%        x: 输入图像，W: 卷积核kernel, 或称滤波器filter
%        假设输入图像x大小为ma x na，卷积核W大小为mb x nb，则
%        当shape=full时，返回全部二维卷积结果，即返回C的大小为（ma+mb-1）x（na+nb-1）
%       shape=same时，返回与A同样大小的卷积中心部分
%        shape=valid时，不考虑边界补零，即只要有边界补出的零参与运算的都舍去，返回C的大小为（ma-mb+1）x（na-nb+1）
% x = [1, 1, 1, 3; 4, 6, 4, 8; 30, 0, 1, 5; 0, 2, 2, 4]; W = [1, 0; 0, 1];
% y = conv(x, W)


