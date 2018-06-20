
image_names = dir('../sample_images');
image_names = image_names(3:end);

image = zeros(512*10,512*10,'uint8');
variances = zeros(10, 10);

for i=0:(length(image_names)-1)
    im = imread(strcat('../sample_images/',image_names(i+1).name));

    x = mod(i, 10) * 512 + 1;
    y = floor(i / 10) * 512 + 1;

    image(y:y+511, x:x+511) = im;
    
    x = mod(i, 10) + 1;
    y = floor(i / 10) + 1;
    
    % needed for trimmping triangular black corners
    imvar = im(12:end-12, 12:end-12);
    variances(x,y) = var(double(imvar(:)));
end

XS = repmat(1:10, 1, 10) + 0.5;
YS = reshape(repmat(1:10, 10, 1),[1,100]) + 0.5;
variances = reshape(variances, [1,100]);

imagesc([1, 11], [1, 11], image);
colormap(gray);
hold on;

scatter(XS, YS, variances);



