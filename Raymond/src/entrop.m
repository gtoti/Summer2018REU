
image_names = dir('../sample_images');
image_names = image_names(3:end);

imhisteq = zeros(512*10, 512*10,'uint8');
image = zeros(512*10,512*10,'uint8');
image_th = zeros(5120);

entropies = zeros(10, 10);

for i=0:(length(image_names)-1)
    im = imread(strcat('../sample_images/',image_names(i+1).name));

    x = mod(i, 10) * 512 + 1;
    y = floor(i / 10) * 512 + 1;

    image(y:y+511, x:x+511) = im;
    
    x = mod(i, 10) + 1;
    y = floor(i / 10) + 1;
    
    % needed for trimmping triangular black corners
    imtrim = im(12:end-12, 12:end-12);
    entropies(x,y) = entropy(imtrim);
    %disp(entropies(x,y));
end

XS = repmat(1:10, 1, 10) + 0.5;
YS = reshape(repmat(1:10, 10, 1),[1,100]) + 0.5;

% exponential of the entropy to maximize differences
entropies = exp(reshape(entropies, [1,100])).^2 * 0.01;

imagesc([1, 11], [1, 11], image);
colormap(gray);
hold on;

scatter(XS, YS, entropies);

