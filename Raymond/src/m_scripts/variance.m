

img = imread('../progress_pics/artifact_removal/mean_removal.jpg');
img = double(img) ./ 255.0;
variances = zeros(1, 100);
n = 1;

for i=0:9
    for j=0:9
        y = i*512;
        x = j*512;
        im = img(y+1:y+512, x+1:x+512);
        variances(n) = var(im(:));
        n=n+1;
    end
end

XS = repmat(1:10, 1, 10) + 0.5;
YS = reshape(repmat(1:10, 10, 1),[1,100]) + 0.5;
variances = variances * 25500 * 1.5;

imagesc([1, 11], [1, 11], img);
colormap(gray);
hold on;

scatter(XS, YS, variances, 'LineWidth', 2);
