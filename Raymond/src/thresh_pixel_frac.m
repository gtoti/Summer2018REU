
image = imread('../16px_crop_shadow_triangle_removed.jpg');
sub_im_pxlen = 512 - 32;

gth = graythresh(image);
im_bin = imbinarize(image, gth);

%imshow(1-im_bin);

blacks = zeros(10);
for i = 0:9
    for j = 0:9
        xmin = j * sub_im_pxlen + 1;
        xmax = xmin + sub_im_pxlen - 1;
        ymin = i * sub_im_pxlen + 1;
        ymax = ymin + sub_im_pxlen - 1;
        
        im = im_bin(ymin:ymax, xmin:xmax);
        blacks(i+1, j+1) = sum(1-im(:)) / (sub_im_pxlen^2) * 255;
    end
end

x = repmat(1:10, 10, 1);
y = x';
x = x(:) + 0.5;
y = y(:) + 0.5;
blacks = blacks(:) * 10;

imagesc([1, 11], [1, 11], image);
colormap(gray);
hold on;

scatter(x, y, blacks);
