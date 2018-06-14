
image_names = dir('../sample_images');
image_names = image_names(3:end);

crop = 16;
image_side_length = 512;
side_length = image_side_length - crop*2;
n_image_side = 10;
image = zeros(side_length*n_image_side, side_length*n_image_side,'uint8');

for i=0:(length(image_names)-1)
    im = imread(strcat('../sample_images/',image_names(i+1).name));
    im = im(crop:end-crop, crop:end-crop);
    
    x = mod(i, n_image_side) * side_length + 1;
    y = floor(i / n_image_side) * side_length + 1;

    image(y:y+side_length, x:x+side_length) = im;
end

imshow(image);
imwrite(image, '16px_cornercrop_glance.jpg');