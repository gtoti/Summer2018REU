
image_names = dir('../sample_images');
image_names = image_names(3:end);

imhisteq = zeros(512*10, 512*10,'uint8');
image = zeros(512*10,512*10,'uint8');
image_th = zeros(5120);

for i=0:(length(image_names)-1)
    im = imread(strcat('../sample_images/',image_names(i+1).name));
    
    im_hq = histeq(im(12:end-12, 12:end-12));
    im_th = imbinarize(im, graythresh(im_hq)*1.15);

    x = mod(i, 10) * 512 + 1;
    y = floor(i / 10) * 512 + 1;

    image(y:y+511, x:x+511) = im;
    %imhisteq(y:y+511, x:x+511) = im_hq;
    image_th(y:y+511, x:x+511) = im_th;
    
    x = mod(i, 10) + 1;
    y = floor(i / 10) + 1;
    
end

imshow(image_th);

