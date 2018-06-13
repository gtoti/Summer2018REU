
image_names = dir('images');
image_names = image_names(3:end);

imhisteq = zeros(512*10, 512*10,'uint8');
image = zeros(512*10,512*10,'uint8');
image_th = zeros(5120);

for i=0:(length(image_names)-1)
    im = imread(strcat('./images/',image_names(i+1).name));
    im_hq = histeq(im);
    im_th = imbinarize(im, graythresh(im)*1.05);

    x = mod(i, 10) * 512 + 1;
    y = floor(i / 10) * 512 + 1;

    image(y:y+511, x:x+511) = im;
    imhisteq(y:y+511, x:x+511) = im_hq;
    image_th(y:y+511, x:x+511) = im_th;
end

imshow(image_th);

