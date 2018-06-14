
image = imread('../16px_cornercrop_glance.jpg');
pxlen = 512 - 32;

im_hq = histeq(image);
im_bin = imbinarize(im_hq, graythresh(im_hq)*0.5);


im_avg = zeros(pxlen);
for i = 0:9
    for j = 0:9
        xmin = j * pxlen + 1;
        xmax = xmin + pxlen - 1;
        ymin = i * pxlen + 1;
        ymax = ymin + pxlen - 1;
        
        subim = double(image(ymin:ymax, xmin:xmax)) ./ 255.0;
        
        im_avg = im_avg + subim / 100;
    end
end

f = histeq(im_avg).*2;
f = imgaussfilt(f, 9);
f = 1 - f;
f = f * 1.1;
%imshow(f);


image = double(image) / 255.0;

for i = 0:9
    for j = 0:9
        xmin = j * pxlen + 1;
        xmax = xmin + pxlen - 1;
        ymin = i * pxlen + 1;
        ymax = ymin + pxlen - 1;
        
        filt = image(ymin:ymax, xmin:xmax) .* f;
        image(ymin:ymax, xmin:xmax) = image(ymin:ymax, xmin:xmax) + 0.09 * filt;
        %image(ymin:ymax, xmin:xmax) = histeq(image(ymin:ymax, xmin:xmax));
    end
end

imshow(image);
%imwrite(image, '../16px_crop_shadow_triangle_removed.jpg')

