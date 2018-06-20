
function A = artifact_remover(images)
% images must be a 3 dimensional array of grayscale images with
% values between 0 and 1. 
% indexes are (a, b, c)
% a and b index row and column
% c indexes the image

% assert that images are square
assert(size(images, 1) == size(images,2));

% average images
im_avg = mean(images, 3);
avg_bright = mean(im_avg(:));
A = zeros(size(images));

for i=1:size(images,3)
    B = images(:,:,i);
    B = B - im_avg;
    B = B - min(B(:));
    B = B - mean(B(:)) + mean(im_avg(:));
    A(:,:,i) = B;
end

mini = min(A(:));
maxi = max(A(:));

A = (A - mini) / (maxi - mini);

end
