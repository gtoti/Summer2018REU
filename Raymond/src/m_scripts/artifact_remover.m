
function A = artifact_remover(images)
% images must be a 3 dimensional array of grayscale images with
% values between 0 and 1. 
% indexes are (a, b, c)
% a and b index row and column
% c indexes the image

% assert that images is a 3d array
assert(length(size(images))==3);

% average images
im_avg = mean(images, 3);
im_avg_avg = mean(im_avg(:));

% using median
im_med = median(images, 3);
im_med_med = median(im_med(:));

A = zeros(size(images));

for i=1:size(images,3)
    % using mean image as artifact mask
    B = images(:,:,i) - im_avg;
    B = B - mean(B(:)) + im_avg_avg
    
    A(:,:,i) = B;
end

mini = min(A(:));
maxi = max(A(:));
A = (A - mini) / (maxi - mini);

end
