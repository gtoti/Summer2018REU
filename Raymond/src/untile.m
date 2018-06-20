
function C = untile(image, height, width)
%  UNTILE
%  Given an image composed of tiled images, this function grabs subimages
%  that are of dimensions of the given height and width row wise from the
%  tiled image
%
%  example:
%
%   magic(4)
%       16     2     3    13
%        5    11    10     8
%        9     7     6    12
%        4    14    15     1
% 
%   untile(magic(4), 2, 2)
%
%   ans(:,:,1) =           ans(:,:,2) = 
% 
%       16     2               3    13
%        5    11               10    8
% 
%   ans(:,:,3) =           ans(:,:,4) = 
% 
%        9     7                6    12
%        4    14               15     1
%
    [r, c] = size(image);
    C = reshape(image, height, r / height, c);
    C = permute(C, [1,3,2]);
    C = reshape(C, height, width, (r * c) / (height * width));
end