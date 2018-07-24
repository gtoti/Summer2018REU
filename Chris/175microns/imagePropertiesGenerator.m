clear all;
imgSet = imageSet('C:\Users\chris\Documents\csreu\dewetting\175microns','recursive');
dirs = {imgSet.Description}; % array of folder names

for folder = dirs
    cd(folder{1,1});
    images = dir('*].jpg');
    images = {images.name};
    imageNum = length(images);
    count = 1;
    Polymer = strings(imageNum,1);
    WhiteDensity_unweighted = zeros(imageNum,1);
    WhiteDensity_weighted = zeros(imageNum,1);
    Entropy = zeros(imageNum,1);
    IntensitySum = zeros(imageNum,1);
    IntensityMean = zeros(imageNum,1);
    IntensityStd = zeros(imageNum,1);
    IntensityMed = zeros(imageNum,1);
    Skew = zeros(imageNum,1);
    
    for image = images % each an array with name and folder
        title = image{1,1};
        im = imread(title);
        [height,width,values] = size(im);
        im = imcrop(im, [0,0,width,height-125]); % [xmin, ymin, width, height]
        
        %NW,NE,SW,SE
        
        pixelCount = (height-125) * width;
        hsv = rgb2hsv(im); % hsv (mx3)
        gs = rgb2gray(im);
        gs2bw = im2bw(gs,graythresh(gs));
        s = hsv(:,:,2); % saturation (mX1)
        s2bw = im2bw(s,graythresh(s)); % saturation, binarized
        s2bw = ~s2bw;
        stats = regionprops(s2bw,'Area');
        allArea = [stats.Area];
        WhiteDensity_unweighted(count,1) = sum(allArea)/ pixelCount; % from binarized saturation image***
        WhiteDensity_weighted(count,1) = bwarea(s2bw) / pixelCount;
                
        % Ignores really small regions
        %{
        ignores regions with area < 50 pixels
        to_delete = properties.Area < 50;
        properties(to_delete,:) = [];
        % inverts black and white
        if height(properties) == 1
            bw = ~bw;
            properties = regionprops('table', bw, {'Area', 'Eccentricity', 'EquivDiameter', 'EulerNumber', 'MajorAxisLength', 'MinorAxisLength', 'Orientation', 'Perimeter'});
        end
        %}
    
        % Outputs BW(gs), S, BW(s) images
        [filepath, name, ext] = fileparts(title);
        snm = strcat(name,'-s','.jpg'); % saturation image
        gsbwnm = strcat(name,'-gsbw','.jpg');
        sbwnm = strcat(name,'-sbw','.jpg'); % binarized of saturated image
        imwrite(s, snm);
        imwrite(gs2bw,gsbwnm);
        imwrite(s2bw, sbwnm);

        % Updates matrix of general properties
        Polymer(count,1) = replace(name," [Published]", "");
        Entropy(count,1) = entropy(s);
        IntensitySum(count,1) = sum(s(:)); % sum of all saturation values
        IntensityStd(count,1) = std2(s); % std
        IntensityMean(count,1) = mean(s(:));
        IntensityMed(count,1) = median(s(:));
        Skew(count,1) = skewness(s(:));
        count = count+1;
    end
    prop = table(Polymer, WhiteDensity_unweighted, WhiteDensity_weighted, Entropy, IntensitySum, IntensityStd, IntensityMean, IntensityMed, Skew);
    cd ../
    writetable(prop,strcat('summary', '_', extractBefore(folder{1,1},'_')));        
end


