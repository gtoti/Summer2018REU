
function deviations = glo_loc_deviation(imgs, n, feature_func)
    dims = size(imgs);
    assert(mod(dims(1), n) == 0 & mod(dims(2), n) == 0);
    
    lenx = dims(1) / n;
    leny = dims(2) / n;
    assert(lenx > 1 & leny > 1);
    stridex = lenx / 2;
    stridey = leny / 2;
    
    deviations = zeros(1, dims(3));
    
    for k = 1:dims(3)
        locals = zeros(2 * n - 1);
        globals = feature_func(imgs(:,:,k));
        
        for i = 0:(2*n-2)
            for j = 0:(2*n-2)
               x = i * stridex;
               y = j * stridey;
               img = imgs(x+1:x+lenx, y+1:y+leny, k);
               locals(i+1, j+1) = feature_func(img(:));
            end
        end
        
        locals = locals(:) - globals;
        deviations(k) = norm(locals);
    end
    
end
