% Computer Vision HW1

% Load input img
checkerboard = imread('imgs/checkerboard.png');
box = imread('imgs/box.pgm');
book = imread('imgs/book.pgm');
basmati = imread('imgs/basmati.pgm');
scene = imread('imgs/scene.pgm');
cow = imread('imgs/cow.png');
noised_check = imnoise(checkerboard, 'gaussian', 0, 3/255);
noised_scene = imnoise(scene, 'gaussian', 0, 3/255);

% 1.1
subplot(1,3,1); 
imshow(checkerboard, []);
title('Input image');
subplot(1,3,2);
imshow(Harris(checkerboard, 5, 'shi'), []);
title('Shi-Tomasi, 5x5 window');
subplot(1,3,3);
imshow(Harris(checkerboard, 5, 'harmonic'), []);
title('Harmonic Mean, 5x5 window');
figure; 

% 1.2
subplot(1,4,1);
imshow(box, []);
title('Input image');
subplot(1,4,2);
imshow(Harris(box, 3, 'harmonic'), []);
title('Box, 3x3 window');
subplot(1,4,3);
imshow(Harris(box, 5, 'harmonic'), []);
title('Box, 5x5 window');
subplot(1,4,4);
imshow(Harris(box, 9, 'harmonic'), []);
title('Box, 9x9 window');
figure; 

subplot(1,4,1); 
imshow(book, []);
title('Input image');
subplot(1,4,2); 
imshow(Harris(book, 3, 'harmonic'), []);
title('Book, 3x3 window');
subplot(1,4,3); 
imshow(Harris(book, 5, 'harmonic'), []);
title('Book, 5x5 window');
subplot(1,4,4); 
imshow(Harris(book, 9, 'harmonic'), []);
title('Book, 9x9 window');
figure;

subplot(1,4,1); 
imshow(basmati, []);
title('Input image');
subplot(1,4,2); 
imshow(Harris(basmati, 3, 'harmonic'), []);
title('Basmati, 3x3 window');
subplot(1,4,3); 
imshow(Harris(basmati, 5, 'harmonic'), []);
title('Basmati, 5x5 window');
subplot(1,4,4);
imshow(Harris(basmati, 9, 'harmonic'), []);
title('Basmati, 9x9 window');
figure;

subplot(1,4,1); 
imshow(scene, []);
title('Input image');
subplot(1,4,2); 
imshow(Harris(scene, 3, 'harmonic'), []);
title('Scene, 3x3 window');
subplot(1,4,3); 
imshow(Harris(scene, 5, 'harmonic'), []);
title('Scene, 5x5 window');
subplot(1,4,4); 
imshow(Harris(scene, 9, 'harmonic'), []);
title('Scene, 9x9 window');
figure;

% noise
subplot(1,4,1); 
imshow(noised_scene, []);
title('Gaussian noised scene');
subplot(1,4,2); 
imshow(Harris(noised_scene, 3, 'harmonic'), []);
title('noised, 3x3 window');
subplot(1,4,3); 
imshow(Harris(noised_scene, 5, 'harmonic'), []);
title('noised, 5x5 window');
subplot(1,4,4); 
imshow(Harris(noised_scene, 9, 'harmonic'), []);
title('noised, 9x9 window');
figure;

subplot(1,4,1); 
imshow(noised_check, []);
title('Gaussian noised checkboard');
subplot(1,4,2); 
imshow(Harris(noised_check, 3, 'harmonic'), []);
title('noised, 3x3 window');
subplot(1,4,3); 
imshow(Harris(noised_check, 5, 'harmonic'), []);
title('noised, 5x5 window');
subplot(1,4,4); 
imshow(Harris(noised_check, 9, 'harmonic'), []);
title('noised, 9x9 window');
figure;

% 1.3 Local Maxima
cow = rgb2gray(cow);
subplot(1,3,1); 
imshow(scene, []);
title('Input image');
subplot(1,3,2); 
imshow(Harris(scene, 5, 'harmonic'), []);
title('Harmonic mean');
subplot(1,3,3); 
imshow(Harris(scene, 5, 'maxima'), []);
title('Local Maxima(threshold=100000)');
figure;

imshow(Harris(cow, 5, 'maxima'), []);
title('Bonus: cow from material');
figure;

function h = Harris(img, window_size, detector)
    [filter_horizontal, filter_vertical] = imgradientxy(img);
    I_x = filter_horizontal;
    I_y = filter_vertical;
    IxIy = I_x.*I_y;
    
    supporting_window = ones(window_size, window_size);

    sum_IxIy = filter2(supporting_window, IxIy, 'same');
    I_x_sqr = I_x.*I_x;
    sum_IxSqr = filter2(supporting_window, I_x_sqr, 'same');
    I_y_sqr = I_y.*I_y;
    sum_IySqr = filter2(supporting_window, I_y_sqr, 'same');
    
    if strcmp(detector,"harmonic") || strcmp(detector,"maxima")
        threshold_input = zeros(size(img, 1), size(img, 2));
        output_harmonic = zeros(size(img, 1), size(img, 2));
        for u = 1 : size(img, 1)
            for v = 1 : size(img, 2)
                H = [sum_IxSqr(u,v) sum_IxIy(u,v) ; sum_IxIy(u,v) sum_IySqr(u,v)];
                harmonic_mean = det(H)/trace(H);
                output_harmonic(u, v) = harmonic_mean;
                if harmonic_mean > 100000 % threshold
                    threshold_input(u, v) = harmonic_mean;
                end
            end
        end
        h = output_harmonic;
        threshold_input = threshold_input/norm(threshold_input);
        
        if strcmp(detector,"maxima")
            local_maxima = zeros(size(img, 1), size(img, 2));
            for u = 1 : size(img, 1)
                for v = 1 : size(img, 2)
                    flag = 0;
                    for s = -2 : 2
                        for p = -2 : 2
                            if s == 0 && p == 0
                                continue;
                            end
                            if u+s >= 1 && v + p >= 1 && u+s <= size(img, 1) && v+p<=size(img, 2)
                            if threshold_input(u, v) <= threshold_input(u+s, v+p)
                                flag = 1;
                                break;
                            end
                            end
                        end
                        if flag == 1
                            break;
                        end
                    end
                    if flag == 0
                        local_maxima(u, v) = 1;
                    end
                end
            end
            h = local_maxima;
        end
        
    elseif strcmp(detector,"shi")
        output_shi = zeros(size(img, 1), size(img, 2));
        for u = 1 : size(img, 1)
            for v = 1 : size(img, 2)
                H = [sum_IxSqr(u,v) sum_IxIy(u,v) ; sum_IxIy(u,v) sum_IySqr(u,v)];
                eigen_min = min(eig(H));
                output_shi(u, v) = eigen_min;
            end
        end
        h = output_shi;
    end
end