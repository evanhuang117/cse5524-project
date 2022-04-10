% 1.

grayIm = imread('buckeyes_gray.bmp');
imagesc(grayIm);
axis('image');
colormap('gray');
imwrite(grayIm, 'buckeyes_gray.jpg');
pause;

rgbIm = imread('buckeyes_rgb.bmp');
imagesc(rgbIm);
axis('image');
imwrite(rgbIm, 'buckeyes_rgb.jpg');
pause;

% 2.
grayIm = rgb2gray(rgbIm);
imagesc(grayIm);
axis('image');
imwrite(grayIm, 'buckeyes_converted.jpg')
pause;

% 3.
zBlock = zeros(10, 10);
oBlock = ones(10, 10) * 255;
pattern = [zBlock oBlock; oBlock zBlock];
checkerIm = repmat(pattern, 5, 5);
imwrite(uint8(checkerIm), 'checkerIm.bmp');
Im = imread('checkerIm.bmp');
imagesc(Im)
colormap('gray')
axis('image');
