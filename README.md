# **Finding Lane Lines on the Road**
---

The goals / steps of this project are the following:
1. Make a pipeline that finds lane lines on the road
2.Reflect on your work in a written report
---

### Reflection

## 1. Describe your pipeline.

My pipeline consisted of seven steps:
#### 1. Read in the original image
This was the easiest part. Aside from importing all of the libraries, reading in an image was quite easy to accomplish.
[Original](./test_images_output/final.jpg)

#### 2. Grayscale the image
Before applying any of other transformations, the image must be gray-scaled to make it easier for Canny edge detection to be able to find gradients between areas of dark and light pixels.
[Grayscaled](./test_images_output/gray.jpg)
#### 3. Apply a Gaussian blur
To reduce noise in the picture we apply a blur that allows nearby pixels to blend in together and facilitate the Canny edge algorithm.

[Blurred](./test_images_output/blur.jpg)
#### 4. Apply Canny Edge detection
#### 5. Apply a Region Mask
#### 6. Apply the Hough transform
#### 7. Apply Final Mask

## 2. Identify potential shortcomings with your current pipeline
Completetely fails the challenge video

One potential shortcoming would be what would happen when ...

Another shortcoming could be ...


## 3. Suggest possible improvements to your pipeline

A possible improvement would be to ...

Another potential improvement could be to ...
