import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import stats
import numpy as np
import math
import cv2
import os

class Point:
    """Abstraction for an (x, y) coordinate pair

    # Arguments:
        x: any integer or floating point number
        y: any integer or floating point number

    # Example:

    ```python

        # Instantiate some points
        >>> point = Point(3, 5)
        >>> another_point = Point(7, 10)
        >>> point()
        (3, 5)

        # Calculate the slope between two points
        >>> slope = Point.slope(point, another_point)
        >>> slope
        1.25

    ```
    """
    def __init__(self, x, y):
        self._x = x
        self._y = y

    def __call__(self):
        """Returns a tuple containt an (x, y) pair
        """
        return (self.x, self.y)

    @staticmethod
    def slope(one, two):
        """Calculate the slope between two Point instances

        # Arguments:
            one: instance of Point
            two: instance of Point

        # Returns:
            The slope between two points. If the delta between
            the `x` coordinates is inf, nan, or 0.0, returns 0.0
        """
        x_delta = np.subtract(two.x, one.x, dtype=np.float32)
        if x_delta in {np.nan, np.inf, np.float32(0)}:
            return np.float32(0)
        y_delta = np.subtract(two.y, one.y)

        return np.divide(y_delta, x_delta, dtype=np.float32)

    @property
    def x(self):
        """Returns x coordinate"""
        return self._x

    @property
    def y(self):
        """Returns y coordinate"""
        return self._y

class Lane:
    """Abstraction for a lane

    A lane is composed of a bunch of small lines represented by (x1, y1)
    and (x2, y2) coordinate pairs.

    # Arguments:
        _points: list of Point instances
        min_x: threshold for the smallest x value that the lane can contain.
        max_x: threshold for the largest x value that the lane can contain.
            This is useful for filtering out lines that do not fall into a
            a specific window that we have specified.

    # Example:

        # Create an instance of a Lane
        >>> left = Lane(min_x=0, max_x=100)
        >>> point = Point(4, 4)
        >>> another_point = Point(120, 3)

        # Add the points to the Lane
        >>> left.points = [point, another_point]

        # Only one point is added since the second point
        # was beyond the threshold (120 > 100).
        >>> left.points
        [<__main__.Point at 0x1155ae240>]

    """
    def __init__(self, min_x, max_x):
        self._points = []
        self.min_x = min_x
        self.max_x = max_x

    @property
    def points(self):
        """Returns all the points in the lane"""
        return self._points

    @points.setter
    def points(self, points):
        """Add points to the lane

        The points are only added if they fall within the boundary
        specified by `min_x` and `max_x`.

        # Returns:
            None
        """
        for point in points:
            if np.greater(point.x, self.min_x) and np.less(point.x, self.max_x):
                self._points.append(point)
            else:
                continue

        return None

    @property
    def xs(self):
        """Return all x values for the points stored in the Lane

        # Example:

            >>> lane = Lane(0, 10)
            >>> lane.points = [Point(1, 2), Point(3, 4)]

            >>> lane.xs
            [1, 3]

        """
        return [point.x for point in self.points]

    @property
    def ys(self):
        """Return all x values for the points stored in the Lane

        # Example:

            >>> lane = Lane(0, 10)
            >>> lane.points = [Point(1, 2), Point(3, 4)]

            >>> lane.ys
            [2, 4]

        # Returns:
            A list of integers
        """
        return [point.y for point in self.points]

    def fit(self, dimension=1):
        """Fits a line to the points in the lane

        Given a list of x-coordinates and y-coordinates,
        will fit a line that runs through those points.

        Returns:
            Two tuples representing the starting and ending vertices
            of the line to draw.
        """
        slope, intercept = np.polyfit(self.xs, self.ys, dimension)
        function = np.poly1d([slope, intercept])

        left_min = min(min(self.xs), self.min_x)
        right_max = max((max(self.xs)), self.max_x)

        one = left_min, int(function(left_min))
        two = right_max, int(function(right_max))

        return (one, two)

class Pipeline:
    """Abstraction for the pipeline

    The pipeline contains the layers that apply the various
    transformations on the images.

    # Arguments:
        layers: list of instance of transformation classes
        images: list of numpy representations of the images

    # Example:

        >>> pipeline = Pipeline()
        >>> pipeline.add(Image(path='some/path/to/image'))

        # The pipeline can now apply a grayscale transformation
        # to the images in the pipline.
        >>> pipeline.add(Gray())

        # Applies all the transformations in `layers` to `images`.
        >>>> pipeline.run()

    """
    def __init__(self):
        self.layers = []
        self.images = []

    def add(self, layer):
        """Adds a layer to the pipeline"""
        return self.layers.append(layer)

    def add_images(self, images, base_path='test_images/', single_image=False, array=False):
        """Adds a single image or batch of images to the pipeline.

        # Arguments:
            matrix: if True, will directly add the numpy image array to the pipeline.
        """

        if array:
            self.images.append(images)
            return images

        if type(images) is not list:
            images = list(images)

        for path in images:
            image = mpimg.imread(base_path + path)
            self.images.append(image)

        if single_image:
            return self.images[0]


    def run(self, single_image=False):
        """Applies the transformations to the images in the pipeline

        # Arguments:
            single_image: if True, will run the transformations only
                on one image. If there are multiple images in the pipeline,
                it will the first one.
        """
        if not single_image:
            for index, image in enumerate(self.images):
                initial_image = image

                for layer in self.layers:
                    image = layer(image, initial_image=initial_image, key=index)
        else:
            image = self.images[0]
            initial_image = image
            for layer in self.layers:
                image = layer(image, initial_image=initial_image)
            return image

class Gray:
    """Grayscales an image

    # Arguments:
        save: if True, writes the image to disk

    # Returns:
        A grayscaled image
    """
    def __call__(self, image, save=False, **kwargs):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        if save and 'key' in kwargs:
            mpimg.imsave(f"test_images_output/image-{kwargs['key']}/gray.jpg", image, cmap="gray")
        return image

class Canny:
    """Applies Canny edge detection transform"""
    def __init__(self, low_threshold=50, high_threshold=150):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def __call__(self, image, save=False, **kwargs):
        image = cv2.Canny(image, self.low_threshold, self.high_threshold)
        if save and 'key' in kwargs:
            mpimg.imsave(f"test_images_output/image-{kwargs['key']}/canny.jpg", image, cmap='gray')
        return image

class Gaussian:
    """Applies Gaussian blurring

    # Arguments:
        kernel_size: how blurry the image should be. Must be an odd number.
    """
    def __init__(self, kernel_size=3):
        if kernel_size is not 1 and kernel_size % 2 == 0:
            raise ValueError('`kernel_size` must be an odd number')
        self.kernel_size = kernel_size

    def __call__(self, image, save=False, **kwargs):
        image = cv2.GaussianBlur(image, (self.kernel_size, self.kernel_size), 0)
        if save and 'key' in kwargs:
            mpimg.imsave(f"test_images_output/image-{kwargs['key']}/blur.jpg", image, cmap='gray')
        return image

class Region:
    """Applies a mask to an image as defined by vertices

    # Arguments:
        vertices: numpy 2D array of vertices of a polygon

    # Returns:
        The image with the mask applied.
    """
    def __init__(self, vertices):
        self.vertices = vertices

    def __call__(self, image, save=False, **kwargs):
        mask = np.zeros_like(image)

        if len(image.shape) > 2:
            channel_count = image.shape[2]
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255

        cv2.fillPoly(mask, self.vertices, ignore_mask_color)

        image = cv2.bitwise_and(image, mask)
        if save and 'key' in kwargs:
            mpimg.imsave(f"test_images_output/image-{kwargs['key']}/region.jpg", image, cmap='gray')
        return image

class HoughTransform:
    """Applies the Hough transform"""
    def __init__(self, rho=2, theta=np.pi/180, threshold=1, min_line_len=5, max_line_gap=30, extrapolate=False):
        self.rho = rho
        self.theta = theta
        self.threshold = threshold
        self.min_line_len = min_line_len
        self.max_line_gap = max_line_gap
        self.extrapolate = extrapolate

    def __call__(self, image, **kwargs):
        self.kwargs = kwargs
        lines = cv2.HoughLinesP(image, self.rho, self.theta, self.threshold, np.array([]),
                                minLineLength=self.min_line_len, maxLineGap=self.max_line_gap)
        line_img = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

        if self.extrapolate:
            self.draw_extrapolated_lines(line_img, lines)
        else:
            self.draw_lines(line_img, lines)

        return line_img

    def draw_lines(self, img, lines, color=[255, 0, 0], thickness=2):
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)

    def draw_extrapolated_lines(self, image, lines, color=[255, 0, 0], thickness=10, save=False):
        """This is where the magic happens!

        # Arguments:
            image: image to apply the transform on
            lines: Hough lines generated by cv2.HoughLinesP
            color: RGB value of line color
            thickness: thickness of line
            save: if True, saves the image to disk

        # Returns:
            Image with extrapolated lines drawn
        """
        left_lane = Lane(min_x=100, max_x=450)
        right_lane = Lane(min_x=505, max_x=940)

        for line in lines:
            for x1,y1,x2,y2 in line:
                one, two = np.array([Point(x1, y1), Point(x2, y2)])
                slope = Point.slope(one, two)

                # In an image coordinate system, (0, 0) is in the top left
                # corner and thus a positive slope in the Cartesian plane
                # will actually have a negative slope in the image coordinate
                # system.
                if slope < 0:
                    left_lane.points = [one, two]
                if slope > 0:
                    right_lane.points = [one, two]

        # Fit the points in the lines to get the vertices to draw the extrapolated lines
        start_one, end_one = left_lane.fit()
        start_two, end_two = right_lane.fit()

        # Draw the left and right lane lines.
        cv2.line(image, start_one, end_one, color, thickness)
        cv2.line(image, start_two, end_two, color, thickness)

        if save and 'key' in self.kwargs:
            mpimg.imsave(f"test_images_output/image-{self.kwargs['key']}/hough.jpg", image)

        return image

class WeightMask:
    """Applies the extrapolated lines to the original image"""
    def __call__(self, image, α=0.8, β=1., λ=0, save=False, **kwargs):
        image = cv2.addWeighted(kwargs['initial_image'], α, image, β, λ)
        if save and 'key' in kwargs:
            mpimg.imsave(f"test_images_output/image-{kwargs['key']}/final.jpg", image)
        return image


pipeline = Pipeline()

image_paths = ["solidWhiteCurve.jpg"]
pipeline.add_images(image_paths)

pipeline.add_images(img)
pipeline.add(Gray())
pipeline.add(Gaussian(kernel_size=5))
pipeline.add(Canny())
pipeline.add(Region(vertices=np.array([[[100, 560], [445, 325], [505, 325], [940, 560]]])))
pipeline.add(HoughTransform(extrapolate=True))
pipeline.add(WeightMask())

output_image = pipeline.run(single_image=True)

plt.imshow(output_image)
plt.show()

