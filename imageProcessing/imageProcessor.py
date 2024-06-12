import cv2
import numpy as np
import matplotlib.pyplot as plt

class ImageProcessor:
    def __init__(self, path: str) -> None:
        self.path = path
        self.img = cv2.imread(path)
        self.height, self.width, _ = self.img.shape
        self.img_cropped = None
        self.header_img = None
        self.list_of_pics = None
        
    def show_image(self) -> None:
        plt.imshow(self.img)
        plt.show()
        
    def process_image(self) -> list[np.ndarray]:
        self.crop_image_to_captcha()
        self.cut_captcha_pics()
        self.polishing_the_pics()
        return self.list_of_pics
    
    def crop_image_to_captcha(self) -> np.ndarray:

        kernel_size = self.width // 3

        img_eroded = cv2.erode(self.img, np.ones((kernel_size, kernel_size), np.uint8))
        img_dilated = cv2.dilate(self.img, np.ones((kernel_size, kernel_size), np.uint8))

        # Determine background seed areas where both eroded and dilated images match the original
        img_background_seed = ((img_eroded == self.img) & (img_dilated == self.img))[..., 0]

        img_potential_background = (self.img == 255)[..., 0]
        img_background = img_background_seed

        # Reconstruction (expand the background area iteratively until no further expansion is possible)
        while True:
            img_new_background = cv2.dilate(img_background.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
            img_new_background = img_new_background & img_potential_background
            # Break if the background does not change
            if np.array_equal(img_new_background, img_background):
                break
            img_background = img_new_background

        # Identify connected components in the non-background areas
        totalLabels, label_ids, values, centroids = cv2.connectedComponentsWithStats(~img_background.astype(np.uint8))

        # Largest connected component
        largest_index = np.argmax(values[1:, cv2.CC_STAT_AREA]) + 1
        selected_area = label_ids == largest_index

        xs = np.linspace(0, self.width - 1, self.width, dtype=int)
        ys = np.linspace(0, self.height - 1, self.height, dtype=int)

        xs = np.array([xs] * self.height)
        ys = np.array([ys] * self.width).T

        interesting_ys = ys[selected_area]

        min_y = np.min(interesting_ys)

        interesting_xs = xs[selected_area & (ys == min_y)]
        min_x = np.min(interesting_xs)

        max_y = np.max(interesting_ys)

        interesting_xs = xs[selected_area & (ys == max_y)]
        max_x = np.max(interesting_xs)

        # Crop the image to the bounding box of the largest connected component
        self.img_cropped = self.img[min_y:max_y, min_x:max_x]

        plt.imshow(self.img_cropped)
        plt.show()

    def cut_captcha_pics(self):
        # background areas
        captcha_background = (self.img_cropped == 255)[..., 0]
        height, width = captcha_background.shape

        # Define kernel sizes for erosion operations
        kernel_width = width * 9 // 10
        kernel_height = height // 2

        # horizontal and vertical erosion to identify lines
        captcha_eroded_horizontal = cv2.erode(captcha_background.astype(np.uint8), np.ones((1, kernel_width), np.uint8))
        captcha_eroded_vertical = cv2.erode(captcha_background.astype(np.uint8), np.ones((kernel_height, 1), np.uint8))

        # Extract horizontal and vertical lines from eroded images
        horizontal_lines = captcha_eroded_horizontal == 1
        vertical_lines = captcha_eroded_vertical == 1

        # Identify connected components in the horizontal and vertical lines
        total_labels_horizontal, label_ids_horizontal, values_horizontal, centroids_horizontal = cv2.connectedComponentsWithStats(horizontal_lines.astype(np.uint8))
        total_labels_vertical, label_ids_vertical, values_vertical, centroids_vertical = cv2.connectedComponentsWithStats(vertical_lines.astype(np.uint8))

        xs = np.linspace(0, width - 1, width, dtype=int)
        ys = np.linspace(0, height - 1, height, dtype=int)

        xs = np.array([xs] * height)
        ys = np.array([ys] * width).T

        # horizontal line
        average_ys_for_line = []
        for i in range(1, total_labels_horizontal):
            line = label_ids_horizontal == i
            interesting_ys = ys[line]
            average_ys_for_line.append(np.round(np.mean(interesting_ys)).astype(int))

        # vertical line
        average_xs_for_line = []
        for i in range(1, total_labels_vertical):
            line = label_ids_vertical == i
            interesting_xs = xs[line]
            average_xs_for_line.append(np.round(np.mean(interesting_xs)).astype(int))

        average_ys_for_line.sort()
        average_xs_for_line.sort()

        average_ys_for_line = average_ys_for_line[:-2]

        # header region
        header_start_y = average_ys_for_line[0]
        header_end_y = average_ys_for_line[1]
        header_start_x = average_xs_for_line[0]
        header_end_x = average_xs_for_line[-1]

        self.header_img = self.img_cropped[header_start_y:header_end_y, header_start_x:header_end_x]

        # --- Pictures ---
        # Initialize a list to store pieces of the image
        self.list_of_pics = []
        for i in range(1, len(average_ys_for_line) - 1):
            for j in range(len(average_xs_for_line) - 1):
                start_y = average_ys_for_line[i]
                end_y = average_ys_for_line[i + 1]
                start_x = average_xs_for_line[j]
                end_x = average_xs_for_line[j + 1]
                
                piece = self.img_cropped[start_y:end_y, start_x:end_x]
                self.list_of_pics.append(piece)
                
        for pic in self.list_of_pics:
            plt.imshow(pic)
            plt.show()

       
            
    def polishing_the_pics(self):


        # Process each piece to remove background lines
        processed_pieces = []
        for piece in self.list_of_pics:
            piece_height, piece_width = piece.shape[:2]
            
            piece_background = (piece == 255)[..., 0]
            
            # Perform horizontal and vertical erosion on each piece
            horizontal_lines_piece = cv2.erode(piece_background.astype(np.uint8), np.ones((1, piece_width * 9 // 10), np.uint8))
            vertical_lines_piece = cv2.erode(piece_background.astype(np.uint8), np.ones((piece_height * 9 // 10), np.uint8))
            
            # Check if horizontal and vertical lines exist
            horizontal_line_exists = np.any(horizontal_lines_piece, axis=1)
            vertical_line_exists = np.any(vertical_lines_piece, axis=0)
            
            # Select y and x coordinates where no lines exist
            selected_ys = np.where(~horizontal_line_exists)[0]
            selected_xs = np.where(~vertical_line_exists)[0]
            
            # print(horizontal_line_exists)
            # print(selected_ys)
            # print(selected_xs)
            
            # Crop the piece to remove lines
            processed_piece = piece[selected_ys, :][:, selected_xs]
            processed_pieces.append(processed_piece)

        # Display each processed piece
        for piece in processed_pieces:
            plt.imshow(piece)
            plt.show()