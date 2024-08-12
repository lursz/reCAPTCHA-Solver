import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

class ImageProcessor:
    def __init__(self, path: str) -> None:
        self.path = path
        self.img = cv2.imread(path)
        self.img_cropped = None
        self.header_img = None
        self.list_of_pics = None
        self.height = None 
        self.width = None
        
    def show_image(self) -> None:
        plt.imshow(self.img)
        plt.show()
        
    def process_captcha_image(self, output_folder: str) -> list[np.ndarray]:
        self.crop_image_to_captcha()
        self.cut_captcha_pics()
        self.polishing_the_pics()
        self.save_all_pics(output_folder)
        return self.list_of_pics
    
    def crop_image_to_captcha(self) -> np.ndarray:
        self.height, self.width, _ = self.img.shape

        # Morphological operations
        kernel_size = self.width // 3
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        img_eroded = cv2.erode(self.img, kernel)
        img_dilated = cv2.dilate(self.img, kernel)

        # Determine background seed areas where both eroded and dilated images match the original
        img_background_seed = np.all((img_eroded == self.img) & (img_dilated == self.img), axis=-1)
        img_potential_background = np.all(self.img == 255, axis=-1)
        img_background = img_background_seed

        # Reconstruction (expand the background area iteratively until no further expansion is possible)
        while True:
            dilatated_background = cv2.dilate(img_background.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
            img_new_background = dilatated_background & img_potential_background
            # Break if the background does not change
            if np.array_equal(img_new_background, img_background):
                break
            img_background = img_new_background
        
        # Identify connected components in the non-background areas
        img_foreground = ~img_background
        _, label_ids, values, _ = cv2.connectedComponentsWithStats(img_foreground.astype(np.uint8))

        # Select largest connected component 
        largest_index = np.argmax(values[1:, cv2.CC_STAT_AREA]) + 1
        selected_area = label_ids == largest_index

        y_coords, x_coords = np.where(selected_area)
        min_x, max_x = np.min(x_coords), np.max(x_coords)
        min_y, max_y = np.min(y_coords), np.max(y_coords)

        # Crop the image to the bounding box of the largest connected component
        self.img_cropped = self.img[min_y:max_y, min_x:max_x]

    def cut_captcha_pics(self):
        # background areas
        captcha_background = (self.img_cropped == 255)[..., 0]
        height, width = captcha_background.shape

        # Define kernel sizes for erosion operations
        kernel_width = width * 9 // 10
        kernel_height = height // 2

        # horizontal and vertical erosion to identify lines
        horizontal_kernel = np.ones((1, kernel_width), np.uint8)
        vertical_kernel = np.ones((kernel_height, 1), np.uint8)

        captcha_eroded_horizontal = cv2.erode(captcha_background.astype(np.uint8), horizontal_kernel)
        captcha_eroded_vertical = cv2.erode(captcha_background.astype(np.uint8), vertical_kernel)

        # Extract horizontal and vertical lines from eroded images
        horizontal_lines = captcha_eroded_horizontal == 1
        vertical_lines = captcha_eroded_vertical == 1

        # Identify connected components in the horizontal and vertical lines
        total_labels_horizontal, label_ids_horizontal, _, _ = cv2.connectedComponentsWithStats(horizontal_lines.astype(np.uint8))
        total_labels_vertical, label_ids_vertical, _, _ = cv2.connectedComponentsWithStats(vertical_lines.astype(np.uint8))

        xs = np.linspace(0, width - 1, width, dtype=int)
        ys = np.linspace(0, height - 1, height, dtype=int)
        xs = np.array([xs] * height)
        ys = np.array([ys] * width).T

        # horizontal line
        average_ys_for_line = [np.round(np.mean(ys[label_ids_horizontal == i])).astype(int) for i in range(1, total_labels_horizontal)]

        # vertical line
        average_xs_for_line = [np.round(np.mean(xs[label_ids_vertical == i])).astype(int) for i in range(1, total_labels_vertical)]

        average_ys_for_line.sort()
        average_xs_for_line.sort()

        average_ys_for_line = average_ys_for_line[:-2]

        # header region
        header_start_y = average_ys_for_line[0]
        header_end_y = average_ys_for_line[1]
        header_start_x = average_xs_for_line[0]
        header_end_x = average_xs_for_line[-1]

        self.header_img = self.img_cropped[header_start_y:header_end_y, header_start_x:header_end_x]

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
       
            
    def polishing_the_pics(self) -> None:
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
            
            # Crop the piece to remove lines
            processed_piece = piece[selected_ys, :][:, selected_xs]
            processed_pieces.append(processed_piece)

    def save_all_pics(self, path: str) -> None:
        if not os.path.exists(path):
            os.makedirs(path)
            
        cv2.imwrite(f"{path}/header.png", self.header_img)
        for i, pic in enumerate(self.list_of_pics):
            cv2.imwrite(f"{path}/pic_{i}.png", pic)