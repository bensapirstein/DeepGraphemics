import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

def main():
    # Define the dimensions of the video frames
    frame_width = 1080
    frame_height = 1080

    # Define the duration of the transition
    transition_duration = 120

    # Define the font and size to use
    font_size = 800
    font_path = "NotoSansPhoenician-Regular.ttf"
    title_font_size = 100
    title_font_path = "Alice-Regular.ttf"

    # Define the Phoenician alphabet letters and their corresponding titles
    letters = ["𐤀", "𐤁", "𐤂", "𐤃", "𐤄", "𐤅", "𐤆", "𐤇", "𐤈", "𐤉", "𐤊", "𐤋", "𐤌", "𐤍", "𐤎", "𐤏", "𐤐", "𐤑", "𐤒", "𐤓", "𐤔", "𐤕"]
    titles = ["aleph", "bet", "gimmel", "dalet", "he", "waw", "zayin", "het", "tet", "yod", "kaph", "lamed", "mem", "nun", "samekh", "ayin", "pe", "tsadi", "qoph", "resh", "shin", "taw"]

    # Create a blank image to serve as the background
    background_color = (0, 0, 0)  # White
    background = Image.new("RGB", (frame_width, frame_height), background_color)

    # Define the video output parameters
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    output_path = "letters_animation.mp4"
    video_writer = cv2.VideoWriter(output_path, fourcc, 60, (frame_width, frame_height))

    # Load the font
    font = ImageFont.truetype(font_path, font_size)
    title_font = ImageFont.truetype(title_font_path, title_font_size)

    def calculate_letter_position(letter, y_offset=-200):
        letter_bbox = font.getbbox(letter)
        letter_width = letter_bbox[2] - letter_bbox[0]
        letter_height = letter_bbox[3] - letter_bbox[1]

        letter_x = (frame_width - letter_width) // 2
        letter_y = (frame_height - letter_height) // 2 + y_offset
        return (letter_x, letter_y, letter_width, letter_height)

    def calculate_title_position(title, letter_y):
        title_bbox = title_font.getbbox(title)
        title_width = title_bbox[2] - title_bbox[0]
        title_height = title_bbox[3] - title_bbox[1]

        title_x = (frame_width - title_width) // 2
        return (title_x, letter_y)

    def ease_in_out(t, p=0.2):
        if t <= 0.5:
            return pow(t * 2, p) / 2
        else:
            return 1 - pow(2 - t * 2, p) / 2

    def ease_in_elastic(t, p=0.4):
        return pow(2, 10 * (t - 1)) * np.sin((t - p / 4) * (2 * np.pi) / p)

    def ease_out_elastic(t, p=0.4):
        return pow(2, -10 * t) * np.sin((t - p / 4) * (2 * np.pi) / p) + 1

    def easeOutInBounce(x):
        if x < 0.5:
            return 0.5 * easeOutBounce(x * 2)
        else:
            return 0.5 * (easeInBounce((x * 2) - 1) + 1)

    def easeOutBounce(x):
        n1 = 7.5625
        d1 = 2.75

        if x < 1 / d1:
            return n1 * x * x
        elif x < 2 / d1:
            x -= 1.5 / d1
            return n1 * x * x + 0.75
        elif x < 2.5 / d1:
            x -= 2.25 / d1
            return n1 * x * x + 0.9375
        else:
            x -= 2.625 / d1
            return n1 * x * x + 0.984375

    def easeInBounce(x):
        return 1 - easeOutBounce(1 - x)

    def calculate_transition_position(t, frame_width, letter_width, letter_y):
        transition_x = np.interp(ease_in_out(t), [0, 1], [frame_width, -letter_width])
        transition_y =  letter_y
        return int(transition_x), int(transition_y)

    def draw_transition_frame(transition_frame_draw, letter, title, transition_x, transition_y):
        # Draw the letter with the transition position
        transition_frame_draw.text((transition_x, transition_y), letter, font=font, fill=(255, 255, 255))  # Black

        # Add a title with the name of the letter
        transition_frame_draw.text((title_x, title_y), title, font=title_font, fill=(255, 255, 255))  # Black

    # Iterate over each letter and create frames
    for i, letter in tqdm(enumerate(letters)):
        # Calculate the letter position
        letter_x, letter_y, letter_width, letter_height = calculate_letter_position(letter)

        # Calculate the title position
        title_x, title_y = calculate_title_position(titles[i], letter_y)

        # Generate the transition animation frames
        for t in range(transition_duration):
            # Calculate the transition position
            transition_x, transition_y = calculate_transition_position(t / transition_duration, frame_width, letter_width, letter_y)

            # Create a blank frame
            transition_frame = background.copy()

            # Create an image draw object for the transition frame
            transition_frame_draw = ImageDraw.Draw(transition_frame)

            # Draw the transition frame
            draw_transition_frame(transition_frame_draw, letter, titles[i], transition_x, transition_y)

            # Convert the PIL image to a NumPy array
            transition_frame_np = np.array(transition_frame)

            # Write the transition frame to the video file
            video_writer.write(transition_frame_np)

    # Release the video writer
    video_writer.release()

if __name__ == '__main__':
    main()
