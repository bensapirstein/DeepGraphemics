import os
import pygame

class FontDatasetGenerator:
    def __init__(self):
        # 10 colors in RGB to choose from (back ground and font)
        self.colors = {'red': (220, 20, 60), 'orange': (255,165,0), 'yellow': (255,255,0),
                       'green': (0,128,0), 'cyan': (0,255,255), 'blue': (0,0,255),
                       'purple': (128,0,128), 'pink': (255,192,203), 'chocolate': (210,105,30),
                       'silver': (192,192,192)}
        # 3 Sizes of fonts
        self.sizes = {'small': 20, 'medium': 40, 'large': 60}
        self.all_fonts = self.get_filtered_fonts()
        self.letters = list(range(65, 91)) + list(range(97, 123))
        self.integers = list(range(48, 58))
        self.single_set = self.letters + self.integers
        self.words = []
        self.img_size = 128
        # generate words using a depth-first-search, max_w_len is the maximum length of the words, for now is one(single characters)
        self.max_w_len = 1
        self.font_dir = './dataset'

    def get_filtered_fonts(self):
        all_fonts = pygame.font.get_fonts()
        useless_fonts = ['notocoloremoji', 'droidsansfallback', 'gubbi', 'kalapi', 'lklug',
                         'mrykacstqurn', 'ori1uni', 'pothana2000', 'vemana2000', 'navilu',
                         'opensymbol', 'padmmaa', 'raghumalayalam', 'saab', 'samyakdevanagari']
        useless_fontsets = ['kacst', 'lohit', 'sam']
        filtered_fonts = [font for font in all_fonts if font not in useless_fonts]
        filtered_fonts = [font for font in filtered_fonts if not any(set in font for set in useless_fontsets)]
        return filtered_fonts

    def generate_words(self):
        self._dfs('', self.single_set)

    def _dfs(self, word, character_set):
        # Depth-first-search to generate words
        if len(word) == self.max_w_len:
            return
        for char in character_set:
            self.words.append(word + chr(char))
            self._dfs(word + chr(char), character_set)

    def generate_dataset(self):
        if not os.path.exists(self.font_dir):
            os.makedirs(self.font_dir)
        pygame.init()
        screen = pygame.display.set_mode((self.img_size, self.img_size))
        cnt = 0

        for word in self.words:
            print(word)
            for size_name, size_value in self.sizes.items():
                print(size_name)
                for font_color_name, font_color_value in self.colors.items():
                    for back_color_name, back_color_value in self.colors.items():
                        if not font_color_name == back_color_name:
                            for font in self.all_fonts:
                                cnt += 1
                                try:
                                    screen.fill(back_color_value)
                                    selected_font = pygame.font.SysFont(font, size_value)
                                    font_size = selected_font.size(word)
                                    rtext = selected_font.render(word, True, font_color_value, back_color_value)
                                    draw_x = self.img_size / 2 - (font_size[0] / 2.0)
                                    draw_y = self.img_size / 2 - (font_size[1] / 2.0)
                                    screen.blit(rtext, (draw_x, draw_y))

                                    img_name = f"{word}_{size_name}_{font_color_name}_{back_color_name}_{font}.png"
                                    img_path = os.path.join(self.font_dir, word, size_name, font_color_name, back_color_name, font)
                                    if not os.path.exists(img_path):
                                        os.makedirs(img_path)
                                    pygame.image.save(screen, os.path.join(img_path, img_name))
                                except Exception as e:
                                    print(f"Error: {word}, {size_name}, {font_color_name}, {back_color_name}, {font}")
                                    print(e)
                        else:
                            break
        print('finished')

dataset_generator = FontDatasetGenerator()
dataset_generator.generate_words()
dataset_generator.generate_dataset()
