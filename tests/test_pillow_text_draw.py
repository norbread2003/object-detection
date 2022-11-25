import numpy as np
from PIL import ImageDraw, Image, ImageFont

im = Image.open("../img_lights.png")
im_size = 1000
im = im.crop((0, 0, im_size, im_size))
font_ratio = 0.05
fnt = ImageFont.truetype("Roboto-Bold.ttf", int(im_size * font_ratio))
d = ImageDraw.Draw(im)

rec_upper_left = np.array((0, 0))
rec_size = im_size * font_ratio
rec_outline_width = int(rec_size * 0.05)
text_off = np.array((rec_size * 0.2, -rec_size * 0.07))
text_loc = rec_upper_left + text_off
rec_bottom_right = rec_upper_left + rec_size
d.rectangle(
    (rec_upper_left[0], rec_upper_left[1], rec_bottom_right[0], rec_bottom_right[1]),
    fill="white", outline="black", width=rec_outline_width)
d.text(text_loc.tolist(), "0", fill="black", font=fnt)

im.show()
