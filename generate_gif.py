import numpy as np
from PIL import Image
import pickle
import matplotlib.pyplot as plt
import sys

scaling_factor = 2
frame_color = (196, 135, 102) # grey
paddle_color = (0, 17, 255) # blue

hit_color = (0, 255, 0) # green
reset_color = (255, 0, 0) # red

IMAGE_SIZE = 256
FIELD_SIZE = 128
MATRIX_SIZE = 64



def add_margin(pil_img, thickness, color):
    width, height = pil_img.size
    new_width = width + 2* thickness
    new_height = height + 2 * thickness
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (thickness, thickness))
    return result



def create_canvas(size):
    background = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), frame_color)
    playing_field = Image.new("RGB", (FIELD_SIZE, FIELD_SIZE), hit_color)
    background.paste(playing_field, (MATRIX_SIZE, 20))



if __name__=="__main__":

    with open("./positions.pkl", 'rb') as f:
        foo = pickle.load(f)

    out_array = np.zeros((len(foo), 33, 33, 3), dtype=np.uint8)

    misses = 0
    hits = 0

    for i, positions in enumerate(foo):
        print(positions)
        if positions == ([1, 1], [0, 0]):
            print("hit")
            hits +=1
            out_array[i,:,:] = hit_color
        elif positions == ([0, 0], [0, 0]):
            out_array[i,:,:] = reset_color
            misses +=1
        
        for x,y in positions:

            out_array[i,y,x,:] = 255

    # upscaling
    out_array = np.kron(out_array, np.ones((1, scaling_factor, scaling_factor, 1), dtype=np.uint8))



    imgs = [Image.fromarray(img) for img in out_array]


    imgs = [add_margin(img, 4, frame_color) for img in imgs]

    imgs[0].save("array.gif", save_all=True, append_images=imgs[1:], duration=25, loop=0)
