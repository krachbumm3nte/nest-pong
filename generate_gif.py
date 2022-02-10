#%%
import numpy as np
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 
from PIL import ImagePalette
import pickle
import matplotlib.pyplot as plt
import gzip
import os
import io
import glob

background_hex = "#dcdcdc"

background_color = (220, 220, 220) # grey
black = (0, 0, 0)
white = (255, 255, 255)
left_color = (0, 0, 255) # blue
right_color = (255, 0, 0) # red

colors = [background_color, black, white, left_color, right_color]
colormap_rgb = []
for color in colors:
    for val in color:
        colormap_rgb.append(val)

colormap_rgb[1] = 0

palette = ImagePalette.ImagePalette(mode="RGB", palette=colormap_rgb, size=15)

BALL_RAD = 3
PADDLE_LEN = 20
PADDLE_WID = 5

# final output image size
IMAGE_SIZE = np.array([800, 640])


FIELD_SIZE = np.array([32, 20]) # original size of the playing field inside the simulation
FIELD_IMAGE = FIELD_SIZE * 16 # Field size (in px) in the final image

HEATMAP_SCALE = 8
HEATMAP_SIZE = np.array([FIELD_SIZE[1], FIELD_SIZE[1]]) * HEATMAP_SCALE

PLOT_INTERVAL = 50

HEATMAP_INTERVAL = 50


out_folder = "images"

font_loc = "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf"

score_font = ImageFont.truetype(font_loc, 50, encoding="unic")
vs_font = ImageFont.truetype(font_loc, 30, encoding="unic")

def scale_position(pos):
    return (int(FIELD_IMAGE[0]*pos[0]/1.6), int(FIELD_IMAGE[1]*pos[1]))

def grayscale_to_heatmap(im, min_val, max_val):
    out = np.zeros((20, 20, 3), dtype=np.uint8)

    span = max_val - min_val +0.0001 # add miniscule offsett to avoid zero division error for uniform weight matrix
    for i in range(20):
        for j in range(20):
            val = (im[i,j] - min_val) / span
            out[i,j, :] = 255, int(255*val), int(255 * val)   
    return out

# %%

folder = "../2_player_pong/2022-02-09-12-49-50/"

with open(os.path.join(folder, "gamestate.pkl"), 'rb') as f:
    game_data = pickle.load(f)


with gzip.open(os.path.join(folder, "data_right.pkl.gz"), 'r') as f:
    data = pickle.load(f)
    rewards_right = data[0]
    weights_right = data[2]
    
with gzip.open(os.path.join(folder, "data_left.pkl.gz"), 'r') as f:
    data = pickle.load(f)
    rewards_left = data[0]
    weights_left = data[2]

min_r, max_r = np.min(weights_right), np.max(weights_right)
min_l, max_l = np.min(weights_left), np.max(weights_left)

rewards_left = [np.mean(x) for x in rewards_left]
rewards_right = [np.mean(x) for x in rewards_right]


if os.path.exists(out_folder):
    print("output folder already exists, aborting!")
    exit()
else:
    os.mkdir(out_folder)


max_runs = len(game_data)

i = 0
speedup = 2

while i < max_runs:

    background = Image.new("RGB", tuple(IMAGE_SIZE), background_color)
    draw = ImageDraw.Draw(background)


    # Pong game playing field
    playing_field = np.zeros((FIELD_IMAGE[0], FIELD_IMAGE[1], 3), dtype=np.uint8)

    # scale up floating point positions from the simulation to pixel values within the image
    positions = [scale_position(x) for x in game_data[i][:3]]

    # draw Ball in white
    x, y = positions[0]
    playing_field[x-BALL_RAD:x+BALL_RAD, y-BALL_RAD:y+BALL_RAD, :] = 255

    # draw left paddle in blue
    x,y = positions[1]
    playing_field[x:x+PADDLE_WID,y-PADDLE_LEN:y+PADDLE_LEN] = left_color

    # draw right paddle in red
    x,y = positions[2]
    playing_field[x-1:x+PADDLE_WID,y-PADDLE_LEN:y+PADDLE_LEN] = right_color


    # prepare and paste playing field into image
    playing_field = np.swapaxes(playing_field, 0, 1)
    playing_field = Image.fromarray(playing_field)
    background.paste(playing_field, (144, 72))


    if i % PLOT_INTERVAL == 0 or speedup > 10:
        # only generate reward plot and heatmaps every PLOT_INTERVAL iterations
        print(f"plotting run: {i}")
        plt.close()

        # Right player heatmap (red)
        heatmap_r = grayscale_to_heatmap(weights_right[i], min_r, max_r)
        heatmap_r = np.kron(heatmap_r, np.ones((HEATMAP_SCALE, HEATMAP_SCALE, 1), np.uint8))
        heatmap_r = Image.fromarray(heatmap_r)

        # Left player heatmap (blue)
        heatmap_l = grayscale_to_heatmap(weights_left[i], min_l, max_l)
        heatmap_l[:,:,[0,2]] = heatmap_l[:,:,[2,0]] # swap indices in the RGB dimension to get blue heatmap 
        heatmap_l = np.kron(heatmap_l, np.ones((HEATMAP_SCALE, HEATMAP_SCALE, 1), np.uint8))
        heatmap_l = Image.fromarray(heatmap_l)

        # Plot rewards for both players
        fig = plt.figure(facecolor=background_hex)
        ax = plt.axes()
        ax.set_facecolor(background_hex)
        plt.rcParams["figure.figsize"] = [4.4, 2.8]    
        plt.rcParams["figure.autolayout"] = True
        plt.xlabel("iteration")
        plt.ylabel("mean reward")
        ax.get_xticklabels(False)
        ax.plot(rewards_right[:i], "r")
        ax.plot(rewards_left[:i], "b")
        ax.set_ylim(0,1.0)

        x_min = 0 if i < 2000 else i-2000
        ax.set_xlim(x_min, i)
        
        buf = io.BytesIO()
        fig.savefig(buf)
        buf.seek(0)

        reward_plot = Image.open(buf)
    
    background.paste(heatmap_r, (576, 420))
    draw.text((600, 580), "weights", (0,0,0), vs_font)

    background.paste(heatmap_l, (64, 420))  
    draw.text((82, 580), "weights", (0,0,0), vs_font)

    if i > PLOT_INTERVAL:
        # only start plotting rewards after the first few iterations
        background.paste(reward_plot, (240, 420))

    draw.text((150, 20), "clean input", left_color, vs_font)
    draw.text((375, 20), "VS", (0,0,0), vs_font)
    draw.text((500, 20), "noisy input", right_color, vs_font)

    
    l_score, r_score = game_data[i][3]
    draw.text((60, 200), str(l_score), black, score_font)
    draw.text((720, 200), str(r_score), black, score_font)

    draw.text((10,5), "run:", black, vs_font)
    draw.text((10,40), str(i), black, vs_font)

    draw.text((700, 10), f"speed:", black, vs_font)
    draw.text((700, 40), str(speedup)+'x', black, vs_font)

    background.save(os.path.join(out_folder, f"img_{str(i).zfill(6)}.png"))

    if i in range(150,400) or i in range(4400, 4600):
        speedup = 10
    elif i in range(400, 600) or i in range(4000, 4400):
        speedup = 20
    elif i in range(600, 4000):
        speedup = 50
    else:
        speedup = 2
    
    i+=speedup
# %%
# filepaths
fp_in = os.path.join(out_folder, "img_*.png")
fp_out = "pang.gif"

img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
img.save(fp=fp_out, format='GIF', append_images=imgs,
         save_all=True, duration=40, loop=0)

# %%
