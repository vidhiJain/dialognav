import numpy as np
import matplotlib.pyplot as plt


map_grid_custom_colors = { 
        -1: [200,200,200],                 
        0: [0,0,0],
        1: [0,0,255],
        2: [63,165,76 ], 
        3: [51, 153,245],
        4: [98, 67, 67],
        5: [255, 196, 75], 
        6: [140, 40, 40],
        7: [255, 0, 0],
        8: [0, 0, 255],
        9: [255,255, 255],
        10: [70, 70, 70]
    }

visible_grid_custom_colors = {
        0: [70,70,70],   # [200,200,200],
        1: [255, 255, 255],
        8: [0, 0, 255]
    }


def get_image_from_3D_absolute_map(location_map):
    """
    Creates an image of the numpy array based player map 
    for visualization

    Parameters:
        location_map: 3D numpy array with indices in [-1, 10]

    Returns:
        blended_image: 4D numpy array of size [y_range(e.g. 3), H, W, C]
    """
    blended_image = np.array([[[map_grid_custom_colors[int(val)] for val in row_x]
                                for row_x in z_x_grid] for z_x_grid in player_map], dtype='B')

    blended_image = blended_image.astype(int)

    return blended_image


def get_image_from_3D_visible_map(location_map, visibility_map, mask_ratio=0.45):
    """
    Alpha blending location map with player's visibility mask 
    
    Parameters:
        location map : 3D array
        visible map : 2D array

    Return: 
        blended_image : 4D array (with color channels)
    """
    
    location_image = np.array([[[map_grid_custom_colors[int(val)] for val in row_x] for row_x in z_x_grid] for z_x_grid in location_map], dtype='B')
    visible_image = np.array([[visible_grid_custom_colors[val] for val in row] for row in self.visible_grid], dtype='B')

    blended_image = mask_ratio * location_image + (1-mask_ratio) * np.stack([visible_image]*self.range_y, axis=0)
    blended_image = blended_image.astype(int)

    return blended_image


def plot_grid_2D(a, custom_colors, name, text=None):
    """
    matplotlib plot of location map with custom colors
    
    Parameters:
        a: 2D array representing the grid
        custom_colors : dict mapping each elements of array to RGB values
        name : to save the image file
    """
    d = custom_colors
    plt.clf()    
    plt.xlabel('x axis')
    plt.ylabel('z axis')
    image = np.array([[d[val] for val in row] for row in a], dtype='B')
    plt.imshow(image)
    plt.title('2D map {}'.format(text)) # for fires')
    plt.xticks([])
    plt.yticks([])

    # plt.draw()
    # plt.pause(5)
    if name is not None:
        plt.savefig(name)
    return 



def plot_image_3D(image, name='debug_plot'):
    """
    matplotlib plot of location map with custom colors
    
    Parameters:
        image: 4D array of shape [3, Height, Width, Color Channels] 
    where 3 refers to the three y axis planes relative to the player
    (below, same level, above)
    
    """

    if image is None:
        return 

    plt.clf()
    plt.xlabel('x axis')
    plt.ylabel('z axis')
    
    plt.subplot(131)
    # image1 = np.array([[d[val] for val in row] for row in a[0]], dtype='B')
    plt.imshow(image[0])
    plt.title('below') # for fires')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(132)
    # image2 = np.array([[d[val] for val in row] for row in a[1]], dtype='B')
    plt.imshow(image[1])
    plt.title('same level') # for stone_buttons')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(133)
    # image3 = np.array([[d[val] for val in row] for row in a[2]], dtype='B')
    plt.imshow(image[2])
    plt.title('above') # for levers
    plt.xticks([])
    plt.yticks([])


    plt.draw()
    plt.pause(0.1)
    plt.savefig(name)
    # image = np.array([[d[val] for val in row] for row in a], dtype='B')
    # plt.imshow(image)

    return 