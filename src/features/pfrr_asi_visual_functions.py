""" 
Functions used to visualize PFRR ASI Images such as creating movies

@author Riley Troyer
"""

from datetime import datetime
from dateutil import parser
import h5py
import logging
from matplotlib import animation
from matplotlib import pyplot as plt
import os


def create_timestamped_movie(date:datetime.date, img_base_dir:str, save_base_dir:str, wavelength:str='558') -> None:
    
    """Function to create a movie from PFRR ASI files with a timestamp and frame number.
    Includes a timestamp, and frame number.
    INPUT
    date - day to create movie for
    save_base_dir - base directory to store keogram image
    img_base_dir - base directory to where raw images are stored.
    wavelength - which wavelength images are being used. 428, 558, or 630
    OUTPUT
    logging information
    """
    
    # Select file with images
    img_file = (img_base_dir + '/all-images-'
                + str(date) + '-' + wavelength + '.h5')

    pfrr_file = h5py.File(img_file, "r")

    # Get times from file
    all_times = [parser.isoparse(d) for d in pfrr_file['iso_ut_time']]

    # Get all the images too
    all_images = pfrr_file['images']

    # CREATE MOVIE

    img_num = all_images.shape[0]
    fps = 20.0


    # Construct an animation
    # Setup the figure
    fig, axpic = plt.subplots(1, 1)

    # No axis for images
    axpic.axis('off')

    # Plot the image
    img = axpic.imshow(all_images[0],
                       cmap='gray', animated=True)

    # Add frame number and timestamp to video
    frame_num = axpic.text(10, 500, '0000', fontweight='bold',
                           color='red')
    time_str = str(all_times[0])
    time_label = axpic.text(120, 500,
                            time_str,
                            fontweight='bold',
                            color='red')

    plt.tight_layout()

    def _updatefig(frame):
        """Function to update the animation"""

        # Set new image data
        img.set_data(all_images[frame])
        # And the frame number
        frame_num.set_text(str(frame).zfill(4))
        #...and time
        time_str = str(all_times[frame])
        time_label.set_text(time_str)

        return [img, frame_num, time_label]

    # Construct the animation
    anim = animation.FuncAnimation(fig, _updatefig,
                                   frames=img_num,
                                   interval=int(1000.0/fps),
                                   blit=True)

    # Close the figure
    plt.close(fig)


    # Use ffmpeg writer to save animation
    event_movie_fn = (save_base_dir 
                      + str(date) + '-' + wavelength
                      + '.mp4')
    writer = animation.writers['ffmpeg'](fps=fps)
    anim.save(event_movie_fn,
              writer=writer, dpi=150)

    # Close h5py file
    pfrr_file.close()

    # Reset large image array
    all_images = None
