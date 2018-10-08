
import pyglet
import thread
import itertools
import ctypes

import pykinect
from pykinect import nui
from pykinect.nui import JointId

import pygame
from pygame.color import THECOLORS
from pygame.locals import *
import numpy as np
import copy
import json

import cv2

import time
import moviepy
from moviepy.editor import *
import os
import cv2
import numpy as np
import time
import threading

import cv2
from game import preview2
import os


def win3():

    clip = VideoFileClip('urso_musica_good.mp4')



    os.environ["SDL_VIDEO_CENTERED"] = "1"



    preview2(clip, cv=True, cv_names=['Danca','Danca2'])



    # Closes all the frames
    cv2.destroyWindow('Danca')
    cv2.destroyWindow('Danca')
