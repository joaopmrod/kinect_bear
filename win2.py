import time

import thread
import itertools
import ctypes

import pykinect
from pykinect import nui
from pykinect.nui import JointId

import pygame
from pygame.color import THECOLORS
from pygame.locals import *

VIDEO_WINSIZE = 640*2, 480*2

def win2(framesQ):
    m = 0

    pygame.init()

    screen_lock = thread.allocate()

    screen = pygame.display.set_mode(VIDEO_WINSIZE, RESIZABLE, 32)
    pygame.display.set_caption('Python Kinect Demo2')


    screen.fill(THECOLORS["black"])



    while True:

            e = pygame.event.wait()
            dispInfo = pygame.display.Info()

        #try:
            frameTime = framesQ.get(block=True,timeout=.2)
            print("new frame",m)
            m=m+1



            screen.blit(frameTime, (0, 0))

            #pygame.display.update()

        #except Exception as e:
        #    print("..",e)
        #    pass
        #    time.sleep(.2)


def win3(framesQ):

    #return

    import sys, pygame

    pygame.init()

    black = 0, 0, 0

    screen = pygame.display.set_mode(VIDEO_WINSIZE,0,32)
    pygame.display.set_caption('Python Kinect Demo 2')
    #ball = pygame.image.load("cabeca.png")
    #ballrect = ball.get_rect()

    while 1:
        try:
            frameTime = framesQ.get(block=True,timeout=0.01)
        except:
            break

    #m=0
    while 1:
        event = pygame.event.poll()
        if event.type == pygame.QUIT:
            pygame.quit()
            break

        frameTime = framesQ.get(block=True)
        #print(frameTime, m)
        #m = m + 1

        screen.fill(black)

        surf = pygame.surfarray.make_surface(frameTime)

        surf = pygame.transform.scale(surf, (640 * 2, 480 * 2))

        screen.blit(surf, (0,0))
        pygame.display.flip()
#        pygame.time.delay(5)