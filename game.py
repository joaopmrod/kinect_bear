
screen = None

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


#import draw
#from draw import draw_skeletons

KINECTEVENT = pygame.USEREVENT
DEPTH_WINSIZE = 320, 240
VIDEO_WINSIZE = 640*2, 480*2
pygame.init()

SKELETON_COLORS = [THECOLORS["red"],
                   THECOLORS["blue"],
                   THECOLORS["green"],
                   THECOLORS["orange"],
                   THECOLORS["purple"],
                   THECOLORS["yellow"],
                   THECOLORS["violet"]]

LEFT_ARM = (JointId.ShoulderCenter,
            JointId.ShoulderLeft,
            JointId.ElbowLeft,
            JointId.WristLeft,
            JointId.HandLeft)
RIGHT_ARM = (JointId.ShoulderCenter,
             JointId.ShoulderRight,
             JointId.ElbowRight,
             JointId.WristRight,
             JointId.HandRight)
LEFT_LEG = (JointId.HipCenter,
            JointId.HipLeft,
            JointId.KneeLeft,
            JointId.AnkleLeft,
            JointId.FootLeft)
RIGHT_LEG = (JointId.HipCenter,
             JointId.HipRight,
             JointId.KneeRight,
             JointId.AnkleRight,
             JointId.FootRight)
SPINE = (JointId.HipCenter,
         JointId.Spine,
         JointId.ShoulderCenter,
         JointId.Head)


##################

LEFT_ARM = ( JointId.ShoulderCenter,JointId.ShoulderLeft,
            JointId.ElbowLeft,
            JointId.WristLeft)
RIGHT_ARM = (JointId.ShoulderCenter,
             JointId.ShoulderRight,
             JointId.ElbowRight,
             JointId.WristRight)
LEFT_LEG = (
            JointId.HipLeft,
            JointId.KneeLeft,
            JointId.AnkleLeft)
RIGHT_LEG = (
             JointId.HipRight,
             JointId.KneeRight,
             JointId.AnkleRight)
SPINE = (JointId.HipCenter,
         #JointId.Spine,
         JointId.ShoulderCenter)



def get_screen_pixel (s,w,h):
    pos =nui.SkeletonEngine.skeleton_to_depth_image(s,DEPTH_WINSIZE[0], DEPTH_WINSIZE[1])
    try:
        x = int(pos[0])
        y = int(pos[1])
        if depth_array is not None and video_frame is not None and video_display:
            frame=video_frame
            pos = kinect.camera.get_color_pixel_coordinates_from_depth_pixel(frame.resolution, frame.view_area, x, y, depth_array.item((x, y)))
    except Exception as e:
        #print(e)
        pass
    return pos



# recipe to get address of surface: http://archives.seul.org/pygame/users/Apr-2008/msg00218.html
if hasattr(ctypes.pythonapi, 'Py_InitModule4'):
    Py_ssize_t = ctypes.c_int
elif hasattr(ctypes.pythonapi, 'Py_InitModule4_64'):
    Py_ssize_t = ctypes.c_int64
else:
    raise TypeError("Cannot determine type of Py_ssize_t")

_PyObject_AsWriteBuffer = ctypes.pythonapi.PyObject_AsWriteBuffer
_PyObject_AsWriteBuffer.restype = ctypes.c_int
_PyObject_AsWriteBuffer.argtypes = [ctypes.py_object,
                                    ctypes.POINTER(ctypes.c_void_p),
                                    ctypes.POINTER(Py_ssize_t)]


def surface_to_array(surface):
    buffer_interface = surface.get_buffer()
    address = ctypes.c_void_p()
    size = Py_ssize_t()
    _PyObject_AsWriteBuffer(buffer_interface,
                            ctypes.byref(address), ctypes.byref(size))
    bytes = (ctypes.c_byte * size.value).from_address(address.value)
    bytes.object = buffer_interface
    return bytes



tmp_s = pygame.Surface(DEPTH_WINSIZE, 0, 16)

depth_array = None
def depth_frame_ready(frame):
    global depth_array
    frame.image.copy_bits(tmp_s._pixels_address)
    depth_array = pygame.surfarray.pixels2d(tmp_s)

    if video_display:
        return
    with screen_lock:
        address = surface_to_array(screen)
        frame.image.copy_bits(address)
        del address
        if skeletons is not None and draw_skeleton:
            draw_skeletons(skeletons)
        pygame.display.update()

video_frame = None


def video_frame_ready(frame):


    global video_frame

    video = pygame.Surface((640, 480), 0, 32)
    video_sk = pygame.Surface((640, 480), 0, 32)

    video_frame=frame
    if not video_display:
        return
    with screen_lock:
        address = surface_to_array(video)
        frame.image.copy_bits(address)
        del address

        address2 = surface_to_array(video_sk)
        frame.image.copy_bits(address2)
        del address2

        video_sk_2 = None
        video_2 = None

        if skeletons is not None:
            draw_skeletons(skeletons,video_sk)

        if draw_skeleton and video_sk is not None:
            video_sk_2 = pygame.transform.smoothscale(video_sk, (640 * 2, 480 * 2))
            screen.blit(video_sk_2, (0, 0))
        else:
            video_2 = pygame.transform.smoothscale(video, (640 * 2, 480 * 2))
            screen.blit(video_2, (0, 0))


        pygame.display.update()

        if win2_sk and video_sk is not None:
            #if video_sk_2 is None:
                #video_sk_2 = pygame.transform.smoothscale(video_sk, (640 * 2, 480 * 2))
            s = pygame.surfarray.pixels3d(video_sk)
        else:
            #if video_2 is None:
                #video_2 = pygame.transform.smoothscale(video, (640 * 2, 480 * 2))
            s = pygame.surfarray.pixels3d(video)
        try:
            win2_frames.put(s,block=False)
        except:
            pass




cabeca = pygame.image.load('cabeca.png')
tronco = pygame.image.load('tronco.png')
multi = pygame.image.load('multi.png')
bracoE2 = pygame.image.load('bracoE2.png')
bracoD2 = pygame.image.load('bracoD2.png')
pernaE2 =pygame.image.load('pernaE2.png')
pernaD2 = pygame.image.load('pernaD2.png')
maoD = pygame.image.load('maoD.png')


todos = False

def draw_skeletons(skeletons,surface):

    esquerda = 99
    esquerda_id= None

    direita =-99
    direita_id=None

    for index, data in enumerate(skeletons):
        head = data.SkeletonPositions[JointId.Head]
        if head.w !=0:
            if head.x<esquerda:
                esquerda_id=index
                esquerda=head.x

            if head.x>direita:
                direita_id=index
                direita=head.x

    sks=[]

    if esquerda_id is not None: sks.append(skeletons[esquerda_id])
    if direita_id is not None and direita_id!=esquerda_id: sks.append(skeletons[direita_id])


    for index, data in enumerate(sks):

        if index == 0 or todos:


            '''line = True

            #Left Arm
            body_part(data, multi, JointId.ShoulderLeft, JointId.ElbowLeft, center_dist=0.5, extraH=1.2, racio=0.5, line=line)
            #Right Arm
            body_part(data, multi, JointId.ShoulderRight, JointId.ElbowRight, center_dist=0.5, extraH=1.2, racio=0.5,line=line)

            #Left leg
            body_part(data, multi, JointId.HipLeft, JointId.KneeLeft, center_dist=0.5, extraH=1.2, racio=0.5,line=line)
            body_part(data, pernaE2, JointId.KneeLeft, JointId.AnkleLeft, center_dist=0.5, extraH=1.2, racio=0.5, line=line)

            #Right Leg
            body_part(data, multi, JointId.HipRight, JointId.KneeRight, center_dist=0.5, extraH=1.2, racio=0.5, line=line)
            body_part(data, pernaD2, JointId.KneeRight, JointId.AnkleRight, center_dist=0.5, extraH=1.2, racio=0.5, line=line)

            #Tronco
            body_part(data, tronco, JointId.ShoulderCenter, JointId.HipCenter, center_dist=1, extraH=2.5, racio=0.5,line=line)

            #A frente do tronco
            # Left Arm
            body_part(data, bracoE2, JointId.ElbowLeft, JointId.WristLeft, center_dist=0.5, extraH=1.2, racio=0.5,line=line)
            # Right Arm
            body_part(data, bracoD2, JointId.ElbowRight, JointId.WristRight, center_dist=0.5, extraH=1.2, racio=0.5,line=line)

            #Head
            body_part(data,cabeca,JointId.Head,JointId.ShoulderCenter,center_dist=0,extraH=2.5,racio=1.3,line=line)


            '''

            if json is None:
                reload_json()

            body_part_json(data,surface)


            #draw_skeleton_data(data, index, SPINE, 10)
json_data = None
def reload_json():
    global json_data
    with open("data.json", "r") as read_file:
        json_data = json.load(read_file)

    print (json_data)

    for part in json_data['parts']:
        print(part.values()[0])


def body_part_json(data,surface):
    for part in json_data['parts']:
        #part=json_data['parts'][]
        part=part.values()[0]#[part.keys()[0]]
        body_part(surface,data, eval(part['imagem']), eval(part['a']), eval(part['b']), part['center_dist'], part['extraH'], part['racio'], part['w_forced'], part['h_forced'], False)
#        body_part(data, json[name]['imagem'], json[name]['a'], json[name]['b'], json[name]['center_dist'],
#                  json[name]['extraH'], json[name]['racio'], json[name]['w_forced'], json[name]['h_forced'],
#                  json[name]['line'])


def body_part(surface,data, imagem, a, b, center_dist=0.5, extraH=1.2,racio=1, w_forced=None, h_forced=None,line=False ):

    start = np.array(get_screen_pixel(data.SkeletonPositions[a], DEPTH_WINSIZE[0], DEPTH_WINSIZE[1]))
    end = np.array(get_screen_pixel(data.SkeletonPositions[b], DEPTH_WINSIZE[0], DEPTH_WINSIZE[1]))

    vector = end-start

    rot = np.degrees(np.arctan2(vector[0], vector[1]))

    center = start+vector*center_dist

    h = dist(start, end)*extraH

    if h_forced is not None:
        h = h_forced

    w = h * racio
    if w_forced is not None:
        w = w_forced

    body_part = copy.copy(imagem)
    body_part2 = pygame.transform.scale(body_part, (int(w), int(h)))
    body_part3 = pygame.transform.rotate(body_part2, rot)



    surface.blit(body_part3, (int(center[0] - body_part3.get_width() / 2), int(center[1] - body_part3.get_height() / 2)))


    if line:
        pygame.draw.line(surface, THECOLORS["red"], start, end, 5)


def dist(x,y):
    x=np.array(x)
    y = np.array(y)
    return np.sqrt(np.sum((x-y)**2))

win2_sk=False
if __name__ == '__main__':
    full_screen = True
    draw_skeleton = True
    video_display = False

    screen_lock = thread.allocate()

    screen = pygame.display.set_mode(DEPTH_WINSIZE, 0, 16)
    pygame.display.set_caption('Python Kinect Demo')
    skeletons = None
    screen.fill(THECOLORS["black"])

    kinect = nui.Runtime()
    kinect.skeleton_engine.enabled = True

    reload_json()


    def post_frame(frame):
        try:
            pygame.event.post(pygame.event.Event(KINECTEVENT, skeletons=frame.SkeletonData))
        except:
            # event queue full
            pass


    kinect.skeleton_frame_ready += post_frame

    kinect.depth_frame_ready += depth_frame_ready
    kinect.video_frame_ready += video_frame_ready

    kinect.video_stream.open(nui.ImageStreamType.Video, 2, nui.ImageResolution.Resolution640x480, nui.ImageType.Color)
    kinect.depth_stream.open(nui.ImageStreamType.Depth, 2, nui.ImageResolution.Resolution320x240, nui.ImageType.Depth)

    print('Controls: ')
    print('     d - Switch to depth view')
    print('     v - Switch to video view')
    print('     s - Toggle displaing of the skeleton')
    print('     u - Increase elevation angle')
    print('     j - Decrease elevation angle')


    # main game loop
    done = False

    from multiprocessing import Process, Queue, Value, Array, Lock
    import win2

    win2_frames = Queue(200)

    win2_proc = Process(target=win2.win3, args=(win2_frames,))
    win2_proc.start()

    f=True

    while not done:

        if f:
            with screen_lock:
                screen = pygame.display.set_mode(VIDEO_WINSIZE, RESIZABLE, 32)
                video_display = True
                f = False

        e = pygame.event.wait()
        dispInfo = pygame.display.Info()
        if e.type == pygame.QUIT:
            done = True
            break
        elif e.type == KINECTEVENT:
            skeletons = e.skeletons
            #if draw_skeleton:
            #    draw_skeletons(skeletons)
            #    pygame.display.update()
        elif e.type == KEYDOWN:
            if e.key == K_ESCAPE:
                done = True
                break
            elif e.key == K_d:
                pass
                #with screen_lock:
                #    screen = pygame.display.set_mode(DEPTH_WINSIZE, 0, 16)
                #    video_display = False
            elif e.key == K_v or f:
                pass
                #with screen_lock:
                #    screen = pygame.display.set_mode(VIDEO_WINSIZE, RESIZABLE, 32)
                #    video_display = True
                #    f = False
                #    print('VVVV')
            elif e.key == K_s:
                draw_skeleton = not draw_skeleton
            elif e.key == K_u:
                kinect.camera.elevation_angle = kinect.camera.elevation_angle + 2
            elif e.key == K_j:
                kinect.camera.elevation_angle = kinect.camera.elevation_angle - 2
            elif e.key == K_x:
                kinect.camera.elevation_angle = 2
            elif e.key == K_r:
                reload_json()
            elif e.key == K_a:
               win2_sk= not win2_sk
            elif e.key == K_t:
               todos= not todos

