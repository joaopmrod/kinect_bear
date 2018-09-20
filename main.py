from pykinect import nui
from pykinect.nui import JointId

import np
import cv2

def video_handler_function(frame):
    video = np.empty((480,640,4),np.uint8)
    frame.image.copy_bits(video.ctypes.data)
    #cv2.imshow('KINECT Video Stream', video)


def skeleton_handler_function(frame):
    for skeleton in frame.SkeletonData:
         #print("Positions : " , data.SkeletonPositions)

         #print(skeleton.eTrackingState)

         if skeleton.eTrackingState ==  nui.SkeletonTrackingState.TRACKED:
             a = zip(range(20),skeleton.SkeletonPositions)
             print(a[0])

kinect = nui.Runtime()
kinect.skeleton_engine.enabled = True

kinect.video_frame_ready += video_handler_function
kinect.video_stream.open(nui.ImageStreamType.Video, 2,nui.ImageResolution.Resolution640x480,nui.ImageType.Color)

#kinect.depth_frame_ready += video_handler_function
#kinect.depth_stream.open(nui.ImageStreamType.Depth, 2, nui.ImageResolution.Resolution640x480, nui.ImageType.Depth)


kinect.skeleton_frame_ready += skeleton_handler_function

kinect.skeleton_engine.enabled=True

cv2.namedWindow('KINECT Video Stream', cv2.WINDOW_AUTOSIZE)




while True:

    key = cv2.waitKey(1)
    if key == 27: break

kinect.close()
cv2.destroyAllWindows()