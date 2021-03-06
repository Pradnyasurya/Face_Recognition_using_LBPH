import cv2

#vcap = cv2.VideoCapture(0)  # built-in webcamera

vcap = cv2.VideoCapture('Office.mp4')

if vcap.isOpened(): 
    # width  = vcap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
    # height = vcap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
    # or
    width  = vcap.get(3)  # float `width`
    height = vcap.get(4)  # float `height`

    print('width, height:', width, height)
    
    # fps = vcap.get(cv2.CAP_PROP_FPS)
    # or
    fps = vcap.get(5)
    
    print('fps:', fps)  # float `fps`
    
    # frame_count = vcap.get(cv2.CAP_PROP_FRAME_COUNT)
    # or
    frame_count = vcap.get(7)
    
    print('frames count:', frame_count)  # float `frame_count`
