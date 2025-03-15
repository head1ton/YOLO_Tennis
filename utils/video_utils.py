import cv2

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        print(f"Mouse Position: ({x}, {y})")

def read_video(video_path):
    # cv2.namedWindow("Tennis Detection")
    # cv2.setMouseCallback("Tennis Detection", mouse_callback)

    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

        # cv2.imshow("Tennis Detection", frame)
        #
        # if cv2.waitKey(0) & 0xFF == ord('q'):
        #     break

    cap.release()
    # cv2.destroyAllWindows()
    return frames

def save_video(output_video_frames, output_video_path):
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')    # mp4v
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))
    for frame in output_video_frames:
        out.write(frame)
    out.release()
