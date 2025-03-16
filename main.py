import cv2

from mini_court.mini_court import MiniCourt
from utils import (read_video, save_video)
from trackers import PlayerTracker, BallTracker
from court_line_detect import CourtLineDetector

# pip install certifi
# export SSL_CERT_FILE=$(python3 -m certifi)
# /Applications/Python\ 3.12/Install\ Certificates.command


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        print(f"Mouse Position: ({x}, {y})")

def main():
    input_video_path = "input_videos/input_video.mp4"
    cv2.namedWindow("Tennis Analysis")
    cv2.setMouseCallback("Tennis Analysis", mouse_callback)

    player_tracker = PlayerTracker(model_path='yolov8x')
    ball_tracker = BallTracker(model_path='models/best.pt')

    cap = cv2.VideoCapture(input_video_path)
    video_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        video_frames.append(frame)

        player_detections = player_tracker.detect_frames(video_frames,
                                                         read_from_stub=True,
                                                         stub_path="tracker_stubs/player_detections.pkl")
        ball_detections = ball_tracker.detect_frames(video_frames,
                                                     read_from_stub=True,
                                                     stub_path="tracker_stubs/ball_detections.pkl")

        ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)

        court_model_path = "models/keypoints_model.pth"
        court_line_detector = CourtLineDetector(court_model_path)
        court_keypoints = court_line_detector.predict(video_frames[0])

        player_detections = player_tracker.choose_and_filter_players(court_keypoints, player_detections)

        mini_court = MiniCourt(video_frames[0])
        # print('video_frames : ', video_frames)
        # print('video_frames[0] : ', video_frames[0])
        ball_shot_frames = ball_tracker.get_ball_shot_frames(ball_detections)
        # print(ball_shot_frames)


        output_video_frames = player_tracker.draw_bboxes(video_frames,player_detections)
        output_video_frames = ball_tracker.draw_bboxes(video_frames, ball_detections)

        output_video_frames = court_line_detector.draw_keypoints_on_video(output_video_frames, court_keypoints)

        output_video_frames = mini_court.draw_mini_court(output_video_frames)

        for i, frame in enumerate(output_video_frames):
            cv2.putText(frame, f"Frame: {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # save_video(output_video_frames, "output_videos/output_video.avi")

        cv2.imshow("Tennis Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


    # input_video_path = "input_videos/input_video.mp4"
    # video_frames = read_video(input_video_path)
    #
    # player_tracker = PlayerTracker(model_path='yolov8x')
    # ball_tracker = BallTracker(model_path='models/best.pt')
    #
    # player_detections = player_tracker.detect_frames(video_frames,
    #                                                  read_from_stub=True,
    #                                                  stub_path="tracker_stubs/player_detections.pkl")
    # ball_detections = ball_tracker.detect_frames(video_frames,
    #                                              read_from_stub=True,
    #                                              stub_path="tracker_stubs/ball_detections.pkl")
    #
    # ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)
    #
    # court_model_path = "models/keypoints_model.pth"
    # court_line_detector = CourtLineDetector(court_model_path)
    # court_keypoints = court_line_detector.predict(video_frames[0])
    #
    # player_detections = player_tracker.choose_and_filter_players(court_keypoints, player_detections)
    #
    #
    # output_video_frames = player_tracker.draw_bboxes(video_frames,player_detections)
    # output_video_frames = ball_tracker.draw_bboxes(video_frames, ball_detections)
    #
    # output_video_frames = court_line_detector.draw_keypoints_on_video(output_video_frames, court_keypoints)
    #
    # for i, frame in enumerate(output_video_frames):
    #     cv2.putText(frame, f"Frame: {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    #
    # save_video(output_video_frames, "output_videos/output_video.avi")


if __name__ == "__main__":
    main()
