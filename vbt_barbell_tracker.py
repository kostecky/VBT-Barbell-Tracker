from collections import deque
import numpy as np
import argparse
import imutils
from imutils.video import FPS
import cv2
import math
import json
import simpleaudio as sa
import wave


def is_inflection(velocity, opposite_direction_is_true):
    if opposite_direction_is_true:
        if velocity < 0:
            return True
        else:
            return False
    if opposite_direction_is_true is False:
        if velocity >= 0:
            return True
        else:
            return False


def analyze_for_rep(history):
    pos = 0
    first_phase = False
    second_phase = False
    concentric_first_lift = False
    displacement = 0
    first_phase_disp = 0
    second_phase_disp = 0
    velocities = []
    error = 0
    vector_threshold = 8

    print("Analyzing movement for rep...")
    # We need at least vector_threshold points to determine a direction and almost definitely many more for a complete rep
    if len(history) < 2 * vector_threshold:
        return(False, (0.0, 0.0, 0))

    # Method
    # 1. determine direction by looking at last X points
    # 2. keep reading and ensure each point matches initial direction up until inflection point
    # 3. Read all points after inflection up until next inflection or end of history
    # 4. Use criteria to determine if it's a rep
    for direction_pos in range(1, vector_threshold):
        displacement += history[-direction_pos][2]

    if displacement > 0:
        concentric_first_lift = True
    elif displacement < 0:
        concentric_first_lift = False
    else:
        # need more data to determine if it's a rep
        return(False, (0.0, 0.0, 0))

    while True:
        pos += 1

        if pos > len(history):
            break

        print(history[-pos])

        if not first_phase:
            # Count at least 1 first phase point before inflection and at least 100mm of displacement
            if is_inflection(history[-pos][2], concentric_first_lift) and first_phase_disp > 200:
                print("First Phase Displacement: {}".format(first_phase_disp))
                first_phase = True
            else:
                if is_inflection(history[-pos][2], concentric_first_lift):
                    if error > 3:
                        print("Found 3 error points that go in the opposite direction")
                        break
                    error += 1
                    continue
                else:
                    first_phase_disp += abs(history[-pos][1])
                    velocities.append(abs(history[-pos][2]))
                    continue

        if not second_phase:
            # Count at least 1 second phase point before first inflection and 200mm of displacement
            # or we're on the last point in history
            if (is_inflection(history[-pos][2], not concentric_first_lift) and second_phase_disp > 200) or (pos == len(history) and second_phase_disp > 200):
                second_phase = True
                print("Second Phase Displacement: {}".format(second_phase_disp))
            else:
                second_phase_disp += abs(history[-pos][1])
                continue

        # All this criteria should give us a high probability of counting a rep
        # Move more than 100mm, difference between eccentric and concentric displacement < 200mm
        if first_phase and second_phase and abs(second_phase_disp - first_phase_disp) < 100:
            print("Found rep! first: {} mm, second: {} mm".format(first_phase_disp, second_phase_disp))
            avg_vel = sum(velocities) / len(velocities)
            peak_vel = max(velocities)
            if concentric_first_lift:
                concentric_disp = first_phase_disp
            else:
                concentric_disp = second_phase_disp
            return(True, (avg_vel, peak_vel, concentric_disp))

    return(False, (0.0, 0.0, 0))


def analyze_for_x_movement(history):
    # When there is a pause or inflection, let's check our previous points to see
    # if there is more movement in the x direction than the y.
    # This would indicate deracking/racking/setting up and we want to clear the history
    # of any small movements that would confuse the algorithm from detecting the start
    # of a rep, whether it be a concentric or eccentric first exercise.
    pos = 0
    x_displacement = 0
    y_displacement = 0

    print("Analyzing movement for large x displacement over y")

    while pos < len(history):
        x_displacement += abs(history[pos][0])
        y_displacement += abs(history[pos][1])
        pos += 1

    if (x_displacement - y_displacement) >= 0:
        return True

    return False


ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the (optional) video file")
args = vars(ap.parse_args())

# Calibrate our camera
with open("fisheye_calibration_data.json", "r") as f:
    calibration_data = json.load(f)
    print(calibration_data)
    # Need to scale K, D, and dim
    # Dim should be 800x600
    # Original calibration used an image of 5120 × 3840 so same aspect ratio, which is good
    dim = (800, 600)
    scaled_K = np.asarray(calibration_data['K']) * dim[0] / 5120  # The values of K is to scale with image dimension.
    scaled_K[2][2] = 1.0  # Except that K[2][2] is always 1.0
    # This is how scaled_K, dim2 and balance are used to determine the final K used to un-distort image. OpenCV document failed to make this clear!
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, np.asarray(calibration_data['D']), dim, np.eye(3), balance=calibration_data['balance'])
    calibration_map1, calibration_map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, np.asarray(calibration_data['D']), np.eye(3), new_K, dim, cv2.CV_16SC2)

# Different lighting conditions
# 1.png
# (hMin = 40 , sMin = 46, vMin = 0), (hMax = 86 , sMax = 88, vMax = 181)
# 2.png
# (hMin = 33 , sMin = 48, vMin = 103), (hMax = 64 , sMax = 156, vMax = 255)
greenLower = (33, 46, 0)
greenUpper = (86, 156, 255)

if not args.get("video", False):
    camera = cv2.VideoCapture(0)
else:
    camera = cv2.VideoCapture(args["video"])

# Read first image
(grabbed, frame) = camera.read()

frameCount = int(camera.get(cv2.CAP_PROP_FRAME_COUNT))
vid_fps = int(camera.get(cv2.CAP_PROP_FPS))
print(vid_fps)
fps = FPS().start()
points = deque(maxlen=10000)

last_x = None
last_y = None
ref_radius = None
velocity = 0.0
x_velocity = 0.0
y_velocity = 0.0
y_vector_up = False
moving = False
analyzed_rep = False
barbell_radius = 25
reps = 0
history = []
# How many milliseconds at rest until we consider it a rep?
rep_rest_threshold = 80.0
rep_rest_time = 0.0
avg_vel = 0.0
peak_vel = 0.0
displacement = 0
avg_velocities = []
peak_velocities = []
velocity_loss_threshold = 20
in_range = True
end_set = False
colour = (0, 255, 0)
avg_velocity = 0
avg_velocity_loss = 0

cv2.namedWindow("output", cv2.WINDOW_OPENGL)

while True:
    (grabbed, frame) = camera.read()

    if args.get("video") and not grabbed:
        break

    # video image is 2560 × 1920 = 1.3333333...
    # should be 800/600 to maintain aspect ratio
    frame = imutils.resize(frame, width=800)

    # Remove barrel/fisheye distortion
    frame = cv2.remap(frame, calibration_map1, calibration_map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    # Crop the frame to get rid of the deadspace from undistorting it
    frame = frame[100:500, 100:700]
    frame = imutils.resize(frame, width=800)

    im_height, im_width, _ = frame.shape

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, greenLower, greenUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    center = None

    # Our barbell end is 5cm in diameter
    # fps = camera.get(cv2.CAP_PROP_FPS)
    # Each frame is 1/fps seconds in length
    # 1. Calculate the pixel distance between last position and current position. That provides pixels travelled in 1/FPS seconds.
    # That gives pixel distance between frames
    # 2. How many pixels does the circle diameter occupy? Divide by 50mm and we get pixels/mm. Invert and we get mm/pixel
    # Multiply 1 by 2 and we get instantaneous mm/s every 1/FPS

    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)

        if last_x is None:
            last_x = x
            last_y = y

        velocity = 0
        y_velocity = 0
        rep = False
        if reps == 0:
            in_range = True
            avg_velocity = 0
            peak_velocity = 0
            avg_velocity_loss = 0
            peak_velocity_loss = 0
            end_set = False
            colour = (0, 255, 0)

        if radius / im_height > 0.01:
            # Take the first radius as the reference radius as it's stationary and most accurately represents dimensions
            if ref_radius is None:
                ref_radius = radius
                mmpp = barbell_radius / ref_radius
                print("ref_radius: {:.2f}, mmpp: {:.2f}".format(ref_radius, mmpp))
            x_disp = last_x - x
            y_disp = last_y - y
            y_distance = y_disp * mmpp
            x_distance = x_disp * mmpp
            distance = math.sqrt(x_disp ** 2 + y_disp ** 2) * mmpp
            if abs(y_distance) > barbell_radius / 4:
                moving = True
                analyzed_rep = False
                velocity = distance * vid_fps / 1000
                y_velocity = y_distance * vid_fps / 1000
                rep_rest_time = 0.0
                if y_velocity < 0.01 and y_velocity > -0.01:
                    moving = False
                    y_velocity = 0
                    rep_rest_time += 1 / vid_fps * 1000
                print("distance: {} mm, velocity: {:.2f} m/s, x_dist: {} mm, y_dist: {} mm, y_vel: {:.2f} m/s".format(int(distance), float(velocity), int(x_distance), int(y_distance), float(y_velocity)))
                history.append((int(x_distance), int(y_distance), y_velocity))
                if (y_velocity > 0 and y_vector_up is False) or (y_velocity < 0 and y_vector_up is True):
                    y_vector_up = not y_vector_up
                    (rep, ret) = analyze_for_rep(history)
            else:
                # Only log 0 once
                if moving is True:
                    history.append((0, 0, 0))
                moving = False
                # Count how many milliseconds we're at 0 velocity
                rep_rest_time += 1 / vid_fps * 1000
                # analyze for last rep when we appear to rest for a threshold time
                if (rep_rest_time > rep_rest_threshold) and not analyzed_rep:
                    analyzed_rep = True
                    if analyze_for_x_movement(history):
                        print("Detected significant x movement over y, resetting history...")
                        history = []
                    (rep, ret) = analyze_for_rep(history)

            if rep:
                history = []
                reps += 1
                avg_velocities.append(ret[0])
                peak_velocities.append(ret[1])
                displacement = ret[2]

                if reps == 1:
                    avg_velocity = avg_velocities[0]
                    peak_velocity = peak_velocities[0]
                    if avg_velocity > 0.5 and avg_velocity < 0.75:
                        in_range = True
                        wave_read = wave.open('good.wav', 'rb')
                    else:
                        in_range = False
                        wave_read = wave.open('bad.wav', 'rb')
                else:
                    avg_velocity = avg_velocities[-1]
                    peak_velocity = peak_velocities[-1]
                    avg_velocity_loss = (avg_velocities[0] - avg_velocities[-1]) / avg_velocities[0] * 100
                    peak_velocity_loss = (peak_velocities[0] - peak_velocities[-1]) / peak_velocities[0] * 100

                    if avg_velocity_loss > velocity_loss_threshold:
                        end_set = True
                        wave_read = wave.open('bad.wav', 'rb')
                        colour = (0, 0, 255)
                    else:
                        end_set = False
                        wave_read = wave.open('good.wav', 'rb')
                        colour = (0, 255, 0)

                audio_data = wave_read.readframes(wave_read.getnframes())
                num_channels = wave_read.getnchannels()
                bytes_per_sample = wave_read.getsampwidth()
                sample_rate = wave_read.getframerate()
                wave_obj = sa.WaveObject(audio_data, num_channels, bytes_per_sample, sample_rate)
                play_obj = wave_obj.play()

            last_x = x
            last_y = y
            cv2.circle(frame, (int(x), int(y)), int(ref_radius), (0, 255, 255), 2)
            path_color = (0, 255, 0)
            center = (int(x), int(y))
            points.appendleft(center)
            for i in range(1, len(points)):
                if points[i - 1] is None or points[i] is None:
                    continue
                cv2.line(frame, points[i - 1], points[i], path_color, 2)

    cur_frame = camera.get(cv2.CAP_PROP_POS_FRAMES)
    fps.update()
    fps.stop()
    info = [
        ("First set in range", "{}".format(in_range), (0, 255, 0)),
        ("Last AVG Con Velocity", "{:.2f} m/s".format(avg_velocity), (0, 255, 0)),
#        ("Last PEAK Con Velocity", "{:.2f} m/s".format(peak_velocity), (0, 255, 0)),
        ("Last Displacement", "{:.2f} mm".format(displacement), (0, 255, 0)),
        ("AVG Velocity Loss", "{:.2f} %".format(avg_velocity_loss), (0, 255, 0)),
#        ("PEAK Velocity Loss", "{:.2f} %".format(peak_velocity_loss), (0, 255, 0)),
        ("Reps", "{}".format(reps), (0, 255, 0)),
#        ("FPS", "{:.2f}".format(fps.fps()), (0, 255, 0)),
#        ("Y Velocity", "{:.2f} m/s".format(y_velocity), (0, 255, 0)),
        ("END SET", "{}".format(end_set), colour),
    ]

    # loop over the info tuples and draw them on our frame
    for (i, (k, v, c)) in enumerate(info):
        text = "{}: {}".format(k, v)
        cv2.putText(frame, text, (10, im_height - ((i * 40) + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, c, 2)

    cv2.imshow("output", frame)
    cv2.resizeWindow("output", (1500, 1000))
#    cv2.imshow("Mask", mask)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break
    elif key == ord("r"):
        reps = 0
        avg_velocities = []
        peak_velocities = []
        points.clear()


camera.release()
cv2.destroyAllWindows()
