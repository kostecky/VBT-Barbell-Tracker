# VBT-Barbell-Tracker
A proof of concept app to optically track a barbell through its range of motion using OpenCV to give the lifter realtime feedback on concentric avg velocity, cutoff velocity, and displacement for a Velocity Based Training program.

![Demo](/images/demo.gif?raw=true "Squat Demo")

## How it works
The app will detect a solid green area painted on to the end of your barbell. Given the measured diameter of this circle it will determine the pixel per mm scale to calculate distances and velocities of the barbell.

Movement of the barbell is continuously scanned to see if a lift is happening. When it detects a rep, it displays the most commonly referenced VBT metrics and informs you if your lifts are within 0.5-0.75 m/s as well as if they fall below a 20% velocity cutoff. There is an audible signal that is played of your lifts comply or fall outside of the range so you know when to terminate your set without having to keep your eyes on the computer.

The display tells you what rep you're on, the displacement of the bar, the velocity of the concentric part of the lift and whether the art should be terminated.

Once the barbell is at rest for 2 minutes, the counters and path tracking reset. You can also reset it by hand by pressing 'r'

## Motivation

- Optimize workouts by taking meaningful measurements, getting live feedback, and putting in minimum effective effort
- Autoregulation of load-volume
- Injury reduction
- Initial exposure to OpenCV and optical processing
- Initial exposure to training CNNs
- Avoid the purchase of a hardware unit like openbarbell, push, etc.

## Journey

- Started off wanting the app to auto-detect a wide variety of barbells in generalized surroundings
- Spent a week playing with Google Colab and training CNNs to do so. Moderate success, lots of GPU power needed and way too slow for running on CPU only devices
- Spent time playing with many different tracking models in OpenCV. None were reliable in a variety of scenarios.
- Realized that tracking a differentiated colour and shape was very fast, didn't require a GPU and was orders of magnitude more reliable than any tracking algorithm. Simple wins with a bit more initial setup, but it was worth it.

## VBT
https://www.scienceforsport.com/velocity-based-training

## Requirements
- Python 3.7
- A webcam or IP camera that supports RTSP
- Lime Green paper/paint/etc.

## Installation

1. Create a new python 3.7 virtualenv
```
mkvirtualenv VBT-barbell-tracker
```

2. Install python requirements
```
pip install -r requirements.txt
```

3. Generate intrinsic camera values to undistort fisheye/barrel effect
Generate intrinsic camera distortion values to remove any barrel/fisheye distortion that your camera may have. You can easily spot this by looking at the outer perimeter of your camera to see if straight lines appear curved.

    a. Using your webcam, take 10 or more snapshots of the chessboard printout. Save the images as `.png` files. Adhere the printout to a board, so it's very flat. The images should be taken to ensure you cover all areas of your image plane, paying attention to the outer perimeter where most of the distortion will take place. You can find the opencv chessboard image here: https://github.com/opencv/opencv/blob/master/doc/pattern.png

    b. Place these images in the `./images/` directory

    c. Take another image of the area you want to undistort as a test. Save it as `test.png` and place it in the `./images/` directory.

    d. Run the `python undistort_fisheye.py` script to discover the intrinsict values. They will be dumped out in a json file called `fisheye_calibration_data.json`

4. Place lime green paper/paint/etc. on the end of the barbell that faces the camera, ensuring that it covers the entire face of it. Then, take a measurement of the diameter of the face of the barbell end you covered. Mine comes out to 50mm, so the radius would be 25mm. This is **essential** to calibrating the distance scale within the app. You can use any colour you want, but you will have to adjust parameters accordingly. The idea here is to use a colour that is highly differentiated from anything being captured in your surroundings or your clothing.

## Usage

```
workon VBT-barbell-tracker
```

### Running it on a video
```
python vbt_barbell_tracker.py --video press.mp4
```

### Running it with your webcam
```
python VBT-barbell-tracker.py
```

### Running it on a RTSP stream
```
python VBT-barbell-tracker.py --video 'rtsp://USER:PASSWORD@192.168.1.127?tcp'
```

## Hotkeys

`r` Clear set and reset all counters and bar tracker. This also happens automatically after 2 minutes of no movement.

## Roadmap
- [ ] Refactor POC
- [ ] Use QT or simpleGUI to reformulate into a proper app with input/output to adjust parameters on the fly
