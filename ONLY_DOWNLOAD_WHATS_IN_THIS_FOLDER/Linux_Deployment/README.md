# 9-Ball Foul Detector — Linux Docker Release

This package runs the Tkinter desktop interface on an x86_64 Linux desktop,
uses a USB webcam through Linux V4L2, performs inference on the CPU, and saves
all generated files under `Output_Folder/`.

The default camera settings match the Logitech C922:

- 1280 x 720
- 60 FPS request
- MJPEG capture

The camera may negotiate a lower frame rate if its USB connection, selected
format, or lighting mode does not support 720p60.

## Package contents

```text
Docker_Linux_Release/
├── Docker_Linux_Main_Code.py
├── Trained_Models/
│   ├── YOLO11n/weights/best.pt
│   ├── YOLOv10n/weights/best.pt
│   ├── YOLOv10n-2/weights/best.pt
│   ├── YOLOv8n/weights/best.pt
│   └── YOLOv8n-2/weights/best.pt
├── Output_Folder/
├── Dockerfile
├── compose.yaml
├── compose.release.yaml
├── requirements.txt
├── run-linux.sh
└── run-loaded-image.sh
```

Only the Linux entry-point code and the five `best.pt` inference weights
are copied into the image. Training notebooks, datasets, plots, test videos,
and `last.pt` weights are excluded.

## Linux prerequisites

Install Docker Engine with the Docker Compose v2 plugin. The logged-in user
must be able to run Docker and access the webcam. If camera permission is
missing, add the user to the `video` group and sign out and back in:

```bash
sudo usermod -aG video "$USER"
```

The desktop session must provide X11 or XWayland. Most Ubuntu, Linux Mint,
Debian, Fedora, and similar desktop installations already do.

## Find the Logitech camera

Connect the C922 and run:

```bash
ls -l /dev/video*
```

The first usable capture node is commonly `/dev/video0`. Some systems expose
multiple nodes for one camera. To inspect them more clearly:

```bash
sudo apt install v4l-utils
v4l2-ctl --list-devices
```

## Build and run

From this directory:

```bash
chmod +x run-linux.sh run-loaded-image.sh
./run-linux.sh
```

The first run downloads the Ubuntu base image and Python dependencies, so it
can take several minutes. Later starts reuse the built image.

If the camera is not `/dev/video0`, specify the correct host device:

```bash
WEBCAM_DEVICE=/dev/video2 ./run-linux.sh
```

The program scans the mapped camera, asks you to select a model, and already
uses `/output` inside the container. That path is mapped to the host's
`Output_Folder/`, so no additional output-folder selection is required.

Close the Tkinter window or press Ctrl+C in the terminal to stop it. If
necessary:

```bash
docker compose down
```

## Optional camera overrides

Set any of these before the launch command:

```bash
CAMERA_WIDTH=1920 CAMERA_HEIGHT=1080 CAMERA_FPS=30 ./run-linux.sh
```

The defaults are `1280`, `720`, and `60`.

## Export for another Linux device

Build and export the image:

```bash
docker compose build
docker save -o nine-ball-foul-detector-linux-1.0.tar \
  nine-ball-foul-detector-linux:1.0
```

Give the other user:

- `nine-ball-foul-detector-linux-1.0.tar`
- `compose.release.yaml`
- `run-linux.sh`
- `run-loaded-image.sh`
- an empty `Output_Folder/` directory

They load and run it with:

```bash
docker load -i nine-ball-foul-detector-linux-1.0.tar
chmod +x run-linux.sh run-loaded-image.sh
./run-loaded-image.sh
```

The exported image already contains the program, dependencies, and model
weights.

## Troubleshooting

- **No camera found:** verify the selected `/dev/videoN` node with
  `v4l2-ctl --list-devices`, then set `WEBCAM_DEVICE`.
- **Permission denied opening the camera:** check membership with
  `groups` and ensure `video` is listed.
- **Tk window does not appear:** confirm `echo "$DISPLAY"` is not empty and
  that XWayland is enabled for a Wayland desktop session.
- **720p60 is not reached:** connect the C922 directly to a suitable USB port
  and try 720p30 before testing higher settings.

## Distribution note

Docker sets the intended entry point but does not make source code or weights
unextractable. Also review the Ultralytics AGPL-3.0 or Enterprise licensing
requirements before distributing a closed-source image.
