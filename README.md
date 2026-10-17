# Ad Muter

Ad Muter automatically mutes your TV when it detects ads and restores sound when they end, using a Raspberry Pi, a camera, and infrared control.

The current preset recognizes Hulu's light Ad badge using normalized glyph descriptors. It finds candidate badges in each current image, measures the TV bezel, and uses a native-resolution search for faint lettering. There is no fixed camera crop. Brief geometry ambiguity can use a confirmed bezel from the preceding five seconds, but only if one strong measurement still matches in the current frame.

This is an experimental prototype. Hulu has been tested; Netflix, Max, Prime Video and other badge styles are not validated. Camera framing, glare, blur, small lettering and ambiguous screen edges can cause missed or uncertain detections.

## Hardware

- Raspberry Pi 5 and a supported Raspberry Pi OS camera stack.
- IMX708 camera; the prototype uses an Arducam wide-angle module.
- Linux LIRC-compatible IR transmitter and receiver. The prototype uses a TSAL6200 LED with a transistor driver and a 38 kHz receiver.
- A TV remote whose mute toggle and power command you can learn.

Use a suitable LED driver; do not connect a high-current IR LED directly to GPIO or expose GPIO to 5 V. Verify the transistor pinout and your own circuit before connecting the Pi.

Printable camera and IR enclosures are in [cad](cad/README.md).

## Setup

On Raspberry Pi OS, install the camera, vision, OCR and LIRC packages:

```bash
sudo apt update
sudo apt install python3-picamera2 python3-opencv python3-numpy python3-venv tesseract-ocr v4l-utils
python3 -m venv --system-site-packages .venv
.venv/bin/pip install pytesseract
```

For offline development on a computer, install `pip install -r requirements.txt`. Picamera2 is imported only by the live camera worker; offline tests do not require a Pi. Generic OCR needs a Tesseract executable; the included glyph-model path does not.

Configure the Pi's GPIO IR overlays for your wiring separately. Check `ir-ctl --features` for each `/dev/lirc*` device: device numbers can change. Capture your own remote timings using `ir-ctl` and store them locally, outside the repository. No learned household IR configuration is included.

## Inspect without changing TV sound

```bash
.venv/bin/python ad_detector.py --image /path/to/frame.jpg --diagnostics diagnostics/check
.venv/bin/python ad_detector.py --live --seconds 600 --save-live-diagnostics
```

`ad` is `true`, `false`, or `null` for uncertainty. Logs include processing time, frame age, assumed mute state and confirmation progress. `screen.tracking_used` identifies a geometry tie resolved using a recent measurement.

Mute requires three consecutive positive readings spanning at least one second. Unmute requires four clear negative readings within 12 seconds, spanning at least three seconds; at least half of the observations must be clear and the last two must be clear. Positive Ad readings, detected camera movement, power checks and frame gaps over five seconds reset pending unmute evidence. Unknown alone never sends a command. Frames older than three seconds are not used for decisions.

## Enable IR for one supervised test

Start with audible sound on, verify your raw mute signal separately, and pass your transmitter device and signal file explicitly:

```bash
.venv/bin/python ad_detector.py --live --seconds 600 --save-live-diagnostics \
  --send-ir --tv-starts-unmuted --ir-device /dev/lirc0 --ir-signal /path/to/mute.txt
```

The sender uses one captured frame per transition at 38 kHz and 33% duty cycle, with no repeated toggle burst. A successful driver receipt is **not** an acknowledgement from the TV. The TV mute banner is not OCR'd, and the software cannot independently read mute state. If a command is missed or someone changes mute manually, assumed state can become incorrect. Stopping the program sends no cleanup toggle; the TV may remain muted.

## Optional remote power listener

Copy `power-watcher.example.json` to the ignored `power-watcher.json`. The example code is fictional. Run the listener to capture the four NEC wire bytes from your own remote, then update the configuration:

```bash
cp power-watcher.example.json power-watcher.json
.venv/bin/python power_watcher.py --listen-only --seconds 30
.venv/bin/python power_watcher.py
```

A recognized power press starts a camera check to distinguish TV-on from TV-off; it is not blindly treated as an on command. The ordinary listener is dry-run. An optional systemd template is in `deploy/admuter-power@.service`; it expects a checkout at `/home/<user>/ad-muter` and requires your local configuration and video-device permissions. Do not enable it until manual tests work.

## Data and tests

```bash
python -m unittest -v
```

The public suite includes synthetic geometry, gate timing, power lifecycle, IR send behavior, labeling metadata, model validation and a minimal decision-only replay. Tests requiring private camera photos explicitly skip when those fixtures are absent. Raw room/TV photographs, raw journals, dataset manifests, SSH keys, network addresses, local preferences and slicer profiles are not distributed.

`collect_frames.py` saves human-labeled images locally. `dataset_tools.py` audits datasets and freezes splits by burst/ad break; `train_marker_model.py` trains from reviewed development annotations. Keep new validation footage separate and never use detector predictions as ground truth. See [the model notes](models/README.md) for the reused-image regression results and their limits.

The latest optimizations reduced badge-proposal time by about 11% in a paired Pi benchmark. They also resolved one additional show frame and one processing-timeout ad in the existing dataset. Actual live unmute delay after these changes still needs testing.
