# Motor Fault Predictions

This repository contains deployment code for a 3-phase induction motor fault detection system built around the already-trained XGBoost models in this repo.

## 1. Project Overview

This project uses three RMS current measurements from a 3-phase induction motor:

- `I1`
- `I2`
- `I3`

From those three RMS values, the V2 system generates the richer 16-feature vector used during training:

1. `I1_rms`
2. `I2_rms`
3. `I3_rms`
4. `I1 + I2 + I3`
5. `mean(I1_rms, I2_rms, I3_rms)`
6. `max(abs currents)`
7. `min(abs currents)`
8. `max(abs currents) - min(abs currents)`
9. `std(I1_rms, I2_rms, I3_rms)`
10. `imbalance_percent`
11. `I1_to_mean`
12. `I2_to_mean`
13. `I3_to_mean`
14. `I1_minus_I2`
15. `I2_minus_I3`
16. `I3_minus_I1`

These features are passed into the V2 scaler and models in `trained_models_v2/`.

The deployed system predicts:

- `binary`: Healthy or Faulty
- `severity`: `1u` / `3u` / `5u` when the motor is predicted faulty
- `phase`: Phase 1 / Phase 2 / Phase 3 when the motor is predicted faulty
- `load`: No Load / Half Load / Full Load

The live deployment path uses the rolling decision helper from the V2 notebook, not the older one-shot direct prediction path.

Important: the models are already trained. This deployment setup improves the inference and hardware-integration side only.

## 2. What Is In This Repository

Key files:

- `trained_models_v2/`
- `motor_fault/`
- `motor_monitor.py`
- `test_sensors.py`
- `.env.example`
- `requirements.txt`
- `requirements-rpi.txt`
- `tests/`

## 3. Serial Wiring Supported

The deployment path is now intentionally simpler:

- one sensor per serial device
- no GPIO-based multiplexing
- no reset-pin switching logic

For the current setup, use:

- `I1` on the Raspberry Pi UART path, usually `/dev/serial0`
- `I2` on the first USB-TTL adapter, usually `/dev/ttyUSB0`
- `I3` on the next USB-TTL adapter, usually `/dev/ttyUSB1`

## 4. Hardware You Need

Before deployment, make sure you have:

- Raspberry Pi with Raspberry Pi OS
- microSD card with OS installed
- stable power supply for Raspberry Pi
- 3 current sensors
- motor setup with 3 measurable phase currents
- jumper wires
- common ground where required by your hardware setup
- internet connection if you want ThingSpeak upload

For this repository's default wiring, also prepare:

- Raspberry Pi UART enabled for the first sensor
- one USB-to-TTL adapter for the second sensor
- another USB-to-TTL adapter for the third sensor

## 5. Step-By-Step Deployment Guide

This is the full deployment workflow for actual hardware.

### Step 1: Copy the Project to Raspberry Pi

Move this project folder to the Pi.

Examples:

```bash
scp -r Motor_fault_IOP pi@<RASPBERRY_PI_IP>:/home/pi/
```

or clone it using git if the repository is hosted.

Then log into the Pi:

```bash
ssh pi@<RASPBERRY_PI_IP>
cd /home/pi/Motor_fault_IOP
```

### Step 2: Update the Raspberry Pi

Update packages before installing Python dependencies:

```bash
sudo apt update
sudo apt upgrade -y
```

### Step 3: Install Required System Packages

Install Python tools and serial support:

```bash
sudo apt install -y python3 python3-pip python3-venv
```

If your serial devices need additional support packages, keep the normal Raspberry Pi serial stack enabled.

### Step 4: Confirm the Serial Device Mapping

For the current three-sensor setup, keep the phase mapping:

- `I1` -> Raspberry Pi UART, usually `/dev/serial0`
- `I2` -> USB-TTL adapter, usually `/dev/ttyUSB0`
- `I3` -> second USB-TTL adapter, usually `/dev/ttyUSB1`

This keeps the wiring simple and matches the current code defaults.

### Step 5: Enable UART on the Pi

Run:

```bash
sudo raspi-config
```

Then:

1. Go to `Interface Options`
2. Open `Serial Port`
3. Set `Login shell over serial` to `No`
4. Set `Serial port hardware enabled` to `Yes`
5. Exit and reboot

Reboot:

```bash
sudo reboot
```

After reboot, log in again and return to the project folder.

### Step 6: Connect the Hardware

The exact wiring depends on your sensor module and interface board, but the deployment logic is:

- connect the first sensor to the Pi UART receive path used as `I1`
- connect the second sensor through the USB-TTL adapter used as `I2`
- connect the third sensor through the second USB-TTL adapter used as `I3`
- keep a common ground where your hardware requires it

Important deployment rule:

- keep your phase naming consistent all the way through wiring, variable setup, and testing

That means:

- the sensor physically measuring motor phase 1 must remain `I1`
- the sensor physically measuring motor phase 2 must remain `I2`
- the sensor physically measuring motor phase 3 must remain `I3`

### Step 7: Create a Python Virtual Environment

Inside the project directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

You should now see your shell running inside `.venv`.

### Step 8: Install Python Dependencies

On Raspberry Pi use:

```bash
pip install --upgrade pip
pip install -r requirements-rpi.txt
```

This installs:

- `numpy`
- `joblib`
- `xgboost`
- `scikit-learn`
- `requests`
- `pyserial`
- `pytest`

### Step 9: Create Your Deployment Variables File

Copy the template:

```bash
cp .env.example .env
```

Now open `.env` in an editor:

```bash
nano .env
```

### Step 10: Fill In the Variables in `.env`

Below is what each variable means.

#### Serial settings

```bash
BAUD_RATE=9600
SERIAL_TIMEOUT=1.0
```

Use these for:

- serial baud rate
- read timeout

Normally keep `BAUD_RATE=9600` unless your hardware requires something else.

#### Sample timing

```bash
SAMPLE_INTERVAL=1
```

This is how often the monitor performs one inference cycle for live on-screen monitoring.

If you later enable ThingSpeak, you can increase this value to reduce upload frequency.

#### Rolling decision settings

```bash
PREDICTION_CONFIDENCE_THRESHOLD=0.60
ROLLING_WINDOW_SIZE=5
FAULT_VOTES_REQUIRED=3
MAX_SAFE_CURRENT=
```

Use these for:

- minimum confidence before the binary result is treated as decisive
- rolling history length
- number of recent `Faulty` votes required to promote the rolling decision to `Faulty`
- optional immediate overcurrent cutoff

#### Serial port variables

```bash
I1_PORT=/dev/serial0
I2_PORT=/dev/ttyUSB0
I3_PORT=/dev/ttyUSB1
```

These should match the actual serial devices on your Raspberry Pi.

For the default three-sensor setup:

- keep `I1_PORT=/dev/serial0`
- keep `I2_PORT=/dev/ttyUSB0`
- keep `I3_PORT=/dev/ttyUSB1`

To check connected serial devices:

```bash
ls /dev/ttyUSB*
```

If the device names change after reboot or reconnection, you may later want persistent udev rules, but first get the system working with direct paths.

#### ThingSpeak variables

```bash
THINGSPEAK_ENABLED=false
THINGSPEAK_API_KEY=REPLACE_WITH_YOUR_THINGSPEAK_WRITE_API_KEY
THINGSPEAK_URL=https://api.thingspeak.com/update
```

Use:

- `THINGSPEAK_ENABLED=false` while first validating hardware locally
- `THINGSPEAK_ENABLED=true` after sensor readings and predictions are stable
- replace `THINGSPEAK_API_KEY` with the real write API key

### Step 11: Export the Variables Into the Shell

The current code reads environment variables directly. A simple way to load them is:

```bash
set -a
source .env
set +a
```

Run these commands in every new shell session before starting the app, unless you later automate it with `systemd` or a shell profile.

### Step 12: Confirm the Model Files Exist

Check that these files are present:

```bash
ls
```

You should see:

- `scaler.joblib`
- `model_binary.joblib`
- `model_severity.joblib`
- `model_phase.joblib`
- `model_load.joblib`

The code automatically searches:

1. `MODEL_DIR` if you set it
2. `trained_models_v2/`
3. `trained_models/`
4. repository root

So with the current repository structure, the V2 models are used automatically.

### Step 13: Run a Pure Software Inference Test First

Before testing sensors, verify that Python, scaler, and model loading all work:

```bash
python motor_monitor.py predict --i1 -2.23046875 --i2 0.51171875 --i3 1.58984375
```

This should print the raw task outputs together with the confidence-aware current prediction summary.

If this step fails, do not move to hardware testing yet. Fix the Python environment first.

### Step 14: Check That the Pi Can See the Serial Devices

Run:

```bash
ls -l /dev/serial0
ls /dev/ttyUSB*
```

If your expected serial devices are missing:

- re-check wiring
- re-check UART enable settings
- re-check USB adapter detection
- re-check power to the sensors

### Step 15: Run the Raw Sensor Test

This is the most important hardware validation step before live monitoring.

Run:

```bash
python test_sensors.py
```

This reads and prints raw current values without running the full monitoring loop.

What you want to confirm:

- all three configured channels return values
- the values are not empty
- the values are not random garbage text
- the phase-to-sensor mapping is correct

If the motor is off, values may be close to zero.

If the motor is on, you should see meaningful current readings for all three phases.

### Step 16: Troubleshoot Sensor Read Problems If Needed

If `test_sensors.py` does not work, check these one by one.

If you get no output:

- check power to sensor modules
- check ground reference
- check serial wiring
- check selected serial device path
- check UART enable on Pi

If you get unreadable characters:

- check baud rate
- check serial line wiring
- check whether the sensor output format matches the expected ASCII numeric format

If only one sensor works:

- check `/dev/serial0`, `/dev/ttyUSB0`, and `/dev/ttyUSB1` one by one
- swap adapters or cables to isolate whether the issue is the sensor or the USB interface
- confirm the sensor still streams plain ASCII numeric values

Only proceed to model inference after all three currents can be read reliably.

### Step 17: Run One Full Inference Cycle

Once the third sensor is connected and `I3_PORT` is set, run:

```bash
python motor_monitor.py run --once
```

This performs:

1. sensor read
2. V2 RMS feature generation
3. confidence-aware model inference
4. rolling decision update
5. optional ThingSpeak upload

What to verify:

- all three currents appear in output
- all prediction tasks are present
- no serial read exception occurs
- no model loading exception occurs

### Step 18: Start Continuous Monitoring

When the one-cycle test is good, run:

```bash
python motor_monitor.py --interval 1
```

This starts the continuous loop and prints each cycle on screen as:

- the three sensor currents
- the current V2 prediction
- the rolling decision summary and vote counts

If you omit `--interval`, the app uses `SAMPLE_INTERVAL` from `.env`.

### Step 19: Enable ThingSpeak Only After Local Validation

Once local monitoring is stable:

1. edit `.env`
2. set:

```bash
THINGSPEAK_ENABLED=true
THINGSPEAK_API_KEY=YOUR_REAL_API_KEY
```

3. reload the environment:

```bash
set -a
source .env
set +a
```

4. run:

```bash
python motor_monitor.py run --once
```

Then verify that data appears in your ThingSpeak channel.

### Step 20: Run Tests on the Pi

If you want to verify the software stack on the Pi:

```bash
pytest -q
```

These tests validate:

- feature engineering
- model loading
- sensor value parsing
- ThingSpeak payload formatting

They do not test the real physical hardware connections.

## 6. Recommended Real Deployment Order

Follow this exact order:

1. move project to Pi
2. install OS packages
3. create `.venv`
4. install `requirements-rpi.txt`
5. fill `.env`
6. export variables
7. run software-only prediction command
8. confirm serial devices are visible
9. run `python test_sensors.py`
10. fix all sensor issues
11. run `python motor_monitor.py run --once`
12. run `python motor_monitor.py`
13. enable ThingSpeak after local validation succeeds

## 7. Important Notes About Real Hardware Deployment

### Keep phase wiring consistent

Do not change phase naming between:

- the physical wiring
- the variable names
- the interpretation of predictions

If you accidentally swap `I1`, `I2`, and `I3`, the model may still produce outputs, but the predicted faulty phase could be misleading relative to the actual motor phase.

### Validate with healthy condition first

Before testing faulty conditions:

- first run the system on a healthy motor state
- confirm readings are stable
- confirm the app runs continuously without sensor failures

### Add cloud upload last

Do not begin with cloud upload enabled.

First confirm:

- serial reading works
- predictions are produced
- loop timing is stable

Then enable cloud upload.

### Saved model compatibility

The local validation was done successfully with:

- `xgboost==2.0.3`
- `scikit-learn==1.6.1`

These versions are pinned in the requirements files for that reason.

## 8. Useful Commands

Activate environment:

```bash
source .venv/bin/activate
```

Load `.env` variables:

```bash
set -a
source .env
set +a
```

Software-only prediction:

```bash
python motor_monitor.py predict --i1 -2.23046875 --i2 0.51171875 --i3 1.58984375
```

Raw sensor test:

```bash
python test_sensors.py
```

Run one full cycle:

```bash
python motor_monitor.py run --once
```

Run continuously:

```bash
python motor_monitor.py
```

Run tests:

```bash
pytest -q
```

## 9. If You Want To Automate Startup Later

After you validate everything manually, the next natural step is to create a `systemd` service so the monitor starts automatically on boot.

That is not included yet in this README because manual deployment and sensor validation should come first.
