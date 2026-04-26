====================================================================
  AUTOMATED ATTENDANCE SYSTEM — RASPBERRY PI 5 SETUP GUIDE
====================================================================
Project   : Face Recognition Classroom Attendance System
Hardware  : Raspberry Pi 5 (4GB / 8GB RAM)
OS        : Raspberry Pi OS (Bookworm, 64-bit Desktop recommended)
Author    : IDP Group / chan
====================================================================


--------------------------------------------------------------------
SECTION 1 — BEFORE YOU START
--------------------------------------------------------------------

Make sure you have the following ready:

  Hardware:
  [ ] Raspberry Pi 5 (4GB or 8GB recommended)
  [ ] MicroSD card (32GB or larger, Class 10 / A2 rated)
  [ ] Official Raspberry Pi 27W USB-C power supply
      (IMPORTANT: do NOT use a phone charger — Pi 5 needs 5V/5A)
  [ ] USB webcam OR Raspberry Pi Camera Module 3
  [ ] HDMI monitor + keyboard + mouse (for first-time setup)
  [ ] Active cooling (heatsink + fan) — strongly recommended

  Software on your Windows PC:
  [ ] AnyDesk installed (remote desktop access)
  [ ] SSH enabled on the Pi (see Section 2)


--------------------------------------------------------------------
SECTION 2 — ENABLE SSH ON RASPBERRY PI
--------------------------------------------------------------------

Open a terminal on the Pi and run:

    sudo systemctl enable ssh
    sudo systemctl start ssh

Check it is running:

    sudo systemctl status ssh

You should see "active (running)" in green.

Find your Pi's IP address:

    hostname -I

Note this IP — you will need it to transfer files from your PC.


--------------------------------------------------------------------
SECTION 3 — TRANSFER FILES FROM WINDOWS PC TO PI
--------------------------------------------------------------------

On your Windows PC, open Command Prompt and run:

    scp -r "C:\Users\jian6\OneDrive\Documents\automated_attendance" chan@<PI_IP>:/home/chan/

Replace <PI_IP> with the IP from hostname -I (e.g. 192.168.1.105).

NOTE: Do NOT copy the .venv folder — it is Windows-only and will
not work on Linux. Only copy the source code files.

If the transfer keeps dropping, zip first (on Windows CMD):

    cd "C:\Users\jian6\OneDrive\Documents\automated_attendance"
    tar -czf attendance_code.tar.gz --exclude=".venv" --exclude="__pycache__" --exclude="*.pyc" admin admin_gui.py anti_spoof*.py attendance_logger.py attendance_logs database embeddings enroll.py face_utils.py main*.py recognition_state.py requirements.txt schedule_manager.py ui_overlay.py download_weights*.py

    scp attendance_code.tar.gz chan@<PI_IP>:/home/chan/

Then on the Pi terminal:

    mkdir -p /home/chan/automated_attendance
    tar -xzf /home/chan/attendance_code.tar.gz -C /home/chan/automated_attendance/
    ls /home/chan/automated_attendance/


--------------------------------------------------------------------
SECTION 4 — INSTALL SYSTEM DEPENDENCIES
--------------------------------------------------------------------

On the Pi terminal, run:

    sudo apt update
    sudo apt upgrade -y
    sudo apt install -y \
        python3-venv \
        python3-pip \
        python3-opencv \
        libopenblas-dev \
        libjpeg-dev \
        libatlas-base-dev \
        libhdf5-dev \
        git \
        cmake

NOTE: libatlas-base-dev may show "no installation candidate" on
Raspberry Pi OS Bookworm — this is normal, skip it if it fails.
libopenblas-dev is the replacement and will be installed instead.


--------------------------------------------------------------------
SECTION 5 — CREATE PYTHON VIRTUAL ENVIRONMENT
--------------------------------------------------------------------

Navigate to the project folder:

    cd /home/chan/automated_attendance

Create the virtual environment:

    python3 -m venv .venv

Activate it (LINUX command — NOT activate.bat):

    source .venv/bin/activate

You will see (.venv) appear at the start of the prompt.
You must run this activation command every time you open a new
terminal before running any Python scripts.

Upgrade pip:

    pip install --upgrade pip


--------------------------------------------------------------------
SECTION 6 — INSTALL PYTHON PACKAGES
--------------------------------------------------------------------

Step 1 — Install PyTorch (CPU-only build for ARM64):

    pip install torch==2.2.2 torchvision==0.17.2 \
        --index-url https://download.pytorch.org/whl/cpu

    WARNING: This will take 10–30 minutes on the Pi.
    Do NOT close the terminal. Let it finish completely.

Step 2 — Install remaining packages:

    pip install facenet-pytorch
    pip install opencv-python-headless
    pip install onnxruntime
    pip install pillow

Step 3 — Verify installation:

    python3 -c "import torch; import cv2; import facenet_pytorch; print('All OK')"

You should see: All OK


--------------------------------------------------------------------
SECTION 7 — DOWNLOAD ANTI-SPOOFING MODEL WEIGHTS
--------------------------------------------------------------------

If you plan to use the anti-spoofing feature, download the weights:

    cd /home/chan/automated_attendance
    source .venv/bin/activate
    python3 download_weights.py

This downloads MiniFASNetV2.onnx into the anti_spoof_weights/ folder.

NOTE: Anti-spoofing is DISABLED by default in main.py
(LIVENESS_THRESHOLD = None). You can leave it disabled for now.


--------------------------------------------------------------------
SECTION 8 — SET UP TIMETABLE AND STUDENTS (ADMIN GUI)
--------------------------------------------------------------------

Run the admin dashboard:

    cd /home/chan/automated_attendance
    source .venv/bin/activate
    python3 admin_gui.py

This opens a window with 3 tabs:

  Tab 1 — Timetable:
    Add your course schedule (e.g. EE4001, Monday, 08:00–10:00)
    The system uses this to know which course is running right now.

  Tab 2 — Students:
    Add enrolled students for each course (name + matrix number).
    You can also import from a CSV file with columns: name, matrix_number

  Tab 3 — Attendance Log:
    View and filter attendance records by date and course.
    Export to CSV for submission.

CSV files are saved to:
    admin/timetable.csv
    admin/students/<COURSE_CODE>.csv


--------------------------------------------------------------------
SECTION 9 — ENROL STUDENT FACES
--------------------------------------------------------------------

Before running the main system, each student's face must be enrolled.

    cd /home/chan/automated_attendance
    source .venv/bin/activate
    python3 enroll.py

Follow the prompts:
  - Enter student name (e.g. Ahmad Faris)
  - Enter matrix number (e.g. A12345)
  - The webcam opens — the student looks at the camera
  - 15 face samples are captured automatically
  - Press Q when done

Face data is saved to:
    embeddings/<matrix_number>.npy
    database/students.json

Repeat for every student in the class.


--------------------------------------------------------------------
SECTION 10 — RUN THE ATTENDANCE SYSTEM
--------------------------------------------------------------------

    cd /home/chan/automated_attendance
    source .venv/bin/activate
    python3 main.py

What happens:
  1. System loads face models (takes ~20 seconds on first run)
  2. Checks timetable for the currently active course
  3. Opens the camera window
  4. When a student stands in front of the camera:
     - Yellow box = scanning (hold still for 2.5 seconds)
     - Green box  = attendance recorded
     - Red box    = unknown face
     - Blue box   = already recorded today

  5. Attendance is saved to:
     attendance_logs/attendance_YYYY-MM-DD.csv

  Columns: Name, Matrix Number, Course Code, Course Name, Date, Time, Status
  Status will be "Present" (within 10 min of class start) or "Late"

Press Q to quit the system.


--------------------------------------------------------------------
SECTION 11 — AUTO-START ON BOOT (OPTIONAL)
--------------------------------------------------------------------

To make the system start automatically when the Pi boots:

    sudo nano /etc/rc.local

Add this line BEFORE "exit 0":

    su -c "cd /home/chan/automated_attendance && source .venv/bin/activate && python3 main.py &" chan

Save (Ctrl+O, Enter) and exit (Ctrl+X).

Or use a systemd service for more control:

    sudo nano /etc/systemd/system/attendance.service

Paste:

    [Unit]
    Description=Face Recognition Attendance System
    After=network.target

    [Service]
    User=chan
    WorkingDirectory=/home/chan/automated_attendance
    ExecStart=/home/chan/automated_attendance/.venv/bin/python3 main.py
    Restart=on-failure
    Environment=DISPLAY=:0

    [Install]
    WantedBy=multi-user.target

Enable and start:

    sudo systemctl daemon-reload
    sudo systemctl enable attendance
    sudo systemctl start attendance


--------------------------------------------------------------------
SECTION 12 — PERFORMANCE TIPS FOR PI 5
--------------------------------------------------------------------

1. Lower camera resolution for better FPS:
   In main.py, change:
       FRAME_W = 320
       FRAME_H = 240
   Expected: ~5–8 FPS (enough for door entry use)

2. Set CPU to performance mode:
       sudo apt install -y cpufrequtils
       sudo cpufreq-set -g performance

3. Use active cooling:
   Pi 5 runs hot under sustained AI load. Use the official
   Raspberry Pi Active Cooler or a heatsink + fan.

4. Use a fast MicroSD card (A2 rated) or boot from USB SSD
   for much faster model loading times.

5. Close unnecessary apps (browser, file manager) before running.


--------------------------------------------------------------------
SECTION 13 — USING PI CAMERA MODULE (INSTEAD OF USB WEBCAM)
--------------------------------------------------------------------

If you are using the official Raspberry Pi Camera Module 3:

    pip install picamera2

Then in main.py, replace the camera section with:

    from picamera2 import Picamera2
    import numpy as np

    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(
        main={"format": "RGB888", "size": (640, 480)}
    ))
    picam2.start()

    # In the main loop, replace cap.read() with:
    frame = picam2.capture_array()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

Contact your lecturer or open an issue if you need help with this.


--------------------------------------------------------------------
SECTION 14 — TROUBLESHOOTING
--------------------------------------------------------------------

Problem : "source .venv/bin/activate.bat: No such file or directory"
Solution: You used the Windows command. On Linux use:
              source .venv/bin/activate     (no .bat)

Problem : "cmd: command not found"
Solution: cmd is a Windows command. On Linux just type commands
          directly in the terminal.

Problem : "Could not open camera index 0"
Solution: Check if camera is detected:
              ls /dev/video*
          If you see /dev/video1, change CAMERA_INDEX = 1 in main.py

Problem : "No course scheduled right now"
Solution: The current time is not within any timetable slot.
          Run admin_gui.py and add a slot covering the current time.

Problem : PyTorch install fails or takes forever
Solution: Run this exact command:
              pip install torch==2.2.2 torchvision==0.17.2 \
                  --index-url https://download.pytorch.org/whl/cpu
          Be patient — 20-30 minutes is normal on the Pi.

Problem : Low FPS / camera lag
Solution: Lower resolution in main.py:
              FRAME_W = 320
              FRAME_H = 240
          Also close other applications.

Problem : "ModuleNotFoundError: No module named 'cv2'"
Solution: The virtual environment is not activated, or opencv is
          not installed. Run:
              source .venv/bin/activate
              pip install opencv-python-headless

Problem : Screen is blank / black when running via AnyDesk
Solution: Make sure DISPLAY environment variable is set:
              export DISPLAY=:0
          Then run main.py again.

Problem : SCP transfer keeps dropping
Solution: Zip the files first and transfer the zip.
          See Section 3 for zip commands.


--------------------------------------------------------------------
SECTION 15 — QUICK REFERENCE COMMANDS
--------------------------------------------------------------------

Every time you open a terminal:

    cd /home/chan/automated_attendance
    source .venv/bin/activate

Run admin GUI:
    python3 admin_gui.py

Enrol a student:
    python3 enroll.py

Run attendance system:
    python3 main.py

Check today's attendance log:
    cat attendance_logs/attendance_$(date +%Y-%m-%d).csv

View all log files:
    ls attendance_logs/

Deactivate virtual environment when done:
    deactivate


====================================================================
  END OF RASPBERRY PI 5 SETUP GUIDE
====================================================================
