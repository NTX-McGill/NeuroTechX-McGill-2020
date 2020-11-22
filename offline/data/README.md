# Data

This folder contains the data we collected from 13 subjects, organized by date.
Two files are produced for each trial; they share the same filename prefix. 
Files whose names include `OpenBCI-RAW` contain the raw EMG data saved by the OpenBCI GUI.
The other files contain timestamps and labels saved by our dashboard.

The `data_visualization` folder contains plots of a random subset of windowed data for each finger movement in a trial,
organized by subject ID. The script to generate these plots is `NeuroTechX-McGill-2020/offline/signal_processing/plot_trials.py`.
We visually inspected these plots and identified trials that were too noisy ("bad" trials) and trials that were particularly clean ("good" trials).
This information can be found in `NeuroTechX-McGill-2020/offline/signal_processing/trials.json`.

## Equipment used
* 3D-printed armband pieces
* Velcro straps
* [Passive dry flat snap Ag/AgCl electrodes](https://www.fri-fl-shop.com/product/disposable-reusable-flat-snap-eegecg-electrode-tde-202/)
* [Electrode cables](https://shop.openbci.com/products/emg-ecg-snap-electrode-cables?variant=32372786958)
* [OpenBCI Cyton board, dongle](https://shop.openbci.com/products/cyton-biosensing-board-8-channel?variant=38958638542), and batteries
* Alcohol swabs
* Medical tape
* Computer with the following pieces of software installed:
  * [OpenBCI Graphical User Interface (GUI)](https://openbci.com/index.php/downloads) v4.2.0 (other versions may also work)
  * McGill NeuroTech 2020 dashboard (see `NeuroTechX-McGill-2020/src/dashboard` for code and instructions)
* Wireless keyboard (for some trials)

## Instructions

### Setting up the OpenBCI board

![Cyton board](https://github.com/NTX-McGill/NeuroTechX-McGill-2020/blob/main/offline/data/img/cyton_board.png?raw=true)

1. Connect the electrode cables to these 17 pins:
   * Bottom row of BIAS (for ground electrode).
   * Top and bottom rows of N1P to N8P.
     We use electrodes in a bipolar configuration, meaning that each channel has its own reference electrode.
2. Connect the USB dongle to the computer.
3. Turn on the board (set the switch to PC).

### Setting up the OpenBCI GUI

1. Launch the OpenBCI GUI.
2. Select **LIVE (from Cyton)** > **Serial (from Dongle)** > **AUTO-CONNECT**. This should start the session.
![GUI start session](https://github.com/NTX-McGill/NeuroTechX-McGill-2020/blob/main/offline/data/img/GUI_start_session.png?raw=true)
3. Click **Start Data Stream** in the top right corner of the GUI.
4. Click on **Hardware Settings** and do the following for all rows:
    * Set the **Bias** column to **Include**.
    * Set the **SRB2** column to **Off**.
    * *Note: in the middle of a session, if the EMG signals for all channels become very noisy and/or big heartbeat artifacts can be seen,
      check these settings again and make sure that they didn't change.*
![GUI hardware seetting](https://github.com/NTX-McGill/NeuroTechX-McGill-2020/blob/main/offline/data/img/GUI_hardware_settings.png?raw=true)

### Putting on the armband

![armband and electrodes](https://github.com/NTX-McGill/NeuroTechX-McGill-2020/blob/main/offline/data/img/armband_electrodes.png?raw=true)

1. Snap the electrodes to the electrode cables by passing them through holes in the armband.
2. Wipe the skin surface and the electrodes with alcohol swabs.
3. For new subjects, we suggest starting with electrodes for one finger at a time and 
   verifying by asking them to move their finger and checking signal quality on the OpenBCI GUI.
4. Dedicate these channels to the following finger(s):
   * **Channel 1**: right thumb.
   * **Channel 2**: right index/middle fingers.
   * **Channel 3**: right ring finger.
   * **Channel 4**: right pinky fingers.
   * **Channel 5**: left index finger.
   * **Channel 6**: left middle finger.
   * **Channel 7**: left ring finger.
   * **Channel 8**: left pinkie finger.
   * *Note: multiple channels may show activity when a single finger is moved. 
      The goal is to place electrodes such that, when different fingers are moved, the overall
      activity pattern (all 8 channels together) is different.*
5. Put the ground electrode on a bony area of the forearm (see electrode labelled **G** in image above)
6. Make sure that there is good skin contact, otherwise the signal can be very noisy.
   Tighter contact can sometimes be achieved by placing the Velcro straps directly above the electrodes when securing the armband.
7. If armband pieces are not available, use medical tape to secure the electrodes instead.
8. Take pictures of the electrode configuration that achieves the best results, and refer to them for quicker setup in the future.

### Using the data collection dashboard

![data collection dashboard](https://github.com/NTX-McGill/NeuroTechX-McGill-2020/blob/main/offline/data/img/data_collection_dashboard.png?raw=true)

1. Launch the data collection dashboard. See `NeuroTechX-McGill-2020/src/dashboard` for instructions.
2. Click on the **DATA COLLECTION** tab.
3. Fill in the **Subject ID** text field.
4. Choose the recording mode. Here is a brief description of what each mode does:
    * **Self-directed**: subject types what they want at their own pace on a keyboard, and the dashboard logs the timestamp of each keypress.
    * **Guided**: subject is prompted every 2 seconds or so to type a key on the keyboard using a specific finger.
      Timestamps of the prompts and the actual keypresses are logged.
    * **In the air**: instead of being prompted to type a key (move a finger), the subjet is prompted for custom gestures, 
      which need to be provided in the given text box. This mode is used to collect data from gestures other than finger movements. 
      Timestamps of the prompts are logged.
    * **Touch-Type**: subjects are prompted to move their fingers based on a custom text input. Timestamps of the prompts are logged.
    * **Guided in the air**: very similar to **Guided**, except the subject moves their fingers "in the air" instead of typing on a keyboard.
      Timestamps of the prompts are logged.
5. Click **START RECORDING**.

#### Touch-typing finger-to-key mapping
To ensure that all participants use the same finger mapping when typing,
ask them to follow the standard touch-typing finger mapping, as depicted below.

![finger mapping](https://github.com/NTX-McGill/NeuroTechX-McGill-2020/blob/main/offline/data/img/finger_mapping.png?raw=true)
*Image source: [Wikipedia Commons](https://commons.wikimedia.org/wiki/File:Italian_keyboard_touchtyping.png)*

### Ending a trial
1. On the dashboard, click **STOP RECORDING**.
2. On the OpenBCI GUI, click **Stop Data Stream**, then select **System Control Panel** > **STOP SESSION**.
3. Move the two data files to the appropriate date subfolder in `NeuroTechX-McGill-2020/offline/data` and rename them.
    * The OpenBCI raw data file is a session subfolder in `~/Documents/OpenBCI_GUI/Recordings`.
    * The log file written by the dashboard is in a date subfolder in `NeuroTechX-McGill-2020/src/dashboard/backend/data`.
