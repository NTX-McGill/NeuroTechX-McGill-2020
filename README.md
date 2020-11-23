# EMGeyboard

The EMGeyboard is a novel control interface that allows users to type words without using a keyboard. It consists of two 3-piece armbands that record electromyography (EMG) signals from a user's forearm muscle activity. The data are wirelessly transmitted to a computer that processes them and converts them to finger movement predictions. Then, consecutive finger predictions are converted into the most likely word the user has typed. The finger movement predictions are visually communicated to the user through a web application, and the predicted words are sent to a Unity game environment, were we implemented a mail application and a music application.

## Inspiration
Artificial reality (AR) technologies open up a whole new realm of possibilities for human-computer interactions. However, most current AR interfaces require the use of specialized handheld controllers to interact with the virtual component of their environment. We set out to address this limitation by designing a device that would allow users to seemlessly transition between interacting with the physical world and interacting with the virtual world. AR interface users can wear our armbands and . Moreover, our device is low-cost: the armband pieces are 3D-printed, we use consumer-grade EMG equipement, and the software we created and/or used is freely available.

<!-- YouTube video link/image -->

## Navigating the repository
* [`hardware`](https://github.com/NTX-McGill/NeuroTechX-McGill-2020/tree/main/hardware)
  contains STL files for 3D-printing our armband pieces.
* [`offline`](https://github.com/NTX-McGill/NeuroTechX-McGill-2020/tree/main/offline)
  contains our data and scripts for data visualization, signal processing and machine learning.
    * [`offline/data`](https://github.com/NTX-McGill/NeuroTechX-McGill-2020/tree/main/offline/data)
      contains raw EMG data and label files, as well as instruction for data collection.
    * [`offlines/signal_processing`](https://github.com/NTX-McGill/NeuroTechX-McGill-2020/tree/main/offline/signal_processing)
      contains scripts to label, filter and epoch our data.
    * [`offline/machine_learning`](https://github.com/NTX-McGill/NeuroTechX-McGill-2020/tree/main/offline/machine_learning)
      contains scripts to extract meaningful features from processed data and train machine learning models.
* [`src`](https://github.com/NTX-McGill/NeuroTechX-McGill-2020/tree/main/src)
  contains the source code for the software we wrote.
    * [`src/dashboard`](https://github.com/NTX-McGill/NeuroTechX-McGill-2020/tree/main/src/dashboard)
      contains installation instructions and source code for our dashboard, a web application with three functions:
      * Prompting users and saving timestamps during data collection
      * Visualizing real-time finger movement predictions
      * Converting finger movements to text and communicating it to other applications
    * [`src/unity_ar`](https://github.com/NTX-McGill/NeuroTechX-McGill-2020/tree/main/src/unity_ar)
      contains the Unity project files for an environment where a user can search for and use a mail or music application.

## Brief overview of the project

*Note: more details about specific parts of the project can be found by using navigation links above.*

### EMG armband
<!-- image: armband piece + electrodes -->
We designed our own EMG armband to allow for flexible electrode placement. It was important for each user to have their own optimal electrode configuration, because otherwise the electrodes could potentially be on completely different muscles for different users; this would produce significantly different EMG signals and make classification essentially impossible. Our armband is made up of a hand piece, a wrist piece, and a forearm piece, and we 3D-printed them using flexible TPU material. Each piece consists of a grid of small holes where electrodes can be secured to their cables. Velcro straps are used to tighten and secure the armband pieces.

### Data collection
<!-- image of armband on hands -->
We used collected finger movement data from 13 participants, using an 8-channel OpenBCI Cyton board, dry Ag/AgCl electrodes, armband pieces made by the hardware team, and the data collection dashboard developed by the software team. We designed several data collection paradigms. For some trials, participants typed on a wireless keyboard; this produced very accurate timestamps for each keypress, but our end-user would not be typing on a physical keyboard (they would be moving their fingers in the air). Hence, for most of our trials, we asked participants the participants to move their fingers in the air when they saw a progress bar on the dashboard move to a specific point; here the accuracy of the timestamp saved by the dashboard depended on the participant's timing accuracy. We collected data from all fingers except the left thumb, because we only had 8 available channels and the left thumb is redundant in a typing interface (since we only need the right thumb for the spacebar).
<!-- image of dashboard prompts -->

### Signal processing
<!-- image of top row of good trials -->
We used a 60 Hz notch filter to remove noise from the power lines, then bandpassed-filtered the signals to only keep activity in the 5-50 Hz frequency band. Then, we epoched the raw data into short overlapping "windows" (epochs) and labelled each one as "baseline" or one of nine finger movement (all fingers except the left thumb).

### Machine learning for finger movement prediction
We used 11 time-domain features and 4 frequency-domain features based on existing literature and our own experiments. We extracted these features from the processed data and tested several types of machine learning models. We obtained the best results with a K-Nearest Neighbours (KNN) model. Logistic regression also performed well during offline evaluation, but real-time testing of the models showed that KNN produced more accurate predictions. 
<!-- confusion matrix ?? -->

### Text prediction
<!-- Terminal -->
We used a dictionary of the 10000 most common English words and converted them to number strings (with each number representing the finger that types this letter in the standard touch-typing mapping). A query number string is updated when a new finger movement is detected by the machine learning model; whenever this happens, we search for matches in our dictionary of 10000 "finger words"; matches are sorted by decreasing usage frequency. If the user moves their thumb ("presses the spacebar"), the first matched word is selected. The user can also enter a word selection mode by clenching both of their fists if they want to select one of the other matched words. In this mode, finger numbers are mapped to the indices of the possible words. Finally, if the user clenches their left fist, the last character of the query string is deleted and, if they clench their right fist, the entire current query string is reset.

### Dashboard
The dashboard was used for data collection for offline analysis and to connect all parts of our real-time pipeline.

For data collection, the dashboard user interface prompted participants to move a finger in the air (or, in some trials, to type a letter on a keyboard). Timestamps for prompts (and keypresses, when available), were saved in a text file, along with information about which finger the participant moved. This was used by the signal processing team to label the EMG data.
<!-- image of dashboard prompts -->

For real-time production, the dashboard received data from the OpenBCI GUI via Lab Streaming Layer, performed filtering and feature extraction on the fly, obtained finger predictions from the machine learning model, and converted them to text. A SocketIO server was used to communicate finger and text predictions to other parts of the pipeline. Finger predictions were displayed on the frontend of the dashboard; this helped evaluate model performance and identify areas of weakness (for example, if there were certain fingers that the model was consistently mispredicting). Predicted text was used as input to the Unity application.
<!-- -->

### Unity application
<!-- screenshot of Unity hub -->
As proof-of-concept for our typing interface, we developed a Unity application where the user can search for an application by its name using our armbands. We also implemented mail and music applications.

## Limitations and future directions
Due to the COVID-19 pandemic, we were no longer able to meet in person. The hardware and data teams were the most affected by this. The hardware team planned on implementing a haptic feedback mechanism for the armband so that the user would know which finger the machine learning model predicted they moved; this could be beneficial because the user could learn how to best move their fingers so that the model predictions are accurate. Data collection could not proceed as before, and we were limited in subject recruitment and the amount of new data we could collect. 

Future directions include further improving our armband prototype (in addition to adding a haptic feedback mechanism), for example by making it easier to attach armband pieces and by adding a cover to protect the electronics. We also aquired a mixed reality headset, and are hoping to experiment with it to test and improve our Unity interface. We are also looking into improving text prediction by using sentence-level contextual information.

## Partners
* [wrnch](https://wrnch.ai/)
* [McGill University Faculty of Engineering](https://www.mcgill.ca/engineering/)
* [Microsoft](https://www.microsoft.com/en-ca)
* [Building 21](https://building21.ca/)

## McGill NeuroTech
We are an interdisciplinary group of dedicated undergraduate students from McGill University and our mission is to raise awareness and interest in neurotechnology, biosignals and human-computer interfaces. For more information, see our [Facebook page](https://www.facebook.com/McGillNeurotech/) or our [website](https://www.mcgillneurotech.com/).
