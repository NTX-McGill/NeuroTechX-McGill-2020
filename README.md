# EMGeyboard

## What is the EMGeyboard?
The EMGeyboard is a novel control interface that allows users to type words without using a keyboard. It consists of two 3-piece armbands that record electromyography (EMG) signals from a user's forearm muscle activity. The data are wirelessly transmitted to a computer that processes them and converts them to finger movement predictions. Then, consecutive finger predictions are converted into the most likely word the user has typed. The finger movement predictions are visually communicated to the user through a web application, and the predicted words are sent to a Unity game environment, were we implemented a mail application and a music application.

## Why did we make it?
Artificial reality (AR) technologies open up a whole new realm of possibilities for human-computer interactions. However, most current AR interfaces require the use of specialized handheld controllers to interact with the virtual component of their environment. We set out to address this limitation by designing a device that would allow users to seemlessly transition between interacting with the physical world and interacting with the virtual world. AR interface users can wear our armbands and . Moreover, our device is low-cost: the armband pieces are 3D-printed, we use consumer-grade EMG equipement, and the software we created and/or used is freely available.

<!-- YouTube video link/image -->

## Navigating the repository
* [`hardware`](https://github.com/NTX-McGill/NeuroTechX-McGill-2020/tree/main/hardware)
  contains STL files for 3D-printing our armband pieces.
* [`offline`](https://github.com/NTX-McGill/NeuroTechX-McGill-2020/tree/main/offline)
  contains our data and scripts for data visualization, signal processing and machine learning.
    * [`offline/data`](https://github.com/NTX-McGill/NeuroTechX-McGill-2020/tree/main/offline/data)
      contains raw EMG data and label files.
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
*Note: for more information on a specific part of the project, refer to the navigation links above.*
    
