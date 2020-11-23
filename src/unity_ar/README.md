# AR - Unity
The AR frontend interface to demonstrate an application of the EMGeybard.

## Description
The idea behind the AR frontend was to show how the EMGeyboard redefines the way humans can interact with computers.
We intended to create an AR frontend where users can discretely interact with everyday apps. For example, browse and send emails, or pull up and change music being listened to.

## Installation and Setup
l. Install unity
l. Open up the console scene
l. Ensure the socketIO components server URL property matches the URL of the socketIO server (backend.py sets the server up as 127.0.0.1:4002)
l. Make sure the backend python script is running and the socketIO server is up
l. Hit the play button

## Usage
The main scene is the console scene which has the main user interface where users would be able to open up apps from a list of favorites:

The keys D,F,J,K each map to opening up one of the favorites.
Alternatively, the space bar can be pressed (moving the thumb) and the user has the ability to search from all the apps and open one up.
While searching, the matched options show up above a letter which when pressed opens up the corrosponding app.
Apps open up scenes additively. These scenes can then update options and what letters pressed do in the console scene via the exposed public KeyboardWindowManager game object.
