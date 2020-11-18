# Hardware

Designing and building an EMG armband for McGill Neurotech 2020.

## Design
The design process can go any of a million different directions, so upfront we selected goals to drive our work. 
1. prioritize signal quality
2. be accessible to other researchers (inexpensive, easy to share)

### Process
First, we figured out what the characteristics the armband needed to have to meet our design goals. 

From very early on in data testing, we noticed that different electrode configurations worked better for different people, with positioning ranging from low on the hand to a little over halfway up the forearm. So we could do a simple, non-adjustable wristband like many companies operating in similar spaces have, but that would give us a non-optimal signal. For a better signal, the armband would need to allow for a **wide range of electrode configurations** and **have the option of placing electrodes anywhere from low on the hand to midway up the forearm**. Also, signal quality is better when the electrode has **consistent and strong contact** with skin.

To make our device accessible, it needs to be **inexpensive**, **easy to share** and require **no highly specialized tools** to build.

As secondary priorities, we want the device to be comfortable and unobtrusive; users should be able to do daily tasks while wearing the device without issue.

### Early Design Decisions
1. The armband would primarily be 3D-printed with minimal non-3D-printed components. 
	* 3D printing is cheap, everywhere, and makes it easy to share. All other researchers need to do is download the CAD files.
	* It allows for fast test-adjustment-test sprints.

![3d printing](https://github.com/NTX-McGill/NeuroTechX-McGill-2020/blob/main/hardware/img/3d-printing.jpg?raw=true)

2. The armband would have holes or some kind of mesh to allow electrode placement anywhere on the band.
	* This way, all electrode configurations are supported.

![sample electrode configuration](https://github.com/NTX-McGill/NeuroTechX-McGill-2020/blob/main/hardware/img/electrode-config.jpg?raw=true)

### Iterations
1. We tried various 3D-printing techniques and patterns for a mesh for holding the electrodes.
	* **Takeaways:** flexible TPU worked best, need to balance thickness for comfort and strength, found optimal mesh pattern

![v1](https://github.com/NTX-McGill/NeuroTechX-McGill-2020/blob/main/hardware/img/v1.jpg?raw=true)

2. A single piece of thick 3D-printed polymer wrapped around the wrist like a brace, extending a few inches below the wrist. The material had holes printed around areas where electrodes were usually placed. The armband is held tight by a Velcro strap.
	* **Improvements:** adjustable for different electrode placements, mesh worked well 
	* **Problems:** too thick, doesn't cover entire area where electrodes could be used, single Velcro strap didn't keep the whole band tight, limited mobility

![v2](https://github.com/NTX-McGill/NeuroTechX-McGill-2020/blob/main/hardware/img/v2.jpg?raw=true)

3. We kept the same general design, but made the material thinner (0.7mm) and used multiple Velcro straps.
	* **Improvements:** more stable signal, more comfortable, less limited mobility
	* **Problems:** doesn't cover entire area where electrodes, somewhat limited mobility
4. Extended the armband further down the arm. We couldn't extend very far due to limitations in available 3D-printer bed sizes.
	* **Improvements:** supports more electrode configurations
	* **Problems:** uncomfortable, limited mobility
5. Make the armband three pieces: one lower hand over-thumb piece, one wrist piece, and one forearm piece.
	* **Improvements:** supports more electrode configurations, comfortable, good mobility, good signal strength
	* **Problems:** very minor mobility limitation on hand, covers a large part of the arm

![hand piece](https://github.com/NTX-McGill/NeuroTechX-McGill-2020/blob/main/hardware/img/hand-piece.png?raw=true)
![wrist piece](https://github.com/NTX-McGill/NeuroTechX-McGill-2020/blob/main/hardware/img/wrist-piece.png?raw=true)
![arm piece](https://github.com/NTX-McGill/NeuroTechX-McGill-2020/blob/main/hardware/img/arm-piece.png?raw=true)

## Product
Our final product is a mesh armband 3D-printed with a flexible polymer called TPU. It uses multiple Velcro straps to keep the whole band tight. The armband is in three parts: a palm piece that hooks over the thumb, a wrist piece, and an optional forearm piece to accommodate longer arms/more electrode placements.

See the STL files in this folder.

## Notes for 3D-Printing

We used TPU filament with both Ultimaker and DittoPro printers. Any printer that supports TPU filament should be sufficient for printing this project.

## COVID-19

Unfortunately, the pandemic cut of our access to the workshop and 3D printers, and scattered our team across a few cities. As such, we were not able to fully print our final model, and our design process was cut short. Ideally, we would've continued improving our design to make for a tighter fit without needing as many Velcro straps. We had also wanted to create some sort of cover for the armband to protect the electronics.

![v5](https://github.com/NTX-McGill/NeuroTechX-McGill-2020/blob/main/hardware/img/v5.jpg?raw=true)