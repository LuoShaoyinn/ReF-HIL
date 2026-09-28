# ReF-HIL

**Shaping the Critic around Human Action Neighborhoods for Efficient Human-in-the-Loop Reinforcement Learning**

[Project page](https://anonymous.4open.science/w/ReF-HIL-7762/) ·
[Paper](https://anonymous.4open.science/w/ReF-HIL-7762/assets/paper/ref-hil.pdf) ·
[Code](https://anonymous.4open.science/r/ReF-HIL-D7C2/)

ReF-HIL combines human-reference-guided value shaping with a learned Human
Action Fence for real-world robot learning. It uses initial demonstrations
and online human corrections to guide learning while allowing value-driven
policy improvement within the human action neighborhood.

The project page presents five manipulation tasks: Push-T, cap unscrewing,
gear assembly, plug insertion, and dual-branch cable routing. Each task includes
an edited training-and-evaluation timelapse and three separate recovery scenes.
The full project video presents the method, experiments, and comparisons.

The manuscript reports 90% autonomous success after 18–63 minutes of active
training and final mean autonomous success rates of 91.7–100% across the five
tasks. Training footage includes human assistance; the reported success rates
measure autonomous performance.

The manuscript is under review. Research code is available under GPLv3.
See [acknowledgements and licensing](NOTICE.md) for the project-page materials.
