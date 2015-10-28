# bee-tracking
Python OpenCV scripts for tracking and analysing 2D honeybee movement in a small arena.

#### Filming Arena
Bees should be filmed under red light in a simple arena enclosed by a Petri dish lid. Reflections should be minimised.

#### Included Scripts
multi_tracker.py performs video processing steps. Default settings were found to be appropriate for filming using a Raspberry Pi NoIR camera at 800x600 resolution from about 20-30cm away.

cond_gen.py should be run prior to multi_tracker.py for batch processing, although multi_tracker can also be run on it's own.

post_process.py provides useful functions for interactively manipulating trajectory data but is not currently implemented to be run as a program.
