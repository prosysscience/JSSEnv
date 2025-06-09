import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

# Path to your event file
event_file = r'c:\Users\mfbie\Projects\JSSEnv\runs\dqn\events.out.tfevents.1748928045.mbiehler.50316.0'

# Load the event file
ea = event_accumulator.EventAccumulator(event_file)
ea.Reload()

# Extract scalars
loss = ea.Scalars('train/loss')
epsilon = ea.Scalars('train/epsilon')
returns = ea.Scalars('episode/return')
lengths = ea.Scalars('episode/length')

# Plotting
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot([x.step for x in loss], [x.value for x in loss])
plt.title('Train Loss')
plt.xlabel('Step')
plt.ylabel('Loss')

plt.subplot(2, 2, 2)
plt.plot([x.step for x in epsilon], [x.value for x in epsilon])
plt.title('Train Epsilon')
plt.xlabel('Step')
plt.ylabel('Epsilon')

plt.subplot(2, 2, 3)
plt.plot([x.step for x in returns], [x.value for x in returns])
plt.title('Episode Return')
plt.xlabel('Step')
plt.ylabel('Return')

plt.subplot(2, 2, 4)
plt.plot([x.step for x in lengths], [x.value for x in lengths])
plt.title('Episode Length')
plt.xlabel('Step')
plt.ylabel('Length')

plt.tight_layout()
plt.show()