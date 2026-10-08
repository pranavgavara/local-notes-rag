# Fast.ai notes

## Lesson 1
Trained a resnet18 image classifier to tell cricket bats from cricket balls.
Reached 94% accuracy after 3 epochs using fine_tune with base_lr=1e-3.
The learning rate finder suggested a value near 1e-3, so I used that.

## Lesson 2
Deployed the model as a small Gradio app on Hugging Face Spaces.
Biggest mistake: training on photos with white backgrounds only, so it failed on grass.
Fix: add more varied photos and use data augmentation.
