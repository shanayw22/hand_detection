# Real-time Sign Language Detection Application

Develop a real-time application that detects and interprets sign language digits using a combination of object detection and classification models. The project will involve building a pipeline that detects hands in video frames and classifies the detected hand gestures into sign language digits.

### Model Training:

- **Hand Detection:** Train a YOLO model to detect hands in video frames.
- **Sign Language Classification:** Train models (e.g., ResNet architectures) to classify hand gestures into sign language digits.
- Experiment with both transfer learning (pretrained models) and training from scratch where possible.
- Compare performance metrics (accuracy, loss, inference time) across models and modalities.
- Tune the models as needed for best results.

### Deployment:

- Create an interactive application (e.g., using Streamlit) that allows users to:
  - Upload a video or use a webcam to detect and classify sign language gestures in real-time.
  - Display both detection and classification results with model confidence scores.

### Documentation & Submission:

- Submit to the code repository with clear documentation and comments.
- Provide a brief report summarizing your approach, data preprocessing steps, model comparisons, challenges encountered, and final evaluation metrics.
* After two weeks, you will need to submit the following:
  * A link to your GitHub code repository 
  * Make a slide deck (or jupyter notebook), to describe what you did and showcase your results (submit PDF or HTML to Canvas)
  * Do a recording where you walk through and describe your results and presentation (10-20 minutes) (submit the recording to Canvas presentation recording)
  * At the end of the mini-project, a group will be selected at random to do their presentation "live" in front of class (to fuel discussion)

## Appendix

### Dataset Information 

More information on the provided datasets:

#### Hand Detection

The dataset includes images annotated for hand detection, suitable for training YOLO models.

#### Sign Language

The Sign Language MNIST dataset includes images of hand gestures representing digits 0-9 in American Sign Language.

**Inspiration**: How accurately can we interpret sign language gestures in real-time using video input?

**Source**: [Kaggle Sign Language MNIST](https://www.kaggle.com/datasets/datamunge/sign-language-mnist)
