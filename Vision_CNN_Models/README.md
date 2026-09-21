# Driver Drowsiness Detection Using MobileNetV2

This project develops an image-classification model for detecting whether a driver is **drowsy** or **non-drowsy**. The notebook compares a baseline convolutional neural network (CNN), a CNN with data augmentation, and a transfer-learning model based on **MobileNetV2**.

## Project Contents

- `Driver_Drowsiness.ipynb` - end-to-end data preparation, model training, evaluation, and TensorFlow Lite export notebook.
- `dataset.zip` - image dataset archive containing `train` and `test` directories organized by class.
- `drowsiness_detection_mobilenetv2.tflite` - generated TensorFlow Lite model after running the export cell.

> The current notebook file is named `Driver_Drowsiness.ipynb`. If it is renamed to `Driver_Drowsiness_Detection.ipynb`, update references to the filename accordingly.

## Notebook Workflow

1. Download or locate the image dataset.
2. Extract the archive and remove macOS metadata files such as `__MACOSX` and `._*`.
3. Load the `train` and `test` folders with TensorFlow.
4. Resize images to `128 x 128` pixels and use batches of `32`.
5. Inspect class distribution and sample images.
6. Train and evaluate a baseline CNN.
7. Train a second CNN with horizontal flipping, rotation, zoom, brightness, and contrast augmentation.
8. Train a MobileNetV2 transfer-learning model pretrained on ImageNet.
9. Export the MobileNetV2 model to an optimized TensorFlow Lite file.
10. Test the TFLite model on a sample image and evaluate it across the test set using accuracy, precision, recall, F1-score, and a confusion matrix.

## Requirements

- Python 3.9 or newer
- Jupyter Notebook or Google Colab
- TensorFlow
- NumPy
- pandas
- Matplotlib
- scikit-learn
- seaborn
- gdown

Install the Python dependencies with:

```bash
pip install tensorflow numpy pandas matplotlib scikit-learn seaborn gdown
```

## Dataset Layout

The extracted dataset should contain the following structure:

```text
dataset/
├── train/
│   ├── drowsy/
│   └── non_drowsy/
└── test/
    ├── drowsy/
    └── non_drowsy/
```

TensorFlow infers the class labels from the directory names. Keep the folder names consistent with the dataset used by the notebook.

## Running the Notebook

### Google Colab

1. Open the notebook in Google Colab.
2. Run the cells from top to bottom.
3. The notebook downloads the dataset using `gdown` and extracts it under `/content`.
4. Confirm that the extracted archive contains the `train` and `test` directories before creating the TensorFlow datasets.
5. Run the training, evaluation, and TFLite export cells in order.

### Local Jupyter

1. Place `dataset.zip` in the working directory.
2. Extract it and set `actual_dataset_dir` to the directory containing `train` and `test`.
3. Run the notebook cells sequentially.
4. The exported model is saved as `drowsiness_detection_mobilenetv2.tflite` in the current working directory.

The notebook was written with Colab paths such as `/content/dataset.zip`. These paths must be changed when running locally.

## Model Details

The final transfer-learning model uses:

- MobileNetV2 without its ImageNet classification head
- Frozen pretrained convolutional base
- Global average pooling
- A dense layer with 128 units
- Dropout with a rate of `0.5`
- A softmax output layer for the dataset classes
- Adam optimizer and sparse categorical cross-entropy loss
- Ten training epochs by default

The notebook also applies MobileNetV2 preprocessing, converting input pixel values from `[0, 1]` to `[-1, 1]` before inference.

## TensorFlow Lite Inference

After exporting the model, the notebook uses `tf.lite.Interpreter` to:

1. Load `drowsiness_detection_mobilenetv2.tflite`.
2. Resize and preprocess a test image.
3. Run inference.
4. Select the class with the highest predicted probability.
5. Display the predicted class and confidence.

The TFLite model is intended as an edge-deployment artifact. It should be tested on representative real-world data before being used in any driver-safety system.

## Limitations

- The notebook evaluates image classification rather than continuous video or live camera input.
- Results depend on dataset quality, class balance, lighting, camera angle, and subject diversity.
- No accuracy or other metric is stated here because results depend on the exact dataset and training run.
- This project is for experimentation and should not be treated as a certified safety-critical driver-monitoring system.
