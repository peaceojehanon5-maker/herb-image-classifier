# Medicinal Herb Image Classifier (CNN)

**Peace Ojehanon** · Glasgow, UK · [LinkedIn](https://www.linkedin.com/in/peace-ojehanon) · peaceojehanon5@gmail.com

A convolutional neural network that classifies images of Chinese medicinal
herbs, built for the **Artificial Intelligence and Machine Learning** module of
my MSc in Big Data Technologies (Glasgow Caledonian University, 2024). The
notebook loads and prepares the data, trains a baseline CNN, then works on
improving it.

## The data

A **modified subset of the Chinese-Medicine-163 dataset**: about **17,000
images across 11 herb classes**, with a similar number of images in each class.
The images vary a lot in quality, with blur, watermarks and busy backgrounds.

## Approach

The work follows the two tasks the coursework set:

1. **Baseline CNN.** A small stack of convolution and pooling layers with a
   dense head, trained on the prepared images to give a reference point.
2. **Enhancing the model.** Stronger data augmentation (rotation, shifts,
   shear, zoom and flips) and a lower learning rate with early stopping. The
   notebook also defines a deeper architecture with batch normalisation,
   dropout, L2 regularisation and global average pooling, but never trains it.

The notebook scores the models with accuracy, precision, recall and F1, plus
confusion matrices to see which herbs get mixed up.

## Results

Every output is kept in the notebook as it ran. Keras's own validation
accuracy was about 60% for the baseline and about 70% after retraining with
stronger augmentation and a lower learning rate.

One known problem: the scikit-learn reports near the end score close to
chance, which doesn't match the Keras figures. The validation generator
shuffles the images, so those reports compare the predictions with the labels
in a different order. Setting `shuffle=False` on the validation generator would
fix it.

## Run it

```
pip install tensorflow numpy matplotlib scikit-learn pillow
```

Open `herb_classifier_cnn.ipynb` in Jupyter or Google Colab and point the
data-loading cell at your copy of the dataset. A few later cells use variable
names that aren't defined earlier in the notebook (`model`, `aug_train_data`,
`val_data`), so expect to fix those before it runs from top to bottom. A GPU
runtime is strongly recommended, as each epoch on a CPU is slow.

## Tech

Python · TensorFlow / Keras · NumPy · scikit-learn · Matplotlib.

## Background

I spent three years in an ISP's network operations centre before my MSc in
Big Data Technologies (Merit, 2025). For data work I mostly use Python, SQL and
BigQuery; this project is where I worked through CNN design and data
augmentation hands-on.
