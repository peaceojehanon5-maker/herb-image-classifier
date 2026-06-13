# Medicinal Herb Image Classifier (CNN)

**Peace Ojehanon** · Glasgow, UK · [LinkedIn](https://www.linkedin.com/in/peace-ojehanon) · peaceojehanon5@gmail.com

A convolutional neural network that classifies images of Chinese medicinal
herbs, built for the **AI & Machine Learning** module of my MSc in Big Data
Technologies (Glasgow Caledonian University, 2024). The notebook works through
the full task: load and prepare the data, train a baseline CNN, then try to
improve it — and, importantly, reports honestly on where the improvements
helped and where they didn't.

## The problem

Multi-class image classification on a **modified subset of the
Chinese-Medicine-163 dataset** — roughly **17,000 images across 11 herb
classes**. The images are messy in realistic ways: varying resolution, blur,
watermarks, and uneven numbers of examples per class. That imbalance and noise,
not the network architecture, turn out to be the hard part.

## Approach

The work is split into the two tasks the coursework asked for:

1. **Baseline CNN** — a straightforward convolutional stack (Conv/pooling
   blocks → dense head) trained on the preprocessed images, to establish a
   reference point.
2. **Enhanced model** — adds **data augmentation** (rotation, shifts, flips,
   zoom) to expand effective training variety and fight overfitting and class
   imbalance, with adjustments to the architecture and training schedule.

Both are evaluated with accuracy, precision, recall and F1, plus a confusion
matrix to see *which* herbs get confused with which.

## Results — read honestly

Training accuracy climbed into the **~78–80%** range over the schedule. The
held-out evaluation is more sobering: generalisation to unseen images is far
weaker than training accuracy, which the notebook discusses directly — the
combination of class imbalance, visually similar herbs and image noise limits
how far a from-scratch CNN of this size can go on this dataset. The point of
the write-up is the *analysis* of that gap and what would close it (more/better
data, transfer learning from a pretrained backbone, class re-balancing), not a
headline accuracy number. The full metrics, plots and confusion matrices are in
the notebook.

> This is a coursework exploration, kept as-is. I'd rather show the real
> evaluation and a clear-eyed discussion than overstate the result.

## Run it

The notebook was written in Google Colab.

```
pip install tensorflow numpy pandas matplotlib scikit-learn pillow
```

Open `herb_classifier_cnn.ipynb` in Colab or Jupyter, point the data-loading
cell at your copy of the dataset, and run top to bottom. A GPU runtime is
strongly recommended — each epoch on CPU is slow.

## Tech

Python · TensorFlow / Keras · NumPy · pandas · scikit-learn · Matplotlib.

## Background

Six years across IT operations, ISP/NOC network engineering and data work;
MSc Big Data Technologies (2025). My day-to-day tooling is Python, SQL and
BigQuery — this project is where I worked through CNN design, augmentation and
honest model evaluation hands-on.
