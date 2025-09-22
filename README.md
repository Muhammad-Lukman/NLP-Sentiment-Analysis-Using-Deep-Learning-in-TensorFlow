# NLP-Sentiment Analysis Using Deep Learning in TensorFlow

A comprehensive exploration of sentiment analysis using various deep learning architectures - from **Simple RNNs** to **GRU, LSTM, 1D Convolutions**, and **pretrained Word2Vec embeddings**. This project demonstrates step-by-step model building, training, evaluation, and deployment-ready inference in TensorFlow/Keras.

## **Project Overview**

This repository contains experiments and models for sentiment analysis on movie reviews. The main goal is to predict whether a review is **positive** or **negative**, using different neural network architectures:

- **RNN (Simple Recurrent Neural Network)**
- **LSTM (Long Short-Term Memory)**
- **GRU (Gated Recurrent Unit)**
- **1D Convolutional Neural Network (Conv1D)**
- **Conv1D + Pretrained Word2Vec embeddings**

The workflow includes:

1. Preprocessing text using **vectorization and padding**.
2. Building models with TensorFlow/Keras.
3. Training with **binary crossentropy** and **Adam optimizer**.
4. Saving **best models** with `ModelCheckpoint`.
5. Plotting **accuracy and loss curves**.
6. Making **inference-ready models** for deployment.
7. Visualizing embeddings in **TensorBoard projector**.

## **Text Preprocessing & Representation**

Before training a model, one thing you must know is that the Raw text can’t go directly into deep learning models - it first needs to be cleaned, standardized, and converted into numerical form. This project also walks through several key preprocessing steps, you should read them and get familiar with there pros and cons, (here I'll discuss them in brief, if you want more explanation please read the notebook):

### **1. Standardization**

* All reviews are **lowercased** so “In” and “in” aren’t treated as different words.
* **HTML tags** like `<br />` are stripped out with `tf.regex_replace`.
* **Punctuation, special characters, and accented characters** are removed.
* **Stemming** (using NLTK’s `PorterStemmer`) reduces words to their root form:
  
  > “discussion”, “discussed”, “discussing” → “discuss”
* **Lemmatization** (preferred) ensures context-aware root words:
  
  > “better” → “good”

### **2. Tokenization**

* **Character-level**: splits text into individual letters.
* **Word-level**: splits sentences into words.
* **Sub-word tokenization**: breaks words into meaningful chunks (`"actor"` → `"act"`, `"or"`).
* **n-gram tokenization**: captures short phrases (bi-grams: `"I love"`, `"love this"`).
* Unknown or rare words are mapped to a special `<UNK>` token.

### **3. Numericalization**

* **One-Hot Encoding**: each token → sparse vector (mostly zeros).
* **Bag of Words (BoW)**: counts how often each word appears, ignoring order.
* **TF–IDF**: scales word counts so rare/important words matter more than common ones.
* **Embeddings**: words are mapped into dense, low-dimensional vectors that capture semantic meaning (e.g., `king – man + woman ≈ queen`).

### **4. Word Embeddings**

* Compared different representations:
  * **Trainable embeddings** (learned during model training).
  * **Pretrained Word2Vec (Google News, 300-dim)** for richer, semantic vectors.
* Embeddings were also **visualized in TensorBoard’s 3D Projector**, showing how words with similar meaning cluster together.

## **Models and Architectures**

### **1. Simple RNN**
- Input → Embedding → SimpleRNN → Dense (Sigmoid)
- Pros: Easy to implement.
- Cons: Suffers from **vanishing gradients** on long sequences.

### **2. LSTM (Bidirectional, Multi-layer)**
- Input → Embedding → BiLSTM(64) → BiLSTM(32) → Dense → Dropout → Dense(Sigmoid)
- Handles long-term dependencies.
- Incorporates **forget, input, output gates** for memory control.

### **3. GRU (Bidirectional, Multi-layer)**
- Input → Embedding → BiGRU(64) → BiGRU(32) → Dense → Dropout → Dense(Sigmoid)
- Lightweight alternative to LSTM.
- Combines forget/input gates into **update gate**.

### **4. 1D Convolution (Conv1D)**
- Input → Embedding → Conv1D → Flatten → Dense → Dropout → Dense(Sigmoid)
- Captures **local n-gram patterns**.
- More parallelizable than RNNs.

### **5. Conv1D + Pretrained Word2Vec**
- Uses **Google News pretrained embeddings (300-dim)**.
- Embedding layer initialized with **pretrained vectors**.
- Fine-tunable for improved downstream performance.

## **Installation & Dependencies**

```bash
# Install TensorFlow, Gensim and other dependencies
pip install tensorflow numpy matplotlib gensim
````

Optional for TensorBoard:

```bash
pip install tensorboard
```

## **Usage**

1. Clone this repository:

```bash
git clone https://github.com/Muhammad-Lukman/NLP-Sentiment-Analysis-Using-Deep-Learning-in-TensorFlow.git
cd NLP-Sentiment-Analysis-Using-Deep-Learning-in-TensorFlow.git
```

2. Open the **notebooks** in Jupyter or Colab:

```bash
notebooks/Text_Preprocessing_for_Sentiment_analysis.ipynb
```

3. Load pretrained models for inference:

```python
from keras.models import load_model

model_rnn = load_model('models/lstm.h5')
model_gru = load_model('models/gru.h5')
model_con1d = load_model('models/con1d.h5')
model_conv_word2vec = load_model('models/conv_1d_word2vec.h5')
```

4. Test on custom reviews:

```python
reviews = [
    "The movie was absolutely fantastic, loved every bit of it!",
    "Terrible plot and bad acting, waste of time."
]

vec_reviews = vectorization(reviews)
preds = model_gru.predict(vec_reviews)

for review, p in zip(reviews, preds):
    label = "Positive" if p[0] > 0.5 else "Negative"
    print(f"Review: {review}\nPrediction: {label} (confidence: {p[0]:.4f})\n")
```
## **Pretrained Embeddings**

* Word2Vec embeddings from **Google News (3M vocab, 300-dim)**.
* Only embeddings for **top 10,000 words** in our dataset are extracted.
* Saved as `pretrained_embeddings.npy`.
* Can be loaded in Keras embedding layer with `embeddings_initializer`.

## **Model Evaluation**

* All models are trained on **train\_dataset** and validated on **val\_dataset**.
* Training and validation **loss/accuracy curves** are plotted.
* Models are evaluated on **test\_dataset** using:

```python
model.evaluate(test_dataset)
```

* Predictions return **confidence scores** for binary classification.

## Results & Model Performance

After experimenting with multiple architectures, we observed the following trends and outcomes:

### Training and Validation Trends

* **Simple RNN**: Started weak (\~50% accuracy) but steadily improved, achieving around 91.6% training accuracy and 81.4% validation accuracy by the 10th epoch.
* **LSTM**: Demonstrated strong performance early on, reaching about 94.5% training accuracy and 84.4% validation accuracy.
* **GRU**: Similar to LSTM in behavior, stabilizing with roughly 92% training accuracy and 83–85% validation accuracy.
* **CNN + Word2Vec Hybrid**: Benefited from pretrained embeddings for faster convergence, peaking at 95.8% training accuracy and 86% validation accuracy.

### Final Evaluation on Test Data

| Model                   | Train Accuracy | Validation Accuracy | Test Accuracy |
| ----------------------- | -------------- | ------------------- | ------------- |
| Simple RNN              | 91.6%          | 81.4%               | —             |
| LSTM                    | 94.5%          | 84.4%               | —             |
| GRU                     | 92.0%          | \~83%               | —             |
| CNN + Word2Vec (Hybrid) | 95.8%          | 86.1%               | 86.3%         |


* The **CNN + Word2Vec hybrid model** achieved the strongest overall generalization, with about 86.3% accuracy on unseen test data.
* **LSTM and GRU** also performed well, with robust validation accuracy, but lagged slightly behind the hybrid approach.
* The **Simple RNN** provided a useful baseline, though it struggled with long-term dependencies compared to more advanced models.


## **TensorBoard Embeddings Visualization**

* Embedding weights can be visualized in **3D TensorBoard projector**.
* Steps:

  1. Log embeddings during training.
  2. Save `metadata.tsv` for vocabulary.
  3. Load TensorBoard, select `PROJECTOR` tab.
  4. Search any word to see semantic proximity.

```bash
%tensorboard --logdir logs/imdb/fit/
```

## **Things to Remember**

* RNNs struggle with long sequences; **LSTM/GRU** help mitigate vanishing gradients.
* Conv1D networks can be **faster** and capture local patterns effectively.
* Pretrained embeddings like **Word2Vec** help models learn semantic meaning faster.
* Gradient clipping and careful preprocessing improve model stability.
* This project demonstrates a **full NLP workflow**: preprocessing → training → evaluation → inference → visualization.


## **References**

* [TensorFlow RNN Documentation](https://www.tensorflow.org/api_docs/python/tf/keras/layers/SimpleRNN)
* [Gensim Word2Vec](https://radimrehurek.com/gensim/auto_examples/index.html)
* [CBOW and Skip-gram Explanation](https://arxiv.org/abs/1301.3781)
* [TensorBoard Projector](https://www.tensorflow.org/tensorboard/tensorboard_projector_plugin)

## **License**

This repository is for **educational purposes**. Feel free to explore, modify, and use the notebooks for your personal learning projects.
