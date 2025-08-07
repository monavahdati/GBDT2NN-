# LightGBM + Neural Network (GBDT2NN) for BNPL Credit Classification 💳🧠

This repository demonstrates a hybrid machine learning architecture where LightGBM is used to learn tree-based patterns, and a deep neural network further learns interactions using the leaf indices and raw features.

---

## 📌 Highlights

- LightGBM learns structure in the data and outputs leaf indices per tree.
- These leaf indices are embedded and used as inputs to a DNN (like categorical embeddings).
- The model predicts credit eligibility for BNPL offers.
- Visual and explainable analysis with SHAP + KS + ROC + PR curves.

---

## 🔧 Tech Stack

- PyTorch (DNN & embeddings)
- LightGBM (GBDT leaf extraction)
- SHAP (explainability)
- Scikit-learn (evaluation)
- Matplotlib / Seaborn (visualizations)

---

## 🧠 Model Architecture

```python
LightGBM_Embedding_DNN(
    leaf_embeddings + original_features → Linear → ReLU →
    BatchNorm → Dropout → Linear → ReLU → Output
)
