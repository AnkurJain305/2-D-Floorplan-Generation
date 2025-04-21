
# 🏠 Residential Floor Plan Generation Using Deep Learning Techniques

This project introduces an AI-driven approach to automate residential floor plan generation using deep learning techniques, specifically **Graph Neural Networks (GNNs)** and **Graph Attention Networks (GATs)**. The system allows users to define the structure of their residential space, and generates an optimized 2D layout with accurate room sizing and positioning.

---

## 📘 Project Description

Our goal is to bridge the gap between architectural complexity and user-friendly design tools. Users can specify room types, quantity, or overall carpet area, and our trained GATNet model predicts the best-fit room dimensions based on a graph-based representation of floor plans.

---

## 📁 Dataset

- Source: [RPlan Dataset](http://staff.ustc.edu.cn/~fuxm/projects/DeepLayout/index.html)  
- Size: ~80,000 floor plans
- **Preprocessing Steps:**
  - Converted floor plan images into geometric polygon-based representations using `Shapely`.
  - Further converted polygons into **graph structures**, where:
    - Nodes represent rooms (with type, size, width, height).
    - Edges represent spatial relationships between rooms.
  - Created two types of graphs:
    - Room Graph
    - Boundary Graph

This enables us to leverage GNN models for predicting architectural layouts.

---

## ⚙️ Architecture Overview

The system consists of **two main stages**:

1. **Room Centroid Prediction** (CNN-based) – Optional stage for centroids (not used in latest pipeline).
2. **Room Size Estimation** (GNN-based) – Main pipeline using our custom-built **GATNet**.

### System Flow:

```
User Input ⟶ Graph Construction ⟶ GATNet Prediction ⟶ 2D Layout Rendering
```

### User Inputs:

- Desired room types and counts (e.g., 2 bedrooms, 1 kitchen)
- Optional: Total available carpet area (m²)

### User Output:

- Visual 2D floor plan
- Labeled room types
- Room dimensions (width x height)
- Doors and outer boundary box
- Downloadable PNG image

---

## 🔍 Graph Conversion and Preprocessing

To prepare data for GNN input:

- Converted room polygons → graph nodes with features:
  - Room type
  - Centroid position
  - Area ratio
  - Width, height
- Created various edge connection types:
  - Real Connections
  - All-to-All
  - Living-to-All (used in final model)

**Graph attention** enables the model to learn relevant relationships between rooms even when direct connections are missing.

---

## 🧠 GATNet Architecture

We implemented a **Residual GAT-based model** for predicting room dimensions.

### Key Features:
- Uses multiple GATConv layers for better spatial attention.
- Concatenation-based residuals to prevent oversmoothing.
- Predicts both `width` and `height` for every room.

### Training Details:
- Optimizer: Adam
- Loss: Mean Squared Error (MSE)
- Epochs: 300
- Learning Rate: 0.001
- Weight Decay: 3e-5
- LR Scheduler (gamma = 0.95)

---

## 📈 Evaluation Results

| Metric                  | GCN      | GATNet ✅  |
|-------------------------|----------|------------|
| Room Type Accuracy      | 68%      | **94%**    |
| MAE (room size in meters) | 1.8m     | **0.96m**  |
| IoU (layout overlap)    | 71%      | **84.5%**  |

- Final Training Loss: **0.5828**
- Final Validation Loss: **0.7808**
- Epoch: 38

---

## 🧪 How to Run the Project

### 🧱 Dependencies

```bash
Python >= 3.9
torch
torch-geometric
matplotlib
streamlit
openai (for optional GPT usage, if required)
```

### 📦 Setup

```bash
git clone https://github.com/your-username/floor-plan-generator-gatnet.git
cd floor-plan-generator-gatnet
pip install -r requirements.txt
```

### ▶️ Running the App

```bash
streamlit run app.py
```

---

## 🖼️ Output Preview

> The app generates labeled layouts with boundaries and doors based on GATNet predictions.

```
📍 Bedroom       📏 4m x 4m
📍 Kitchen       📏 3m x 3m
📍 Living Room   📏 5m x 4.5m
```

Each room has:
- Accurate dimensions
- Distinct color-coding
- Door indicator
- Overall site boundary

---

## 📌 Notes

- You can disable GPT-based prompt parsing.
- Fully functional with direct room input or total carpet area constraints.
- Clean interface built using Streamlit.

---

## 📤 Contribution

Feel free to fork this repository, submit issues, or suggest improvements.

---

## 📝 License

MIT License – open for academic and non-commercial use.

---

## 📸 Screenshots

![image](https://github.com/user-attachments/assets/3cd7b1c7-a357-4103-8141-92cd79308aab)

