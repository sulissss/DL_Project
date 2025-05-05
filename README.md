# Frames Matter: Improving Video-Text Alignment via Fusion and Sampling 
**Group 35 — Sulaiman Ahmad (26100350), Muhammad Bazaf Shakeel (26100146)**

---

## Overview

This project focuses on adapting the pretrained **ViCLIP model** for a small-scale **video-to-text retrieval task** involving aesthetically captioned short video clips. Our objective is to:
- Fine-tune ViCLIP for high alignment between visual content and rich descriptive captions.
- Explore architectural and training improvements to handle a **low-data regime** (starting with ~50 video-caption pairs).

We developed and evaluated the model across three distinct phases:
1. **Baseline ViCLIP (Zero-Shot Cosine Similarity)**
2. **First Set of Improvements**: Cross-modal attention, contrastive loss optimization and a cosine gap study
3. **Second Set of Improvements**: Multimodal fusion and custom frame sampling strategies

---

## **[Dataset Construction](scripts/G35_Dataset.ipynb)**

We used the **InternVid-18M-AES** dataset available on HuggingFace. This version was selected due to:
- **Shorter clip lengths (3–15 seconds)** for faster iteration and lower memory usage.
- **High-quality aesthetic captions**, offering rich textual descriptions ideal for fine-grained vision-language learning.

### Download & Processing

The dataset was filtered, clipped, and downloaded using:
- `yt-dlp` for YouTube video retrieval
- `ffmpeg` for timestamp-based cropping
- A script that:
  - Skips corrupt/inaccessible videos
  - Validates duration bounds
  - Generates clean filenames for storage

Final dataset: **50 curated clips**, expandable to 1000+  
Splits: **80% training**, **10% alignment**, **10% test**

Videos are stored in Google Drive and loaded using a custom PyTorch `VideoCaptionDataset` class.

---

## **[Baseline Model](scripts/G35_BaselineModel.ipynb)**

The baseline uses **ViCLIP** with pretrained weights from the `ViClip-InternVid-10M-FLT` checkpoint provided by OpenGVLab. ViCLIP is a dual-encoder architecture with independent encoders for:
- Vision: processes video frames
- Text: processes natural language captions

### Evaluation Setup

We performed a **zero-shot forward pass**:
- Extracted 8 frames from a clip
- Computed cosine similarity between the clip and a pool of caption candidates
- Ranked matches based on similarity scores

### Results

| Evaluation Method     | Score                |
|-----------------------|----------------------|
| Cosine Similarity Top-1 (Zero-shot) | 0.6278 |
| Equivalent InfoNCE Alignment Loss  | 0.1719 |

### Why We Switched to Loss Functions

Although cosine similarity gives insight into retrieval quality, it does not enable **gradient-based optimization**. To improve alignment in a learnable way, we switched to **contrastive losses** (InfoNCE and H-NAC), which:
- Penalize mismatched video-caption pairs
- Offer a consistent metric for training and alignment
- Allow for better convergence monitoring

---

## **[First Set of Improvements: Cross-Attention & Contrastive Losses](scripts/G35_First_Improvement.ipynb)**


### Architectural Change: Cross-Attention

We introduced a **Cross-Attention module** between the latent features of the vision and text encoders, allowing:
- Vision features to attend to text features
- Text features to attend to vision features


### Introduction of a New Loss Function: H-NAC

In addition to the standard **InfoNCE** contrastive loss, we introduced a novel objective— the **H-NAC (Hard Negative-Aware Contrastive) Loss**. This loss explicitly addresses the issue of false negatives during contrastive training:
- **InfoNCE**: standard softmax-based loss that treats all non-matching pairs as equally negative.
- **H-NAC**: introduces a similarity-based decay function to down-weight hard false negatives, encouraging more robust learning.


### Cosine Gap Analysis

For our **[Cosine Gap Analysis](scripts/Cosine_Gap_Analysis.ipynb)**, we computed the cosine similarity gap to assess how well each loss separates positive from hard negative pairs:
- The cosine gap is the average similarity difference between true pairs and the hardest sampled negatives.
- A higher gap indicates better contrastive separation and robustness to false negatives.

### Results

On the baseline model, we performed a zero-shot forward pass using pre-trained ViCLIP weights and obtained an alignment loss of **0.1719**.

After fine-tuning the baseline and the modified cross-attended model on our AES-InternVid dataset (using both InfoNCE and H-NAC), we achieved:

#### Alignment Loss Results

| Loss Type  | Baseline | Cross-Attn |
|------------|----------|------------|
| InfoNCE    | 0.1134   | **0.0835** |
| H-NAC      | 0.0522   | **0.0412** |

#### Cosine Gap Analysis

| Loss Type  | Cosine Gap |
|------------|-------------|
| InfoNCE    | 0.1854      |
| H-NAC      | **0.2036**  |

### Takeaways

- The **Cross-Attention module** enhanced model performance across both the InfoNCE and H-NAC objectives.
- The **cosine gap analysis** confirmed that H-NAC provides better separation in embedding space, increasing robustness against semantic false negatives.


---

## **[Second Set of Improvements: Fusion Transformer + Frame Sampling](scripts/G35_Second_Improvement.ipynb)**

After attention, we explored two further ideas to enhance fine-grained alignment.

### 1. Multimodal Fusion Transformer

A small 4-layer transformer encoder was introduced **after** encoding both modalities.  
Input: concatenated `[video_embeds | text_embeds]`  
Output: temporally aware, modality-integrated representations

This allowed for **deep integration** of video and language features.

### 2. Frame Sampling Strategies

With only 8 frames per video, *which* frames are selected becomes critical. We implemented and tested:

| Strategy   | Description                                                       |
|------------|-------------------------------------------------------------------|
| Default    | Uniformly samples 8 frames across the clip                        |
| Mean       | Computes pixel-wise average frame, repeated 8 times               |
| Keyframes  | Picks start, middle, and end frames, with repetition              |
| Weighted   | Samples based on a linear probability ramp (later frames favored) |
| Dropout    | Randomly drops 30% of frames before sampling from the rest        |

### Results (Alignment Loss @ Epoch 10)

| Loss Type | Default | Mean         | Keyframes     | Weighted      | Dropout       |
|-----------|---------|--------------|---------------|---------------|---------------|
| InfoNCE   | 0.0012  | 0.0011       | **0.0007**        | 0.0016        | 0.0008        |
| H-NAC      | 0.0005  | **0.0004**   | 0.0011        | 0.0011        | 0.0013        |


### Takeaways

- For the InfoNCE objective, **keyframes** gave the best result
- For the H-NAC objective, **mean** frames gave the best result
- Dropout and weighted sampling performed decently under the InfoNCE objective, showing the potential for temporal diversity

---

## Architecture Summary

| Component             | Description                                      |
|-----------------------|--------------------------------------------------|
| Vision Encoder        | From ViCLIP pretrained on InternVid-10M          |
| Text Encoder          | From ViCLIP pretrained on InternVid-10M          |
| CrossAttention (opt)  | Injects query-key-value cross-modality attention |
| Fusion Transformer    | 4-layer encoder to fuse vision & text embeddings |
| Loss Functions        | InfoNCE / H-NAC                                   |
| Frame Strategies      | 5 variants tested for temporal robustness        |

---

## Training & Evaluation

- Optimizer: **AdamW**
- Learning Rate: **2e-5**
- Weight Decay: **0.02**
- Batch Size: **4**
- Epochs: **10–20 per config**
- Device: CUDA (Google Colab)

Metrics:
- **Alignment loss (contrastive)** was the primary metric
- Tracked convergence across all combinations of fusion and sampling

---

## How to Run This Project

1. Mount Google Drive in Colab (or run locally with sufficient storage)
2. Clone this repository and place all notebooks in your drive
3. Run the following notebooks in order:
   - `G35_Dataset.ipynb` — downloads, filters, and clips videos
   - `G35_BaselineModel.ipynb` — evaluates pretrained ViCLIP using cosine similarity
   - `G35_First_Improvement.ipynb` — introduces attention and loss-based training
   - `G35_Second_Improvement.ipynb` — adds fusion layers and frame sampling
   - `Cosine_Gap_Analysis.ipynb` - evaluates the effectiveness of the loss functions used in our experiments

Models are saved to Drive, and loss plots are generated after each run.

---

## Final Summary

### Phase 1: Baseline and First Improvements


#### Alignment Loss Results

| Loss Type  | Baseline | Cross-Attn |
|------------|----------|------------|
| InfoNCE    | 0.1134   | **0.0835** |
| H-NAC      | 0.0522   | **0.0412** |

#### Cosine Gap Analysis

| Loss Type  | Cosine Gap |
|------------|-------------|
| InfoNCE    | 0.1854      |
| H-NAC      | **0.2036**  |


### Phase 2: Fusion Transformer + Frame Sampling (Epoch 10)

| Loss Type | Default | Mean         | Keyframes     | Weighted      | Dropout       |
|-----------|---------|--------------|---------------|---------------|---------------|
| InfoNCE   | 0.0012  | 0.0011       | **0.0007**        | 0.0016        | 0.0008        |
| H-NAC      | 0.0005  | **0.0004**   | 0.0011        | 0.0011        | 0.0013        |

### Observations

- In Phase 1, the **Cross-Attention module** improved performance across both loss objectives, demonstrating the benefit of enabling inter-modal interaction at the feature level.
- The **cosine gap analysis** showed that **H-NAC** provides better separation between true pairs and hard negatives, highlighting its robustness.
- In Phase 2, the addition of a **Fusion Transformer** further enhanced performance by enabling deeper modality integration.
- Frame sampling strategies influenced results depending on the loss used:
  - **Keyframes** worked best under InfoNCE.
  - **Mean frames** worked best under H-NAC, achieving the lowest overall alignment loss (**0.0004**).

**Key Insight**:  
Each phase brought consistent improvements, proving that **even small enhancements—like smarter frame selection or lightweight attention—can significantly boost model performance**, especially when data is limited.

---

## Acknowledgements

- [ViCLIP](https://github.com/OpenGVLab/InternVideo) from OpenGVLab
- HuggingFace Datasets and Hub
- [InternVid-10M & AES](https://huggingface.co/datasets/OpenGVLab/InternVid)
- Inspired by: [ViCLIP Paper (arXiv)](https://arxiv.org/abs/2307.06942)

---

## License

This project is intended for academic and research use.  
Please refer to OpenGVLab and HuggingFace for licensing details regarding pretrained weights and datasets.