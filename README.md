# Frames Matter: Improving Video-Text Alignment via Fusion and Sampling 
**Group 35 — Sulaiman Ahmad (26100350), Muhammad Bazaf Shakeel (26100146)**

---

## Overview

This project focuses on adapting the pretrained **ViCLIP model** for a small-scale **video-to-text retrieval task** involving aesthetically captioned short video clips. Our objective is to:
- Fine-tune ViCLIP for high alignment between visual content and rich descriptive captions.
- Explore architectural and training improvements to handle a **low-data regime** (starting with ~50 video-caption pairs).

We developed and evaluated the model across three distinct phases:
1. **Baseline ViCLIP (Zero-Shot Cosine Similarity)**
2. **First Set of Improvements**: Cross-modal attention and contrastive loss optimization
3. **Second Set of Improvements**: Multimodal fusion and custom frame sampling strategies

---

## Dataset Construction

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
Splits: **80% training**, **10% validation**, **10% test**

Videos are stored in Google Drive and loaded using a custom PyTorch `VideoCaptionDataset` class.

---

## Baseline Model

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
| Equivalent InfoNCE Validation Loss  | 0.1719 |

### Why We Switched to Loss Functions

Although cosine similarity gives insight into retrieval quality, it does not enable **gradient-based optimization**. To improve alignment in a learnable way, we switched to **contrastive losses** (InfoNCE and HNAC), which:
- Penalize mismatched video-caption pairs
- Offer a consistent metric for training and validation
- Allow for better convergence monitoring

---

## First Set of Improvements: Cross Attention & Contrastive Losses

### Architectural Change: Cross-Attention

We introduced **CrossAttention modules**, allowing:
- Vision features to attend to text features
- Text features to attend to vision features

This encourages **cross-modal grounding**, enabling richer interaction between the modalities beyond independent encoding.

### Loss Functions
We compared two contrastive objectives:
- **InfoNCE**: standard softmax-based contrastive loss
- **HNAC (Hard Negative-Aware Contrastive)**: down-weights false negatives using a similarity-based decay function

### Results

| Configuration                 | Final Validation Loss  |
|-------------------------------|------------------------|
| Baseline (Zero-Shot Cosine)   | 0.1719                 |
| Default + InfoNCE             | 0.0461                 |
| Default + HNAC                | 0.0522                 |
| Cross-Attn + InfoNCE          | 0.0835                 |
| Cross-Attn + HNAC             | **0.0412**             |

### Takeaways
- **Fine-tuning alone** greatly improved loss (over 50% drop from baseline)
- **Cross-attention** offered additional benefit
- **HNAC outperformed InfoNCE**, especially when paired with attention

---

## Second Set of Improvements: Fusion Transformer + Frame Sampling

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

### Results (Validation Loss @ Epoch 10)

| Strategy   | InfoNCE        | HNAC           |
|------------|----------------|----------------|
| Default    | 0.0012         | 0.0005         |
| Mean       | 0.0011         | **0.0004**     |
| Keyframes  | 0.0007         | 0.0011         |
| Weighted   | 0.0016         | 0.0011         |
| Dropout    | 0.0008         | 0.0013         |

### Takeaways

- **Mean frames with HNAC** gave the best result
- Surprising finding: **Averaged frames may provide consistent global context**, helping model generalize better
- Even dropout and weighted sampling performed decently, showing the potential for temporal diversity

---

## Architecture Summary

| Component             | Description                                      |
|-----------------------|--------------------------------------------------|
| Vision Encoder        | From ViCLIP pretrained on InternVid-10M          |
| Text Encoder          | From ViCLIP pretrained on InternVid-10M          |
| CrossAttention (opt)  | Injects query-key-value cross-modality attention |
| Fusion Transformer    | 4-layer encoder to fuse vision & text embeddings |
| Loss Functions        | InfoNCE / HNAC                                   |
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
- **Validation loss (contrastive)** was the primary metric
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

Models are saved to Drive, and loss plots are generated after each run.

---

## Final Summary

### Phase 1: Baseline and First Improvements

| Configuration              | Best Validation Loss |
|----------------------------|----------------------|
| Baseline (Zero-Shot Cosine)| 0.6278 (cosine sim)  |
| Baseline (Loss Equivalent) | 0.1719              |
| Default + InfoNCE          | 0.0461               |
| Default + HNAC             | 0.0522               |
| Cross-Attn + InfoNCE       | 0.0835               |
| Cross-Attn + HNAC          | **0.0412**           |

### Phase 2: Fusion Transformer + Frame Sampling (Epoch 10)

| Strategy   | InfoNCE Loss   | HNAC Loss      |
|------------|----------------|----------------|
| Default    | 0.0012         | 0.0005         |
| Mean       | 0.0011         | **0.0004**     |
| Keyframes  | 0.0007         | 0.0011         |
| Weighted   | 0.0016         | 0.0011         |
| Dropout    | 0.0008         | 0.0013         |

### Observations

- Transitioning from cosine similarity to trainable contrastive losses (InfoNCE and HNAC) led to **a 2.5–3x improvement** in validation loss.
- The **Cross-Attention + HNAC** setup outperformed all non-fusion models.
- Introducing the **Fusion Transformer** and testing **frame strategies** further reduced validation loss by quite alot.
- The **Mean frame strategy under HNAC** achieved the best overall result: **0.0004**, showing that even heavily averaged frames can be effective when deeply fused.


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