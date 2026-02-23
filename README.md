# IDGH
This is the official implementation of "Intra-class Distribution-guided Generative Hashing with Neighbor Refinement for Cross-modal Retrieval" (CVPR 2026)

# Training
### **Processing dataset**
Refer to [DSPH](https://github.com/QinLab-WFU/DSPH).

### **Download the CLIP pretrained model**
The pretrained model will be found in the 30 lines of [CLIP/clip/clip.py](https://github.com/openai/CLIP/blob/main/clip/clip.py). This code is based on the "ViT-B/32".
You should copy ViT-B-32.pt to the directory of IDGH.

### **Start**
```bash
python main.py
