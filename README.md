### Cross-modal-Audio-Generation-for-Visually-Impaired-Users-in-Social-Chat

#  Emoji-to-Audio: Enhancing Social Accessibility

This project converts emojis (or emoticons) into **expressive audio descriptions**, enabling visually impaired individuals to perceive emotional and social cues in digital communication.  

It breaks visual barriers, simplifies interactions, and fosters inclusion by bridging image and sound.  

---

##  Key Innovation
- **Training-free cross-modal retrieval**: Efficient emoji-to-audio conversion without extensive pre-training, lowering computational and deployment costs.  
- **Accessible communication**: Provides expressive auditory feedback for visually impaired users, enhancing participation in social platforms.  
- **Multimodal AI advancement**: Builds a pipeline integrating **FAISS, CLIP, Qwen, AudioLDM, and AudioMinimax** for robust image–audio matching.  

---

##  Methodology

### 1. Image Matching
- Input an emoji image.  
- Extract CLIP features.  
- Use FAISS to retrieve top-5 similar images.  

### 2. Description Generation
- Apply a Vision-Language Model (Qwen) to generate natural text descriptions of the input image.  

### 3. Fusion & Audio Search
- Combine textual encodings and search for audio matches in a curated library.  
- Audio resources are enriched with **AudioLDM-generated background music** and **AudioMinimax speech synthesis**.  

### 4. Dynamic Output
- Deliver engaging audio output tailored to the emotional and social context of the emoji.  
- Ensures emotional cues are preserved and accessible.  

---

##  System Architecture
- **Feature Extraction**: CLIP, Qwen Embeddings  
- **Vector Database**: FAISS (IVF256 + Flat) for efficient similarity search  
- **Audio Generation**:  
  - **AudioLDM2** → background music (genre, instrumentation, emotional tone)  
  - **AudioMinimax** → speech synthesis  
  - Fusion into final audio track  
- **AI Agent**: Built on **n8n + RAGFlow**, orchestrating retrieval, aggregation, and matching  

---

##  Results
- Created an **interactive image–audio dataset** enriched with human- and AI-generated samples.  
- Developed a working **AI agent** capable of pairing emojis with expressive audio in real time.  
- Improved inclusivity and accessibility for visually impaired users in online social environments.  

---

##  Future Work
- Expand dataset with **more diverse emotional cues**.  
- Refine multimodal algorithms to enhance the **richness and precision of audio outputs**.  
- Explore **real-time emoji-to-audio generation** for integration in mainstream communication platforms.  

---

##  Team
**Authors**: Hongyi Ding, Haosen Shi, Wansu Mo, Zeyu Yin, Jianghui Sun, Yiming Hu  
**Supervisors**: Xi Yang, Yuyao Yan  

---


## 📌 Poster
![Final Poster](./poster.png)
