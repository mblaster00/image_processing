# Dance Posture-to-Image Generation by Papa Omar DIOP

**Short Description:**  
This project implements a Dance Posture-to-Image generation system using Vanilla Neural Networks and GANs, leveraging video skeleton data for training and image synthesis.

---

## Table of Contents

1. [Overview](#overview)  
2. [Features](#features)  
3. [Installation](#installation)  
---

## Overview

This project processes video skeleton data to generate realistic images of dancers in specific postures. It employs both Vanilla Neural Networks and Generative Adversarial Networks (GANs) to model the relationship between skeletal data and corresponding images.

### Objective:
- To generate realistic images of dancers based on skeleton posture inputs.  

### How It Works:
1. Extract skeleton data from dance videos using the `VideoSkeleton` module.  
2. Train a Vanilla Neural Network to predict image features from skeleton data.  
3. Implement a GAN model to refine image generation.  

### Key Algorithms:
- **Vanilla Neural Network:** Processes reduced skeleton data for basic predictions.  
- **GAN (Generative Adversarial Network):** Enhances image quality and realism.  

### Dataset Used:
- Dance videos in `.mp4` format (e.g., Taichi dance videos).  

---

## Features

- Skeleton-to-image generation using GANs.  
- Vanilla Neural Network for initial skeleton data mapping.  
- Video skeleton extraction and transformation.  
- Visualization of generated images using OpenCV.  

---

## Installation

1. Clone the repository:  
   ```bash
   git clone https://github.com/mblaster00/image_processing.git
   cd image_processing
