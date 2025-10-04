# Modulation Classification using Deep Learning

**Research Project | Jan 2025 – May 2025**  
**Under the guidance of:** *Dr. Swades De*  
**Department of Electrical Engineering, IIT Delhi*

---

## 🧠 Overview
This project explores **automatic modulation classification (AMC)** using deep learning architectures trained on **raw in-phase (I) and quadrature (Q)** signal samples.  
A **hybrid CNN–Transformer model (PCTNet)** was developed to improve classification accuracy over traditional convolutional networks, especially under **low-SNR** and **multi-signal** interference conditions.

---

## ⚙️ Dataset Generation
A custom dataset of raw complex I/Q samples was generated in **Python** to simulate realistic wireless conditions.
The dataset is stored in `.npy` format containing complex I/Q samples and modulation labels.

| Parameter | Details |
|------------|----------|
| **Modulations** | BPSK, QPSK, 16-QAM |
| **SNR Range** | –5 dB to 20 dB |
| **Signals per sample** | 1 – 3 (multi-signal scenarios) |
| **Total Samples** | ~12000 |
| **Sample length** | 512 IQ points |
| **Channel** | AWGN (optional fading model) |

---

## 📊 Results

The trained PCTNet model achieved an overall accuracy of **95.6%**, with major improvement in QPSK and 16-QAM classification compared to baseline CNNs.

### 🔹 Performance Plots
| Metric | Plot |
|:--------|:------:|
| Confusion Matrix |https://github.com/ManjuMtech/AMC-DeepLearning-README.md/tree/main/results |
| Accuracy vs SNR | ![Accuracy vs SNR]() |

You can explore more output files in the .


