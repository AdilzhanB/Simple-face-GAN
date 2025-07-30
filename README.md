# 🎭 Simple Face GAN
### *A straightforward GAN implementation for celebrity face generation*

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-3776ab?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?style=for-the-badge&logo=jupyter&logoColor=white)

![Dataset](https://img.shields.io/badge/📸_Dataset-CelebA-gold?style=flat-square)
![Implementation](https://img.shields.io/badge/🔧_Type-Educational-brightgreen?style=flat-square)
![Difficulty](https://img.shields.io/badge/📚_Level-Beginner_Friendly-blue?style=flat-square)

</div>

---

## 📝 **What This Is**

A **simple, educational implementation** of a Generative Adversarial Network (GAN) for generating celebrity-like faces using the CelebA dataset. Perfect for learning how GANs work!

✨ **Single Jupyter notebook** with clear, commented code  
🎓 **Educational focus** - easy to understand and modify  
🖼️ **CelebA dataset** - generates faces similar to celebrities  
⚡ **Lightweight** - basic GAN architecture without complex features  

---

## 🚀 **Quick Start**

1. **Clone the repository**
   ```bash
   git clone https://github.com/AdilzhanB/Simple-face-GAN.git
   cd Simple-face-GAN
   ```

2. **Install requirements**
   ```bash
   pip install torch torchvision matplotlib numpy jupyter
   ```

3. **Open the notebook**
   ```bash
   jupyter notebook Simple_Face_GAN.ipynb
   ```

4. **Run all cells** and watch the magic happen! 🎉

---

## 📚 **What You'll Learn**

- 🧠 **GAN Basics**: How Generator and Discriminator networks work together
- 🏗️ **Network Architecture**: Simple CNN-based generator and discriminator
- 📊 **Training Process**: Loss functions, backpropagation, and training loops
- 🎨 **Face Generation**: How to generate new celebrity-like faces from noise
- 📈 **Monitoring Training**: Visualizing losses and generated samples

---

## 🏗️ **Architecture Overview**

### **Generator** 🎨
```
Noise (100D) → FC Layer → Reshape → ConvTranspose2D layers → Face (64×64)
```

### **Discriminator** 🕵️
```
Face (64×64) → Conv2D layers → FC Layer → Real/Fake (1D)
```

**That's it!** Simple and straightforward architecture perfect for learning.

---

## 📁 **Project Structure**

```
Simple-face-GAN/
├── 📓 Simple_Face_GAN.ipynb    # Main notebook with everything
├── 📋 README.md                # This file
├── 📋 requirements.txt         # Basic dependencies
└── 🖼️ generated_samples/       # Output folder (created when you run)
```

---

## 🎯 **What to Expect**

- **Training Time**: ~30-60 minutes (depends on your hardware)
- **Output Quality**: Basic but recognizable faces
- **Learning Curve**: Beginner-friendly with lots of comments
- **Customization**: Easy to modify and experiment with

### **Sample Results**
The generated faces won't be perfect, but you'll see:
- ✅ Face-like structures
- ✅ Eyes, nose, mouth in right places  
- ✅ Some celebrity-like features
- ✅ Gradual improvement during training

---

## 🛠️ **Requirements**

```txt
torch>=1.9.0
torchvision>=0.10.0
matplotlib>=3.3.0
numpy>=1.21.0
jupyter>=1.0.0
Pillow>=8.0.0
```

**System Requirements:**
- Python 3.8+
- 4GB+ RAM
- GPU recommended but not required
- ~2GB storage for dataset

---

## 🎓 **Perfect For**

- 📚 **Students** learning about GANs
- 🔬 **Researchers** needing a simple baseline
- 👨‍💻 **Developers** exploring generative models
- 🎨 **Hobbyists** interested in AI art generation

---

## 📖 **How to Use**

1. **Open the notebook** in Jupyter
2. **Read the explanations** in each cell
3. **Run cells one by one** to understand each step
4. **Experiment** with hyperparameters
5. **Generate your own faces** with the trained model

### **Key Sections in the Notebook:**
- 🔧 **Setup & Imports**
- 📊 **Data Loading** (CelebA dataset)
- 🏗️ **Model Definition** (Generator & Discriminator)
- 🎯 **Training Loop** (with visualizations)
- 🎨 **Face Generation** (create new faces)

---

## 🔧 **Customization Ideas**

Once you understand the basics, try:

- 🎛️ **Change hyperparameters** (learning rate, batch size)
- 🏗️ **Modify architecture** (add/remove layers)
- 📊 **Different loss functions** (experiment with WGAN, LSGAN)
- 🎨 **Higher resolution** (try 128×128 instead of 64×64)
- 📈 **Add monitoring** (FID score, Inception Score)

---

## ❓ **Common Questions**

**Q: Do I need a GPU?**  
A: No, but it's much faster. CPU works fine for learning.

**Q: How long does training take?**  
A: 30-60 minutes depending on your hardware and epochs.

**Q: Can I use my own dataset?**  
A: Yes! Just modify the data loading section in the notebook.

**Q: Why are the faces blurry?**  
A: This is a simple implementation. For better quality, you'd need more complex architectures.

---

## 🤝 **Contributing**

This is a simple educational project, but improvements are welcome!

- 🐛 **Bug fixes**
- 📝 **Better documentation**
- 🎓 **More educational content**
- 🔧 **Code optimizations**

---

## 🙏 **Acknowledgments**

- **CelebA Dataset**: For providing the celebrity face images
- **PyTorch Team**: For the amazing framework
---

## 📞 **Contact**

- **GitHub**: [@AdilzhanB](https://github.com/AdilzhanB)
- **Questions**: Open an issue in this repository

---

<div align="center">

**Made for learning and education** 📚  
*Simple, honest, and effective*

⭐ **Star this repo** if it helped you learn GANs!

</div>
