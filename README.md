# 🧠 SVM Breast Cancer Classification

Hey there! 👋  
Welcome to my project for **Elevale’s AI & ML Internship Task 7!**
I’m so thrilled to share this with you—it’s all about diving into **Support Vector Machines (SVM)** to classify breast cancer data.  
I’ve put together a straightforward, functional setup to train models, create awesome visualizations, and evaluate performance, all while learning the ins and outs of SVMs.  
Let’s dive in! 🚀

---

## 🌟 What’s This Project About?

This project is all about using SVMs to predict whether breast cancer tumors are **malignant** or **benign**.  
I used the **Breast Cancer Dataset** and tackled the following steps:

✅ Loaded and prepared the dataset for binary classification  
✅ Trained SVM models with both **linear** and **RBF** kernels  
✅ Visualized the decision boundaries in 2D to see how the models separate the data (it’s super cool to see!)  
✅ Tuned hyperparameters like `C` and `gamma` to get the best performance  
✅ Evaluated the models with **5-fold cross-validation** to ensure they’re reliable

It’s a practical showcase of SVMs in action, built with **Python** and some amazing libraries! 🐍📊

---

## 📂 How It’s Organized

Here’s what you’ll find in the repo:

📁 svm_classifier.py # The main script that does all the heavy lifting—my pride and joy!
📁 dataset/ # A placeholder for the dataset (empty, since Scikit-learn's built-in dataset is used)
📁 screenshots/ # Contains one awesome plot: decision_boundaries_comparison.png
📄 requirements.txt # Lists the libraries you’ll need to run the code
📄 README.md # This file, your friendly guide through the project!


---

## 🛠️ What I Did

Here’s the step-by-step breakdown of how I built this, straight from the Task 7 hints:

🔹 **Loaded and Prepared the Data**:  
I grabbed the Breast Cancer Dataset and selected two features (**mean radius** and **mean texture**) to work with 2D data for visualization.

🔹 **Trained SVM Models**:  
I set up two SVM models—one with a **linear kernel** and the other with an **RBF kernel**.

🔹 **Visualized Decision Boundaries**:  
I plotted the decision boundaries in 2D for both models in a single side-by-side figure (check out the screenshot—it’s so clear!).

🔹 **Tuned Hyperparameters**:  
I used `GridSearchCV` to find the best `C` and `gamma` values for optimal performance.

🔹 **Evaluated with Cross-Validation**:  
I ran **5-fold cross-validation** to make sure the models are solid and trustworthy.

---

## ⚙️ Tools You’ll Need

To run this project, you’ll need:

- 🐍 **Python 3.x** (I used 3.12, but 3.8+ works fine)  
- 🧪 `scikit-learn`: For the ML magic  
- 🔢 `numpy`: For handling numbers  
- 📈 `matplotlib`: For creating the visualizations  
- 🎨 `seaborn`: To make the visualizations look extra nice (optional but used for styling)

---

## 📥 How to Set Up and Run on Your System

Want to run this project on your own computer? Whether you’re on **Windows** or **macOS**, I’ve got you covered with a simple step-by-step guide.  
No prior setup? **No worries at all!**

---

### 🪟 Step 1: Install Python

#### Windows:
1. Download Python 3.12 from [python.org](https://www.python.org/)
2. Run the installer. ✅ Check “Add Python to PATH” before clicking **Install Now**.
3. Open Command Prompt and verify installation:

```bash
python --version

You should see something like: Python 3.12.x

-> macOS:

1. Download Python 3.12 from python.org

2. Run the installer.

3. Open Terminal and verify installation:

python3 --version

You should see: Python 3.12.x
🔧 Step 2: Install Git
Windows:

    Download Git from git-scm.com

    Run the installer with default settings

    Verify installation:

git --version

macOS:

Check if Git is already installed:

git --version

If not, install it via Homebrew:

/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install git

📦 Step 3: Clone the Repository

    Open Command Prompt (Windows) or Terminal (macOS)

    Navigate to a folder where you want the project:

cd Documents

    Clone the repo:

git clone https://github.com/yourusername/svm-breast-cancer.git
cd svm-breast-cancer

📚 Step 4: Install Dependencies

Install the required packages listed in requirements.txt:
Windows:

pip install -r requirements.txt

macOS:

pip3 install -r requirements.txt

This installs scikit-learn, numpy, matplotlib, and seaborn.
▶️ Step 5: Run the Project

Run the main script:
Windows:

python svm_classifier.py

macOS:

python3 svm_classifier.py

The script will:

    Load and prepare the dataset

    Train the SVM models with linear and RBF kernels

    Generate a side-by-side decision boundary plot in the screenshots/ folder

    Tune hyperparameters and evaluate the models

    Print the results to the console ✅

🎯 Check the screenshots/ folder for decision_boundaries_comparison.png.
📈 What You’ll See

When you run the script, you’ll get output like:

    Best Parameters:
    Something like {'C': 10, 'gamma': 0.1, 'kernel': 'rbf'} (depends on the run)

    Best Cross-Validation Score:
    How well the model performed during tuning

    Test Set Score:
    The accuracy on the test data

    Cross-Validation Scores for both models:
    A solid measure of model reliability, with mean scores for each

The decision_boundaries_comparison.png in the screenshots folder will show how each model separates the data—super insightful!
🎉 Wrapping Up

This project was such a fun learning experience!
I got to explore SVMs in depth—learning about margin maximization, the kernel trick, and how to tune hyperparameters for real-world data.
I’m really proud of how the code turned out, especially the clean visuals. 😄

Feel free to explore, tweak, or reach out if you have questions—I’d love to chat about it.
Happy coding! 💻✨