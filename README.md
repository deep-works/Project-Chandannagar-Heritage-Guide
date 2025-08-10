# Heritage Hunt Chandannagar

**Heritage Hunt Chandannagar** is an AI-driven image classification project aimed at identifying and categorizing heritage sites in Chandannagar, India. Leveraging deep learning and computer vision, the project utilizes Convolutional Neural Networks (CNNs) to aid in the discovery, navigation, and preservation of local cultural landmarks.

---

#### Project Overview

The primary objective of this project is to develop an intelligent guide application that recognizes and classifies heritage sites from images. The approach is rooted in deep learning, enabling automated identification to support tourism, historical research, and preservation efforts.


#### Methodology

- **Dataset:** A custom collection of 73 images, split into 58 for training and 15 for validation, representing 6 heritage site classes: Church, Clock Tower, Joraghat, Nandadulal Mondir, Museum, and Patalbari.
- **Model Architecture:** ResNet18 CNN, trained from scratch without transfer learning.
- **Data Augmentation:** Applied to the training set to improve generalization.
- **Technologies:** Python (Google Colab & Google Drive), PyTorch, Pandas, NumPy, Matplotlib, PIL, and Scikit-learn.


#### Results & Challenges

- **Results:** Achieved a peak validation accuracy of 80%, with a final accuracy of 66.67%. The model demonstrated effective classification despite dataset limitations.
- **Challenges:** Significant overfitting due to the small dataset. The "Church" class experienced the most misclassifications.
- **Lessons Learned:** Highlighted the importance of dataset size/diversity, hyperparameter tuning, and the value of transfer learning for small datasets.


#### Future Scope

- **Dataset Expansion:** Increase the volume and diversity of training images.
- **Transfer Learning:** Implement with pre-trained weights for improved performance.
- **Advanced Architectures:** Explore deeper and more complex CNN models.
- **Application Development:** Create a user-friendly guide app for real-world use.
- **Explainable AI (XAI):** Integrate XAI techniques for model transparency and interpretability.

---

## Repository Structure

**Step 1: Data Preparation**  
- `/train_val_split_folder_making_maximize_dataset.ipynb` — Notebook for splitting the `Data` folder into `training_images` and `validate_images` folders, organized by class.

**Step 2: Model Training & Evaluation**  
- `/EXPERIMENTS_all_in_one_parameter_changable.ipynb` — Notebook for training and evaluating the ResNet18 classifier with different parameters and configurations.

**Step 3: Reporting & Visualization**  
- `/final_project.ipynb` — Contains result analysis, visualizations, and project conclusions.

---

## Data Organization & Folder Structure

To use this project with your own data, organize your Google Drive as follows:

```
Project-Chandannagar-Heritage-Guide/
└── Data/
    ├── church/
    ├── clock_tower/
    ├── joraghat/
    ├── nandadulal_mondir/
    ├── museum/
    └── patalbari/
```

- The `Data` folder must contain subfolders for each heritage site class, with all images placed accordingly.
- Use the provided data preparation notebook to split `Data` into `training_images` and `validate_images` folders for model training and validation.
- Folder names must follow the format: `[site_class]` (e.g., `church`, `museum`, `patalbari`, etc.).
- Update the notebook code to mount your Google Drive and refer to these directories as needed.

> [!IMPORTANT]  
> This folder structure is required for the code and notebooks to properly prepare data for training and validation. For additional site classes, follow the same naming convention.

> [!CAUTION]  
> Adjust the mounting code in the notebooks according to your Google Drive setup.

---

## Usage & Access

> [!WARNING]  
> The original heritage site image dataset used for training and evaluation is stored privately in Google Drive and is not publicly accessible due to privacy and institutional policies. This repository contains only the code (Jupyter notebooks and related scripts).

**To use or reproduce this work:**
1. **Clone the repository:**
    ```bash
    git clone https://github.com/deep-works/Project-Chandannagar-Heritage-Guide.git
    ```
2. **Prepare your data:**  
   Organize your heritage site images into the `Data` folder by class as described above. Use the provided notebook to create `training_images` and `validate_images` folders.
3. **Install dependencies:**  
   All code is provided in Jupyter Notebook format and designed for Google Colab. See the notebooks for required Python libraries and setup instructions.
4. **Run the model:**  
   Follow the instructions in the notebooks to train or evaluate the ResNet18 classifier using your dataset.
5. **Reporting & Visualization:**  
   Refer to the final report notebook for result analysis and project insights.

---

## Citation, License & Contact

*If you are a collaborator or researcher seeking access to the original dataset, please contact the project maintainers for information regarding data permissions.  
Users who utilize this work are requested to cite it appropriately or reference this repository in their project.*

**License:**  
This project is released under the GPL License.

**Contact:**  
For questions or collaboration, please contact [workofficialdeep@gmail.com](mailto:workofficialdeep@gmail.com).
