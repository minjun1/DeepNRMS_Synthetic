# DeepNRMS with Synthetic Examples for CCS Monitoring

This project implements anomaly detection for time-lapse seismic monitoring of CCS (Carbon Capture and Storage) projects. It uses autoencoders and embedding space optimization to identify high-anomaly areas indicative of CO₂ saturation.

[Paper link](https://library.seg.org/doi/abs/10.1190/geo2023-0608.1)

## Features

1. Preprocess seismic data (pre-injection and post-injection).
2. Train a Convolutional Autoencoder (CAE) to extract embeddings.
3. Optimize embedding space to maximize separation of anomaly scores.
4. Evaluate model performance using anomaly score maps.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/minjun1/DeepNRMS_Synthetic.git
    cd DeepNRMS
    ```

2. Install dependencies (need Torch):
    ```bash
    pip install -r requirements.txt
    ```

3. Download pre-injection and post-injection seismic data (you can download them through the first notebook:
    Use the following Google Drive links:
    - Pre-injection: [Download](https://drive.google.com/file/d/15mT2lAsPeq-pAfYUA4rj-ZWDm9rzTKuR/view?usp=sharing)
    - Post-injection: [Download](https://drive.google.com/file/d/17gnKmOJoPwmKvwnk1ZcIj08cErzpslos/view?usp=sharing)

    Save these files in:
    ```
    DeepNRMS_Synthetic/data/scenario6_pre/images_25shot_dx05_pre.npy
    DeepNRMS_Synthetic/data/scenario6_post/images_25shot_dx05_post.npy
    ```
