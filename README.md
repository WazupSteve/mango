# Mango 🥭
> Anime Face Generator using Variational Autoencoder (VAE)

Mango is a simple Variational Autoencoder (VAE) project to generate anime-style face images. It uses PyTorch for building and training the model.

## Project Structure
```
MANGO
├── anime_faces/             # image dataset
├── models/
   └── anime_vae_best.pth    # Saved VAE model
├── samples/
│   └── epoch(1-x).jpg       # Generated images
├── scripts/                 # Main script
│   └── vae.py
├── README.md                # Documentation
├── requirements.txt         # Dependencies
```

## Quick Start
1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare Dataset**:
   Add anime face images (PNG/JPG) to `anime_faces/images/`.

3. **Train the Model**:
   ```bash
   python scripts/vae.py
   ```
   - Trains the model and saves it to `models/anime_vae_best.pth`.
   - Generated samples are saved in `samples/`.

4. **Parameters**:
   - `LATENT_DIM`: 256 (size of latent space)
   - `BATCH_SIZE`: 64
   - `NUM_EPOCHS`: 50
   - `LEARNING_RATE`: 1e-4
   - `BETA`: 1.0 (controls KL divergence strength)

5. **Generate Images**:
   After training, images are saved to `samples/` and `generated_anime_faces_best.png`.

## License
This project is licensed under the MIT License.
