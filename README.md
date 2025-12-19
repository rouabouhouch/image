# Dance Generation Demo

This repository demonstrates the generation of dance animations using various neural network models. The project includes implementations of nearest neighbor search, vanilla neural networks, and GAN-based models for generating images of a dancing character based on skeleton postures.

## Features
- Train models on datasets.
- Generate dance animations using pre-trained models.
- Compare different generation methods: nearest neighbor, vanilla neural networks, and GANs.

## Prerequisites
- Python 3.8+
- PyTorch
- OpenCV
- NumPy
- Matplotlib
- PIL
- torchvision

Install dependencies using the provided `environment.yml` file:
```bash
conda env create -f src/environment.yml
conda activate dancegen
```

## What Was Done
This project implements and compares three methods for generating dance animations:
1. **Nearest Neighbor**: Finds the closest skeleton in the dataset and retrieves the corresponding image.
2. **Vanilla Neural Network**: Trains a simple feedforward network to generate images from skeletons and skeleton images.
3. **GAN**: Uses a conditional GAN to generate high-quality images from skeleton images.


## How It Works

1. **Input**: Two videos:
   - **Source Video**: Contains the movements to be replicated.
   - **Target Video**: Contains the person whose movements will be animated.

2. **Skeleton Extraction**:
   - Skeletons are extracted from the source video.
   - These skeletons are used as input to the generation methods.

3. **Image Generation**:
   - The selected generation method (Nearest Neighbor, Vanilla NN, or GAN) generates images of the target person based on the extracted skeletons.

4. **Output**:
   - The generated images are combined to create a video of the target person performing the movements from the source video.


## How to Train the Networks
Make sure to be in the src folder
1. Train the Vanilla Neural Network:  
   ```bash
   python GenVanillaNN.py
   ```
2. Train the GAN:
   ```bash
   python GenGAN.py
   ```

## How to Run the Code Using the Pre-Trained Network
1. Ensure the pre-trained models are available in the Dance directory:
   - `DanceGenVanillaFromSke26.pth`
   - `DanceGenVanillaFromSkeim.pth`
   - `DanceGenGAN_Cond.pth`
2. Run the demo script in the src folder:
   ```bash
   python DanceDemo.py
   ```
3. Select the generation method by modifying the GEN_TYPE variable in DanceDemo.py :
   - `1`: Nearest Neighbor
   - `2`: Vanilla Neural Network (Skeleton to Image)
   - `3`: Vanilla Neural Network (Skeleton Image to Image)
   - `4`: GAN

## Video Demo
A 2-minute video demonstrating the code in action is available [here](https://youtu.be/JIyQFqSxvso). The video showcases the generation of dance animations using the pre-trained models.
```

```



## **Generation Methods**

#### a. **Nearest Neighbor Search**
- Finds the closest skeleton in the target video to the given skeleton.
- Retrieves the corresponding image from the target video.
- Implemented in `GenNearest.py`.

The nearest neighbor approach is implemented through the `GenNeirest` class. This class provides a simple, non-learning-based baseline for image generation, relying on the assumption that similar skeletal poses correspond to visually similar images.

The constructor `__init__(self, videoSkeTgt)` takes as input a `VideoSkeleton` object representing the target video. This object stores a sequence of skeletons extracted from the video, as well as access to the corresponding RGB frames. The target video serves as a fixed database in which nearest neighbor queries are performed.

The core method of this class is `generate(self, ske)`. Its input is a `Skeleton` object representing a pose extracted from a source video. The output is an RGB image normalized to the range \[0, 1]. Internally, the function iterates over all skeletons stored in the target video and computes the distance between the input skeleton and each target skeleton using the `Skeleton.distance` method. This distance aggregates the Euclidean distances between corresponding joints, providing a global measure of pose similarity.

Once the closest skeleton is identified, the function retrieves the image associated with this skeleton index from the target video. The image is then normalized and returned as the generated result. If the image cannot be loaded, a fallback image is produced to ensure robustness. This method is computationally expensive due to its exhaustive search but offers full interpretability and serves as a strong reference baseline.

---

#### b. **Vanilla Neural Networks**
- Two types of vanilla neural networks are implemented:
  1. **Skeleton-to-Image**: Generates an image directly from the reduced skeleton (26-dimensional input).
  2. **Skeleton Image-to-Image**: Generates an image from a skeleton image (64x64 RGB input).
- Implemented in `GenVanillaNN.py`.

The vanilla neural network approach is implemented in the `GenVanillaNN` class and relies on supervised learning to model the mapping between skeletal poses and corresponding images. This class acts as a wrapper that instantiates the appropriate neural architecture, dataset, and training pipeline depending on the selected input representation.

In the skeleton-to-image configuration, the generator network is defined by the `GenNNSke26ToImage` class. This model takes as input a reduced skeleton represented as a 26-dimensional vector, corresponding to the 2D coordinates of 13 key joints. The input tensor is flattened and passed through a sequence of fully connected layers with ReLU activations. The final layer outputs a vector of size 3×64×64, which is reshaped into an RGB image. A `Tanh` activation function is applied to constrain the output values to the range \[-1, 1], matching the normalization of the training images.

In the skeleton image-to-image configuration, the skeleton is first converted into an image using the `SkeToImageTransform` class. This transformation draws the skeleton on a blank image of fixed size, preserving the spatial structure of the pose. The generator network `GenNNSkeImToImage` is a convolutional encoder-decoder that takes this skeleton image as input and produces a corresponding RGB image. The convolutional architecture allows the model to exploit local spatial correlations that are not available in the vector-based representation.

The data loading and preprocessing logic is handled by the `VideoSkeletonDataset` class. This dataset associates each skeleton with its corresponding video frame. The `__getitem__(self, idx)` method returns a pair consisting of a processed skeleton tensor and a normalized target image tensor. The method `preprocessSkeleton(self, ske)` ensures that skeletons are consistently converted into tensors compatible with the selected generator architecture.

The training process is implemented in the `train(self, n_epochs)` method of `GenVanillaNN`. The generator is optimized using a mean squared error loss between the generated image and the ground truth image. The Adam optimizer is used to ensure stable convergence. After training, the model parameters are saved to disk for later reuse.

The `generate(self, ske)` method is the inference interface of the class. It takes a `Skeleton` object as input, applies the same preprocessing steps as during training, and outputs a denormalized RGB image suitable for visualization. This approach effectively learns an average mapping from pose to appearance but is limited in its ability to generate fine textures or high-frequency details.

---

#### c. **Generative Adversarial Networks (GANs)**
- A conditional GAN is implemented to generate high-quality images.
- The generator uses a U-Net-like architecture with residual blocks for better performance.
- The discriminator evaluates the generated image conditioned on the skeleton image.
- Implemented in `GenGAN.py`.

The most advanced generation method is implemented in the `GenGAN` class, which relies on a conditional Generative Adversarial Network trained using the WGAN-GP framework. The objective is to generate visually realistic images while enforcing consistency with the input skeleton.

The generator network is defined by the `GenSkeImToImage` class. It takes as input a skeleton image and produces an RGB image of the same resolution. The architecture follows an encoder-decoder structure inspired by U-Net, where successive convolutional layers encode the input into a compact latent representation, and transposed convolutions decode this representation back into an image. The final `Tanh` activation ensures that the output lies in the normalized range \[-1, 1].

The discriminator is implemented in the `Discriminator` class and is conditional by design. Its forward method takes two inputs: an RGB image and the corresponding skeleton image. These inputs are concatenated along the channel dimension and processed through a sequence of convolutional layers followed by a fully connected layer that outputs a single scalar score. This score reflects both the realism of the image and its coherence with the conditioning skeleton. Batch normalization layers are deliberately omitted to comply with the theoretical requirements of WGAN-GP.

The stability of the adversarial training is enforced by the `gradient_penalty(self, real, fake, ske)` function. This function computes the gradient penalty term by interpolating between real and generated images and penalizing deviations of the gradient norm from 1. This constraint enforces the Lipschitz continuity of the discriminator and significantly improves training stability.

The training loop is implemented in the `train(self, epochs)` method. The discriminator is optimized to maximize the difference between its scores on real and generated images, while the generator is optimized using a combination of adversarial loss and an L1 reconstruction loss. The reconstruction term encourages fidelity to the ground truth image, while the adversarial term promotes realism and sharpness.

Finally, the `generate(self, ske)` method enables inference with a trained generator. It converts the input `Skeleton` object into a skeleton image, feeds it to the generator, and returns a denormalized RGB image. This GAN-based approach combines the strengths of supervised learning and probabilistic generation, making it the most expressive and visually convincing method in the project.


## Results

### Nearest Neighbor
- Simple and fast.
- Produces results that are limited to the existing frames in the target video.
- Not really satisfying

### Vanilla Neural Networks
- Can generalize to unseen skeletons.
- The quality of the generated images depends on the training data.
- Overall good, but limited to the pc capability

### GANs
- Produces the highest quality images.
- Requires more computational resources and training time.

---

## Acknowledgments

This project was developed as part of the "Vision, image and machine learning" course. Special thanks to Alexandre Meyer for providing the dataset and guidance.