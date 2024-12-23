
# Dual-Decoder Transformer for Mechanism Synthesis Using Coupler Curve Images

[**Access the Full Code on GitHub**](https://github.com/anarnuri/path_generation_transformer)

## Overview

Designing mechanisms capable of following specific trajectories, known as coupler curves, is a core challenge in mechanical engineering and robotics. Mechanisms are essential in applications ranging from robotic arms to automated manufacturing processes. However, traditional methods for mechanism synthesis rely heavily on analytical techniques, which are:

1. **Time-Consuming**: Solving complex equations for mechanism synthesis is computationally expensive, especially for higher-order mechanisms.
2. **Single-Solution Oriented**: These methods typically provide only one mechanism design for a given coupler curve.
3. **Limited in Complexity**: Analytical approaches struggle with mechanisms that have a large number of joints or require intricate trajectories.

### Motivation for a Machine Learning Approach

Machine learning offers an innovative alternative to traditional methods, enabling faster and more diverse solutions. By leveraging data-driven models, engineers can explore a wider range of mechanism designs and automate the synthesis process. This project introduces a **Dual-Decoder Transformer** model, designed specifically for mechanism synthesis. Key innovations include:

- **Coupler Curves as Images**: Input trajectories are represented as grayscale images, enabling the use of convolutional layers for spatial feature extraction.
- **Mechanism Type Embeddings**: Each mechanism type is encoded as a unique feature vector to condition the model's output.
- **Joint Coordinates as Output**: Mechanisms are represented as Cartesian coordinates of their joints, split into two independent parts for simplified learning.

---

## Key Contributions

1. **Mechanism Type Conditioning**:
   - Introduces a dedicated embedding layer for mechanism types, enabling the model to generate designs specific to the input type.

2. **Dual-Decoder Architecture**:
   - Employs two independent decoders:
     - The **first decoder** predicts the first set of joint coordinates.
     - The **second decoder** predicts the second set of joint coordinates.
   - This modular design improves performance for complex mechanisms.

3. **Advanced Loss Masking**:
   - Implements a masked Mean Squared Error (MSE) loss to handle variable-length sequences and padding tokens.

4. **Efficiency and Scalability**:
   - Processes coupler curve images efficiently through patch embeddings and scaled positional encodings.
   - Designed to handle a wide range of mechanism types and complexities.

---

## Iterative Development Process

The development process involved several iterations to refine the architecture and improve performance:

### Initial Attempts

The project began with a single-decoder Transformer model inspired by natural language processing. However, early experiments revealed significant limitations:

- **Poor Performance on Complex Mechanisms**: The single decoder struggled with mechanisms having more than six joints.
- **Limited Scalability**: Increasing the model size improved results slightly but introduced overfitting and longer training times.

### Integration of LLAMA Features

To address these challenges, features from the LLAMA architecture were integrated:

1. **RMS Normalization**:
   - Improved training stability and model convergence.
2. **Scaled Embeddings**:
   - Enhanced input and positional embeddings to capture spatial relationships effectively.
3. **Dynamic Causal Masking**:
   - Ensured that predictions were generated step-by-step during training and inference.

### Introduction of Dual Decoders

A major breakthrough came with the introduction of two independent decoders. This design allowed the model to handle mechanisms of varying complexity by splitting the task into two smaller, more manageable subtasks.

---

## Methodology

### Input Representation

1. **Coupler Curves as Images**:
   - Each trajectory is represented as a 2D grayscale image, divided into patches of fixed size.
   - A convolutional layer extracts features from these patches, which are embedded into a fixed-dimensional vector.

2. **Mechanism Type Embeddings**:
   - Each mechanism type is represented as a unique vector using a learnable embedding layer.
   - The embedding is added to the input sequence to condition the model on the desired mechanism type.

### Model Architecture

1. **Transformer Encoder**:
   - Processes the embedded input sequence (coupler curve patches + mechanism type embedding).
   - Captures spatial relationships and encodes them into a latent representation.

2. **Dual Decoders**:
   - Each decoder independently predicts one part of the mechanism (first and second sets of joint coordinates).
   - Cross-attention layers allow the decoders to leverage information from the encoder's latent representation.

3. **Projection Layers**:
   - Map the decoder outputs back to Cartesian coordinates.

### Training Process

1. **Masked MSE Loss**:
   - Handles variable-length sequences by masking padding tokens during loss computation.
   ```python
   def mse_loss(predictions, targets, mask_value=0.5):
       mask = ~(targets == mask_value).all(dim=-1)
       mask = mask.unsqueeze(-1).expand_as(predictions)
       masked_predictions = predictions[mask]
       masked_targets = targets[mask]
       loss = F.mse_loss(masked_predictions, masked_targets, reduction="mean")
       return loss
   ```

2. **Optimization**:
   - The model is trained using the Adam optimizer with a learning rate scheduler.

3. **Dynamic Causal Masking**:
   - Applied during decoding to ensure stepwise predictions.

---

## Inference Process

During inference, the model generates mechanism designs using a conditional greedy decoding approach:

1. **Encoding**:
   - The coupler curve image is encoded along with the mechanism type embedding.

2. **Decoding**:
   - Each decoder independently predicts its part of the mechanism, conditioned on the encoder's latent representation.

3. **Stopping Condition**:
   - Decoding halts when an End-of-Sequence (EOS) token is detected.

### Code Highlights for Inference
```python
def greedy_decode_conditional(model, source, mech_type, max_len, eos_token=torch.tensor([1.0, 1.0])):
    encoder_output = model.encode(source, None, mech_type)
    decoder_input_first = torch.zeros(1, 1, 2).to(device)
    decoder_input_second = torch.zeros(1, 1, 2).to(device)

    # Decoding for both decoders
    while decoder_input_first.size(1) < max_len // 2:
        ...
    while decoder_input_second.size(1) < max_len // 2:
        ...
```

---

## Applications

1. **Robotics**:
   - Generates diverse designs for robotic mechanisms, such as arms and grippers.

2. **Industrial Design**:
   - Facilitates rapid prototyping of mechanisms for manufacturing.

3. **Education**:
   - Provides a framework for teaching mechanism synthesis concepts using advanced machine learning techniques.

---

## Future Directions

1. **Intra-Type Diversity**:
   - Extend the model to generate multiple mechanisms within the same type.

2. **Scalability**:
   - Adapt the architecture to handle mechanisms with more joints and higher complexities.

3. **Optimization Frameworks**:
   - Integrate the model with optimization algorithms for real-time design applications.

4. **Explainability**:
   - Develop visualizations to interpret the model’s attention mechanisms and latent space.
