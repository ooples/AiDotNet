---
title: "GraphGenerationModel<T>"
description: "Represents a Graph Generation Model using Variational Autoencoder (VAE) architecture."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

Represents a Graph Generation Model using Variational Autoencoder (VAE) architecture.

## For Beginners

Graph generation creates new graphs similar to training data.

**How it works:**

- Encoder: Compress graph structure into latent space using GNN
- Latent space: Learn probabilistic representation (mean and variance)
- Decoder: Reconstruct graph from latent representation
- Sampling: Generate new graphs by sampling from latent space

**Example - Drug Discovery:**

- Train on known drug molecules
- Learn latent representation of valid molecular structures
- Generate new candidate molecules by sampling
- Filter candidates by predicted properties

**Key Components:**

- **GNN Encoder**: Maps node features to latent space
- **Variational Layer**: Learns mean and log-variance for each node
- **Inner Product Decoder**: Reconstructs adjacency matrix
- **Reparameterization**: Enables gradient flow through sampling

**Loss Function:**

- **Reconstruction Loss**: How well we reconstruct the adjacency matrix
- **KL Divergence**: Regularization to keep latent space well-structured

**Applications:**

- Molecular design and drug discovery
- Social network generation
- Circuit design
- Protein structure generation

## How It Works

Graph generation models learn to generate new graph structures from latent representations.
This implementation uses a Variational Graph Autoencoder (VGAE) approach that learns
a latent distribution of graph structures and can sample new valid graphs.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GraphGenerationModel(Int32,Int32,Int32,Int32,Int32,GraphGenerationType,Double,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,Double,ILearningRateScheduler,GraphGenerationModelOptions)` | Initializes a new instance of the `GraphGenerationModel` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `GenerationType` | Gets the type of graph generation. |
| `HiddenDim` | Gets the hidden dimension for encoder layers. |
| `KLWeight` | KL divergence weight for balancing reconstruction and regularization. |
| `LatentDim` | Gets the latent dimension size. |
| `MaxNodes` | Gets the maximum number of nodes for graph generation. |
| `NumEncoderLayers` | Gets the number of encoder layers. |
| `NumLayers` | Gets the number of layers in the model. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeKLDivergence` | Computes KL divergence from standard normal. |
| `ComputeLoss(Tensor<>,Tensor<>)` | Computes the ELBO loss (reconstruction + KL divergence). |
| `ComputeReconstructionGradient(Tensor<>,Tensor<>)` | Computes gradient of reconstruction loss. |
| `ComputeReconstructionLoss(Tensor<>,Tensor<>)` | Computes binary cross-entropy reconstruction loss. |
| `CreateArchitecture(Int32,Int32,Int32,Int32)` | Creates the encoder architecture without layers (layers are created in InitializeLayers). |
| `CreateNewInstance` | Creates a new instance of this model type. |
| `Decode(Tensor<>)` | Decodes latent representations to reconstruct the adjacency matrix. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Deserializes network-specific data from a binary reader. |
| `Encode(Tensor<>,Tensor<>)` | Encodes node features into latent space representations. |
| `Forward(Tensor<>,Tensor<>)` | Performs a complete forward pass: encode, sample, decode. |
| `Generate(Int32,Int32,Double)` | Generates new graphs by sampling from the latent space. |
| `Generate(Int32,Int32,Tensor<>,Double)` | Generates new graphs by sampling from the latent space with optional conditioning. |
| `GetModelMetadata` | Gets metadata about this model. |
| `GetNamedLayerActivations(Tensor<>)` | Gets the intermediate activations from each layer, ensuring adjacency is set. |
| `GetOptions` |  |
| `GetParameterCount` | Gets the total number of trainable parameters in the network. |
| `GetParameterGradients` | Gets all parameter gradients as a vector (encoder layers + variational weights). |
| `GetParameters` | Gets all parameters as a vector. |
| `InitializeLayers` | Initializes the layers of the neural network based on the provided architecture. |
| `InitializeVariationalWeights` | Initializes the variational layer weights using Xavier initialization. |
| `Interpolate(Tensor<>,Tensor<>,Tensor<>,Tensor<>,Int32)` | Interpolates between two graphs in latent space. |
| `PredictCore(Tensor<>)` | Makes a prediction by encoding input features and decoding to adjacency matrix. |
| `Reparameterize(Tensor<>,Tensor<>)` | Samples from the latent distribution using the reparameterization trick. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Serializes network-specific data to a binary writer. |
| `Train(Tensor<>,Tensor<>)` | Trains the model on a single batch using tape-based backprop through the VGAE objective (Kipf & Welling 2016 §3): L = L_recon + L_KL with L_recon = BCE(sigmoid(Z Z^T), A) and L_KL = -0.5 Σ (1 + log σ² - μ² - exp(log σ²)). |
| `Train(Tensor<>,Tensor<>,Int32,Double)` | Trains the model on graph data. |
| `UpdateParameters(Vector<>)` | Updates the parameters of all layers in the network. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_cachedAdjacencyMatrix` | Cached input adjacency matrix. |
| `_lastEncoderOutput` | Cached encoder output before variational layer. |
| `_lastLatent` | Cached sampled latent representation. |
| `_lastLogVar` | Cached latent log-variance from last forward pass. |
| `_lastMean` | Cached latent mean from last forward pass. |
| `_logVarWeights` | Encoder weights for log-variance projection. |
| `_logVarWeightsGradient` | Gradient for log-variance weights. |
| `_lossFunction` | The loss function used to calculate the reconstruction error. |
| `_meanWeights` | Encoder weights for mean projection. |
| `_meanWeightsGradient` | Gradient for mean weights. |
| `_optimizer` | The optimization algorithm used to update the network's parameters during training. |
| `_random` | Random number generator for sampling. |

