---
title: "Neural Networks"
description: "All 547 public types in the AiDotNet.neuralnetworks namespace, organized by kind."
section: "API Reference"
---

**547** public types in this namespace, organized by kind.

## Models & Types (198)

| Type | Summary |
|:-----|:--------|
| [`ACGAN<T>`](/docs/reference/wiki/neuralnetworks/acgan/) | Represents an Auxiliary Classifier Generative Adversarial Network (AC-GAN), which extends conditional GANs by having the discriminator also predict the class label of the input. |
| [`AIMGenerator<T>`](/docs/reference/wiki/neuralnetworks/aimgenerator/) | AIM (Adaptive Iterative Mechanism) generator for differentially private synthetic data generation using marginal-based measurements and iterative optimization. |
| [`AttentionNetwork<T>`](/docs/reference/wiki/neuralnetworks/attentionnetwork/) | Represents a neural network that utilizes attention mechanisms for sequence processing. |
| [`AudioTextDualStreamArchitecture<T>`](/docs/reference/wiki/neuralnetworks/audiotextdualstreamarchitecture/) | A neural network architecture for audio + text two-stream models (CLAP-family encoders) that hosts a separate audio encoder and text encoder. |
| [`AudioVisualCorrespondenceNetwork<T>`](/docs/reference/wiki/neuralnetworks/audiovisualcorrespondencenetwork/) | Audio-visual correspondence learning network for cross-modal understanding. |
| [`AudioVisualEventLocalizationNetwork<T>`](/docs/reference/wiki/neuralnetworks/audiovisualeventlocalizationnetwork/) | Neural network for audio-visual event localization - identifying WHEN and WHERE events occur in video by jointly analyzing audio and visual streams with precise temporal boundaries. |
| [`AutoDiffTabGenerator<T>`](/docs/reference/wiki/neuralnetworks/autodifftabgenerator/) | AutoDiff-Tab generator that automatically searches over diffusion configurations (timesteps, noise schedules, network architecture) to find optimal settings for tabular data generation. |
| [`AutoIntClassifier<T>`](/docs/reference/wiki/neuralnetworks/autointclassifier/) | AutoInt implementation for classification tasks. |
| [`AutoIntNetwork<T>`](/docs/reference/wiki/neuralnetworks/autointnetwork/) | AutoInt (Automatic Feature Interaction Learning) neural network for tabular data. |
| [`AutoIntRegression<T>`](/docs/reference/wiki/neuralnetworks/autointregression/) | AutoInt implementation for regression tasks. |
| [`Autoencoder<T>`](/docs/reference/wiki/neuralnetworks/autoencoder/) | Represents an autoencoder neural network that can compress data into a lower-dimensional representation and reconstruct it. |
| [`BGE<T>`](/docs/reference/wiki/neuralnetworks/bge/) | BGE (BAAI General Embedding) neural network implementation. |
| [`BayesianNetworkSynthGenerator<T>`](/docs/reference/wiki/neuralnetworks/bayesiannetworksynthgenerator/) | Bayesian Network Synthesis generator that learns a DAG structure over features, estimates conditional probability tables (CPTs), and generates synthetic data via ancestral sampling. |
| [`BigGAN<T>`](/docs/reference/wiki/neuralnetworks/biggan/) | BigGAN implementation for large-scale high-fidelity image generation. |
| [`Blip2NeuralNetwork<T>`](/docs/reference/wiki/neuralnetworks/blip2neuralnetwork/) | BLIP-2 (Bootstrapped Language-Image Pre-training 2) neural network for vision-language tasks. |
| [`BlipNeuralNetwork<T>`](/docs/reference/wiki/neuralnetworks/blipneuralnetwork/) | BLIP (Bootstrapped Language-Image Pre-training) neural network for vision-language tasks. |
| [`CLSToken<T>`](/docs/reference/wiki/neuralnetworks/clstoken/) | CLS (Classification) Token for transformer-based tabular models. |
| [`CTABGANPlusGenerator<T>`](/docs/reference/wiki/neuralnetworks/ctabganplusgenerator/) | CTAB-GAN+ generator for high-quality synthetic tabular data with auxiliary classifier discriminator, mixed-type encoder, and information loss. |
| [`CTGANDataSampler<T>`](/docs/reference/wiki/neuralnetworks/ctgandatasampler/) | Handles conditional vector generation and training-by-sampling for CTGAN. |
| [`CTGANGenerator<T>`](/docs/reference/wiki/neuralnetworks/ctgangenerator/) | Conditional Tabular GAN (CTGAN) for generating realistic synthetic tabular data. |
| [`CapsuleNetwork<T>`](/docs/reference/wiki/neuralnetworks/capsulenetwork/) | Represents a Capsule Network, a type of neural network that preserves spatial relationships between features. |
| [`CausalGANGenerator<T>`](/docs/reference/wiki/neuralnetworks/causalgangenerator/) | Causal-GAN generator that learns causal graph structure (directed acyclic graph) and generates synthetic data respecting causal relationships between features. |
| [`ClipNeuralNetwork<T>`](/docs/reference/wiki/neuralnetworks/clipneuralnetwork/) | CLIP (Contrastive Language-Image Pre-training) neural network that encodes both text and images into a shared embedding space, enabling cross-modal similarity and zero-shot classification. |
| [`ColBERT<T>`](/docs/reference/wiki/neuralnetworks/colbert/) | ColBERT (Contextualized Late Interaction over BERT) neural network implementation. |
| [`ColumnEmbedding<T>`](/docs/reference/wiki/neuralnetworks/columnembedding/) | Column (positional) embedding for tabular transformers like TabTransformer. |
| [`ColumnMetadata`](/docs/reference/wiki/neuralnetworks/columnmetadata/) | Describes the metadata for a single column in a tabular dataset, including its name, data type, categories (for categorical columns), and summary statistics. |
| [`ColumnTransformInfo`](/docs/reference/wiki/neuralnetworks/columntransforminfo/) | Describes how a single original column maps into the transformed representation. |
| [`ConditionalGAN<T>`](/docs/reference/wiki/neuralnetworks/conditionalgan/) | Represents a Conditional Generative Adversarial Network (cGAN), which generates data conditioned on additional information such as class labels, attributes, or other contextual data. |
| [`Connection<T>`](/docs/reference/wiki/neuralnetworks/connection/) | Represents a connection between two nodes in a neural network, particularly used in evolving neural networks. |
| [`ContextEncoder<T>`](/docs/reference/wiki/neuralnetworks/contextencoder/) | Context encoder for TabR that processes retrieved neighbors into a context representation. |
| [`ContrastivePretraining<T>`](/docs/reference/wiki/neuralnetworks/contrastivepretraining/) | Contrastive pretraining module for SAINT architecture. |
| [`ConvolutionalNeuralNetwork<T>`](/docs/reference/wiki/neuralnetworks/convolutionalneuralnetwork/) | Represents a Convolutional Neural Network (CNN) that processes multi-dimensional data. |
| [`CopulaGANGenerator<T>`](/docs/reference/wiki/neuralnetworks/copulagangenerator/) | CopulaGAN generator for synthetic tabular data, combining Gaussian copula transformations with the CTGAN training pipeline for improved continuous column modeling. |
| [`CopulaSynthGenerator<T>`](/docs/reference/wiki/neuralnetworks/copulasynthgenerator/) | Copula-Based Synthesis generator that models marginal distributions independently and couples them with a Gaussian copula to capture inter-feature dependencies. |
| [`CycleGAN<T>`](/docs/reference/wiki/neuralnetworks/cyclegan/) | Represents a CycleGAN for unpaired image-to-image translation. |
| [`DCGAN<T>`](/docs/reference/wiki/neuralnetworks/dcgan/) | Represents a Deep Convolutional Generative Adversarial Network (DCGAN), an architecture that uses convolutional and transposed convolutional layers with specific design guidelines for stable training. |
| [`DPCTGANGenerator<T>`](/docs/reference/wiki/neuralnetworks/dpctgangenerator/) | Differentially Private CTGAN (DP-CTGAN) for generating synthetic tabular data with formal (epsilon, delta)-differential privacy guarantees. |
| [`DeepBeliefNetwork<T>`](/docs/reference/wiki/neuralnetworks/deepbeliefnetwork/) | Represents a Deep Belief Network, a generative graphical model composed of multiple layers of Restricted Boltzmann Machines. |
| [`DeepBoltzmannMachine<T>`](/docs/reference/wiki/neuralnetworks/deepboltzmannmachine/) | Represents a Deep Boltzmann Machine (DBM), a hierarchical generative model consisting of multiple layers of stochastic neurons. |
| [`DeepQNetwork<T>`](/docs/reference/wiki/neuralnetworks/deepqnetwork/) | Represents a Deep Q-Network (DQN), a reinforcement learning algorithm that combines Q-learning with deep neural networks. |
| [`DenseNetNetwork<T>`](/docs/reference/wiki/neuralnetworks/densenetnetwork/) | Implements the DenseNet (Densely Connected Convolutional Network) architecture. |
| [`DifferentiableNeuralComputer<T>`](/docs/reference/wiki/neuralnetworks/differentiableneuralcomputer/) | Represents a Differentiable Neural Computer (DNC), a neural network architecture that combines neural networks with external memory resources. |
| [`DualStreamArchitecture<T>`](/docs/reference/wiki/neuralnetworks/dualstreamarchitecture/) | A neural network architecture for vision + text two-stream models (CLIP-family encoders: BASIC, DFNCLIP, EVACLIP, LiT, RegionCLIP, etc.) that hosts a separate vision encoder and text encoder. |
| [`EagleLanguageModel<T>`](/docs/reference/wiki/neuralnetworks/eaglelanguagemodel/) | Implements a full RWKV-5 "Eagle" language model: token embedding + N RWKVLayer blocks + RMS normalization + LM head. |
| [`EchoStateNetwork<T>`](/docs/reference/wiki/neuralnetworks/echostatenetwork/) | Represents an Echo State Network (ESN), a type of recurrent neural network with a sparsely connected hidden layer called a reservoir. |
| [`EfficientNetNetwork<T>`](/docs/reference/wiki/neuralnetworks/efficientnetnetwork/) | Implements the EfficientNet architecture with compound scaling. |
| [`EntmaxAttention<T>`](/docs/reference/wiki/neuralnetworks/entmaxattention/) | Entmax sparse attention function for NODE architecture. |
| [`EntmoidActivation<T>`](/docs/reference/wiki/neuralnetworks/entmoidactivation/) | Entmoid activation function for NODE architecture. |
| [`ExtremeLearningMachine<T>`](/docs/reference/wiki/neuralnetworks/extremelearningmachine/) | Represents an Extreme Learning Machine (ELM), a type of feedforward neural network with a unique training approach. |
| [`FTTransformerClassifier<T>`](/docs/reference/wiki/neuralnetworks/fttransformerclassifier/) | FT-Transformer implementation for classification tasks. |
| [`FTTransformerNetwork<T>`](/docs/reference/wiki/neuralnetworks/fttransformernetwork/) | FT-Transformer (Feature Tokenizer + Transformer) neural network for tabular data. |
| [`FTTransformerRegression<T>`](/docs/reference/wiki/neuralnetworks/fttransformerregression/) | FT-Transformer implementation for regression tasks. |
| [`FalconMambaLanguageModel<T>`](/docs/reference/wiki/neuralnetworks/falconmambalanguagemodel/) | Implements a Falcon Mamba language model: embedding + N MambaBlock blocks + RMS norm + LM head. |
| [`FastText<T>`](/docs/reference/wiki/neuralnetworks/fasttext/) | FastText neural network implementation, an extension of Word2Vec that considers subword information. |
| [`FeatureTokenizer<T>`](/docs/reference/wiki/neuralnetworks/featuretokenizer/) | Implements the Feature Tokenizer that converts tabular features into embeddings for FT-Transformer. |
| [`FeedForwardNeuralNetwork<T>`](/docs/reference/wiki/neuralnetworks/feedforwardneuralnetwork/) | Represents a Feed-Forward Neural Network (FFNN) for processing data in a forward path. |
| [`FinDiffGenerator<T>`](/docs/reference/wiki/neuralnetworks/findiffgenerator/) | FinDiff generator for synthesizing realistic financial tabular data using diffusion models with temporal correlation preservation and financial constraint enforcement. |
| [`FinchLanguageModel<T>`](/docs/reference/wiki/neuralnetworks/finchlanguagemodel/) | Implements a full RWKV-6 "Finch" language model: token embedding + N RWKVLayer blocks + RMS normalization + LM head. |
| [`FlamingoNeuralNetwork<T>`](/docs/reference/wiki/neuralnetworks/flamingoneuralnetwork/) | Flamingo neural network for in-context visual learning and few-shot tasks. |
| [`GANDALFClassifier<T>`](/docs/reference/wiki/neuralnetworks/gandalfclassifier/) | GANDALF implementation for classification tasks. |
| [`GANDALFNetwork<T>`](/docs/reference/wiki/neuralnetworks/gandalfnetwork/) | GANDALF (Gated Additive Neural Decision Forest) neural network for tabular data. |
| [`GANDALFRegression<T>`](/docs/reference/wiki/neuralnetworks/gandalfregression/) | GANDALF implementation for regression tasks. |
| [`GLALanguageModel<T>`](/docs/reference/wiki/neuralnetworks/glalanguagemodel/) | Implements a full GLA (Gated Linear Attention) language model: embedding + N GLA blocks + RMS norm + LM head. |
| [`GOGGLEGenerator<T>`](/docs/reference/wiki/neuralnetworks/gogglegenerator/) | GOGGLE generator that learns feature dependency structure via a graph neural network combined with a VAE framework for high-quality synthetic tabular data generation. |
| [`GRUNeuralNetwork<T>`](/docs/reference/wiki/neuralnetworks/gruneuralnetwork/) | Represents a Gated Recurrent Unit (GRU) Neural Network for processing sequential data. |
| [`GatedDeltaNetLanguageModel<T>`](/docs/reference/wiki/neuralnetworks/gateddeltanetlanguagemodel/) | Implements a full Gated DeltaNet language model: token embedding + N GatedDeltaNetLayer blocks + RMS norm + LM head. |
| [`GaussianDiffusion<T>`](/docs/reference/wiki/neuralnetworks/gaussiandiffusion/) | Implements the Gaussian diffusion process for continuous/numerical features in TabDDPM. |
| [`GenerativeAdversarialNetwork<T>`](/docs/reference/wiki/neuralnetworks/generativeadversarialnetwork/) | Represents a Generative Adversarial Network (GAN), a deep learning architecture that consists of two neural networks (a generator and a discriminator) competing against each other in a zero-sum game. |
| [`Genome<T>`](/docs/reference/wiki/neuralnetworks/genome/) | Represents a genome in a neuroevolutionary algorithm, containing a collection of connections between nodes. |
| [`GhostBatchNormalization<T>`](/docs/reference/wiki/neuralnetworks/ghostbatchnormalization/) | Implements Ghost Batch Normalization, a regularization technique used in TabNet that applies batch normalization to virtual mini-batches within each actual batch. |
| [`GloVe<T>`](/docs/reference/wiki/neuralnetworks/glove/) | GloVe (Global Vectors for Word Representation) neural network implementation. |
| [`Gpt4VisionNeuralNetwork<T>`](/docs/reference/wiki/neuralnetworks/gpt4visionneuralnetwork/) | GPT-4V-style neural network that combines vision understanding with large language model capabilities. |
| [`GraphAttentionNetwork<T>`](/docs/reference/wiki/neuralnetworks/graphattentionnetwork/) | Represents a Graph Attention Network (GAT) that uses attention mechanisms to process graph-structured data. |
| [`GraphClassificationModel<T>`](/docs/reference/wiki/neuralnetworks/graphclassificationmodel/) | Implements a complete neural network model for graph classification tasks. |
| [`GraphGenerationModel<T>`](/docs/reference/wiki/neuralnetworks/graphgenerationmodel/) | Represents a Graph Generation Model using Variational Autoencoder (VAE) architecture. |
| [`GraphIsomorphismNetwork<T>`](/docs/reference/wiki/neuralnetworks/graphisomorphismnetwork/) | Represents a Graph Isomorphism Network (GIN) for powerful graph representation learning. |
| [`GraphNeuralNetwork<T>`](/docs/reference/wiki/neuralnetworks/graphneuralnetwork/) | Represents a Graph Neural Network that can process data represented as graphs. |
| [`GraphSAGENetwork<T>`](/docs/reference/wiki/neuralnetworks/graphsagenetwork/) | Represents a GraphSAGE (Graph Sample and Aggregate) Network for inductive learning on graphs. |
| [`GriffinLanguageModel<T>`](/docs/reference/wiki/neuralnetworks/griffinlanguagemodel/) | Implements a Griffin language model: embedding + N RGLR blocks with local attention + layer norm + LM head. |
| [`HTMNetwork<T>`](/docs/reference/wiki/neuralnetworks/htmnetwork/) | Represents a Hierarchical Temporal Memory (HTM) network, a biologically-inspired sequence learning algorithm. |
| [`HawkLanguageModel<T>`](/docs/reference/wiki/neuralnetworks/hawklanguagemodel/) | Implements a Hawk language model: embedding + N pure RGLR blocks + layer norm + LM head. |
| [`HopeNetwork<T>`](/docs/reference/wiki/neuralnetworks/hopenetwork/) | Hope architecture - a self-modifying recurrent neural network variant of Titans with unbounded levels of in-context learning. |
| [`HopfieldNetwork<T>`](/docs/reference/wiki/neuralnetworks/hopfieldnetwork/) | Represents a Hopfield Network, a recurrent neural network designed for pattern storage and retrieval. |
| [`HyperbolicNeuralNetwork<T>`](/docs/reference/wiki/neuralnetworks/hyperbolicneuralnetwork/) | Represents a Hyperbolic Neural Network for learning hierarchical representations in Poincare ball space. |
| [`ImageBindNeuralNetwork<T>`](/docs/reference/wiki/neuralnetworks/imagebindneuralnetwork/) | ImageBind neural network for binding multiple modalities (6+) into a shared embedding space. |
| [`InContextLearning<T>`](/docs/reference/wiki/neuralnetworks/incontextlearning/) | Helper class for in-context learning in tabular foundation models like TabPFN. |
| [`InfoGAN<T>`](/docs/reference/wiki/neuralnetworks/infogan/) | Represents an Information Maximizing Generative Adversarial Network (InfoGAN), which learns disentangled representations in an unsupervised manner by maximizing mutual information between latent codes and generated observations. |
| [`InstructorEmbedding<T>`](/docs/reference/wiki/neuralnetworks/instructorembedding/) | Instructor/E5 (Instruction-Tuned) embedding model implementation. |
| [`JambaLanguageModel<T>`](/docs/reference/wiki/neuralnetworks/jambalanguagemodel/) | Implements a Jamba language model: embedding + HybridBlockScheduler (Mamba + Attention) + RMS norm + LM head. |
| [`LLaVANeuralNetwork<T>`](/docs/reference/wiki/neuralnetworks/llavaneuralnetwork/) | LLaVA (Large Language and Vision Assistant) neural network for visual instruction following. |
| [`LSTMNeuralNetwork<T>`](/docs/reference/wiki/neuralnetworks/lstmneuralnetwork/) | Represents a Long Short-Term Memory (LSTM) Neural Network, which is specialized for processing sequential data like text, time series, or audio. |
| [`LinkPredictionModel<T>`](/docs/reference/wiki/neuralnetworks/linkpredictionmodel/) | Implements a complete neural network model for link prediction tasks on graphs. |
| [`LiquidStateMachine<T>`](/docs/reference/wiki/neuralnetworks/liquidstatemachine/) | Represents a Liquid State Machine (LSM), a type of reservoir computing neural network. |
| [`Mamba2LanguageModel<T>`](/docs/reference/wiki/neuralnetworks/mamba2languagemodel/) | Implements a Mamba-2 language model: token embedding + N Mamba2Blocks + layer normalization + LM head. |
| [`MambaLanguageModel<T>`](/docs/reference/wiki/neuralnetworks/mambalanguagemodel/) | Implements a full Mamba language model: token embedding + N MambaBlocks + RMS normalization + LM head. |
| [`MambaModelState<T>`](/docs/reference/wiki/neuralnetworks/mambamodelstate/) | Per-token (KV-cached) decoding state for a `MambaLanguageModel`: one `MambaStepState` per Mamba block, in layer order. |
| [`MambularClassifier<T>`](/docs/reference/wiki/neuralnetworks/mambularclassifier/) | Mambular implementation for classification tasks. |
| [`MambularNetwork<T>`](/docs/reference/wiki/neuralnetworks/mambularnetwork/) | Mambular (State Space Model for Tabular Data) neural network. |
| [`MambularRegression<T>`](/docs/reference/wiki/neuralnetworks/mambularregression/) | Mambular implementation for regression tasks. |
| [`MatryoshkaEmbedding<T>`](/docs/reference/wiki/neuralnetworks/matryoshkaembedding/) | Matryoshka Representation Learning (MRL) neural network implementation. |
| [`MedSynthGenerator<T>`](/docs/reference/wiki/neuralnetworks/medsynthgenerator/) | MedSynth generator for privacy-preserving medical tabular data synthesis using a VAE/GAN hybrid with clinical validity constraints and optional differential privacy. |
| [`MemoryNetwork<T>`](/docs/reference/wiki/neuralnetworks/memorynetwork/) | Represents a Memory Network, a neural network architecture designed with explicit memory components for improved reasoning and question answering capabilities. |
| [`MeshCNN<T>`](/docs/reference/wiki/neuralnetworks/meshcnn/) | Implements the MeshCNN architecture for processing 3D triangle meshes. |
| [`MisGANGenerator<T>`](/docs/reference/wiki/neuralnetworks/misgangenerator/) | MisGAN generator for learning from incomplete data using dual generator/discriminator pairs for both data values and missingness patterns. |
| [`MixtureOfExpertsNeuralNetwork<T>`](/docs/reference/wiki/neuralnetworks/mixtureofexpertsneuralnetwork/) | Represents a Mixture-of-Experts (MoE) neural network that routes inputs through multiple specialist networks. |
| [`MobileNetV2Network<T>`](/docs/reference/wiki/neuralnetworks/mobilenetv2network/) | Implements the MobileNetV2 architecture for efficient mobile inference. |
| [`MobileNetV3Network<T>`](/docs/reference/wiki/neuralnetworks/mobilenetv3network/) | Implements the MobileNetV3 architecture for efficient mobile inference. |
| [`MultinomialDiffusion<T>`](/docs/reference/wiki/neuralnetworks/multinomialdiffusion/) | Implements the multinomial diffusion process for categorical features in TabDDPM. |
| [`NEAT<T>`](/docs/reference/wiki/neuralnetworks/neat/) | Represents a NeuroEvolution of Augmenting Topologies (NEAT) algorithm implementation, which evolves neural networks through genetic algorithms. |
| [`NODEClassifier<T>`](/docs/reference/wiki/neuralnetworks/nodeclassifier/) | NODE implementation for classification tasks. |
| [`NODENetwork<T>`](/docs/reference/wiki/neuralnetworks/nodenetwork/) | NODE (Neural Oblivious Decision Ensembles) neural network for tabular data. |
| [`NODERegression<T>`](/docs/reference/wiki/neuralnetworks/noderegression/) | NODE implementation for regression tasks. |
| [`NeuralNetworkArchitecture<T>`](/docs/reference/wiki/neuralnetworks/neuralnetworkarchitecture/) | Defines the structure and configuration of a neural network, including its layers, input/output dimensions, and task-specific properties. |
| [`NeuralNetwork<T>`](/docs/reference/wiki/neuralnetworks/neuralnetwork/) | A neural network implementation that processes data through multiple layers to make predictions. |
| [`NeuralTuringMachine<T>`](/docs/reference/wiki/neuralnetworks/neuralturingmachine/) | Represents a Neural Turing Machine, which is a neural network architecture that combines a neural network with external memory. |
| [`NodeClassificationModel<T>`](/docs/reference/wiki/neuralnetworks/nodeclassificationmodel/) | Implements a complete neural network model for node classification tasks on graphs. |
| [`OCTGANGenerator<T>`](/docs/reference/wiki/neuralnetworks/octgangenerator/) | OCT-GAN (One-Class Tabular GAN) generator for synthesizing minority-class tabular data using a one-class discriminator with Deep SVDD (Support Vector Data Description) objective. |
| [`OccupancyNeuralNetwork<T>`](/docs/reference/wiki/neuralnetworks/occupancyneuralnetwork/) | Represents a Neural Network specialized for occupancy detection and prediction in spaces. |
| [`OctonionNeuralNetwork<T>`](/docs/reference/wiki/neuralnetworks/octonionneuralnetwork/) | Represents an Octonion-valued Neural Network for processing data in 8-dimensional hypercomplex space. |
| [`PATEGANGenerator<T>`](/docs/reference/wiki/neuralnetworks/pategangenerator/) | PATE-GAN generator for differentially private synthetic tabular data generation using the Private Aggregation of Teacher Ensembles (PATE) framework. |
| [`Pix2Pix<T>`](/docs/reference/wiki/neuralnetworks/pix2pix/) | Represents a Pix2Pix GAN for paired image-to-image translation tasks. |
| [`PlanCache`](/docs/reference/wiki/neuralnetworks/plancache/) | Disk-backed store for compiled inference plans. |
| [`ProgressiveGAN<T>`](/docs/reference/wiki/neuralnetworks/progressivegan/) | Production-ready Progressive GAN (ProGAN) implementation that generates high-resolution images by progressively growing the generator and discriminator during training. |
| [`QuantumNeuralNetwork<T>`](/docs/reference/wiki/neuralnetworks/quantumneuralnetwork/) | Represents a Quantum Neural Network, which combines quantum computing principles with neural network architecture. |
| [`REaLTabFormerGenerator<T>`](/docs/reference/wiki/neuralnetworks/realtabformergenerator/) | REaLTabFormer generator using GPT-2 style autoregressive transformer for synthetic tabular data generation by treating columns as a sequence of tokens. |
| [`RWKV4LanguageModel<T>`](/docs/reference/wiki/neuralnetworks/rwkv4languagemodel/) | Implements a full RWKV-4 language model: token embedding + N RWKVLayer blocks + layer normalization + LM head. |
| [`RWKV7LanguageModel<T>`](/docs/reference/wiki/neuralnetworks/rwkv7languagemodel/) | Implements a full RWKV-7 "Goose" language model: token embedding + N RWKV7Blocks + RMS normalization + LM head. |
| [`RadialBasisFunctionNetwork<T>`](/docs/reference/wiki/neuralnetworks/radialbasisfunctionnetwork/) | Represents a Radial Basis Function Network, which is a type of neural network that uses radial basis functions as activation functions. |
| [`RecurrentGemmaLanguageModel<T>`](/docs/reference/wiki/neuralnetworks/recurrentgemmalanguagemodel/) | Implements a RecurrentGemma language model: embedding + N RGLR blocks + layer norm + LM head. |
| [`RecurrentNeuralNetwork<T>`](/docs/reference/wiki/neuralnetworks/recurrentneuralnetwork/) | Represents a Recurrent Neural Network, which is a type of neural network designed to process sequential data by maintaining an internal state. |
| [`ResNetNetwork<T>`](/docs/reference/wiki/neuralnetworks/resnetnetwork/) | Represents a ResNet (Residual Network) neural network architecture for image classification. |
| [`ResidualNeuralNetwork<T>`](/docs/reference/wiki/neuralnetworks/residualneuralnetwork/) | Represents a Residual Neural Network, which is a type of neural network that uses skip connections to address the vanishing gradient problem in deep networks. |
| [`RestrictedBoltzmannMachine<T>`](/docs/reference/wiki/neuralnetworks/restrictedboltzmannmachine/) | Represents a Restricted Boltzmann Machine, which is a type of neural network that learns probability distributions over its inputs. |
| [`RetrievalContext<T>`](/docs/reference/wiki/neuralnetworks/retrievalcontext/) | Context returned by the retrieval module. |
| [`RetrievalModule<T>`](/docs/reference/wiki/neuralnetworks/retrievalmodule/) | Retrieval module for TabR (Retrieval-Augmented Tabular Learning). |
| [`SAGAN<T>`](/docs/reference/wiki/neuralnetworks/sagan/) | Self-Attention GAN (SAGAN) implementation that uses self-attention mechanisms to model long-range dependencies in generated images. |
| [`SAINTClassifier<T>`](/docs/reference/wiki/neuralnetworks/saintclassifier/) | SAINT implementation for classification tasks. |
| [`SAINTNetwork<T>`](/docs/reference/wiki/neuralnetworks/saintnetwork/) | SAINT (Self-Attention and Intersample Attention Transformer) neural network for tabular data. |
| [`SAINTRegression<T>`](/docs/reference/wiki/neuralnetworks/saintregression/) | SAINT implementation for regression tasks. |
| [`SGPT<T>`](/docs/reference/wiki/neuralnetworks/sgpt/) | SGPT (Sentence GPT) neural network implementation using decoder-only transformer architectures. |
| [`SMOTENCGenerator<T>`](/docs/reference/wiki/neuralnetworks/smotencgenerator/) | SMOTE-NC generator that creates synthetic minority samples by interpolating between existing minority samples and their k-nearest neighbors, supporting both continuous and categorical features. |
| [`SPLADE<T>`](/docs/reference/wiki/neuralnetworks/splade/) | SPLADE (Sparse Lexical and Expansion Model) neural network implementation. |
| [`SambaLanguageModel<T>`](/docs/reference/wiki/neuralnetworks/sambalanguagemodel/) | Implements a Samba language model: embedding + HybridBlockScheduler (Mamba + sliding window attention) + RMS norm + LM head. |
| [`SelfOrganizingMap<T>`](/docs/reference/wiki/neuralnetworks/selforganizingmap/) | Represents a Self-Organizing Map, which is an unsupervised neural network that produces a low-dimensional representation of input data. |
| [`SiameseNetwork<T>`](/docs/reference/wiki/neuralnetworks/siamesenetwork/) | Implements a Siamese Neural Network for comparing pairs of inputs and determining their similarity. |
| [`SiameseNeuralNetwork<T>`](/docs/reference/wiki/neuralnetworks/siameseneuralnetwork/) | Sentence-BERT (SBERT) style shared sentence-encoder tower: a transformer encoder that maps a tokenized input to a fixed-size embedding (default 768-d, BERT vocab 30522, max length 512) for semantic similarity and retrieval. |
| [`SimCSE<T>`](/docs/reference/wiki/neuralnetworks/simcse/) | SimCSE (Simple Contrastive Learning of Sentence Embeddings) neural network implementation. |
| [`SparseNeuralNetwork<T>`](/docs/reference/wiki/neuralnetworks/sparseneuralnetwork/) | Represents a Sparse Neural Network with efficient sparse weight matrices. |
| [`Sparsemax<T>`](/docs/reference/wiki/neuralnetworks/sparsemax/) | Implements the Sparsemax activation function, which projects input onto the probability simplex with sparse outputs (many exact zeros). |
| [`SpikingNeuralNetwork<T>`](/docs/reference/wiki/neuralnetworks/spikingneuralnetwork/) | Represents a Spiking Neural Network, which is a type of neural network that more closely models biological neurons with temporal dynamics. |
| [`SpiralNet<T>`](/docs/reference/wiki/neuralnetworks/spiralnet/) | Implements the SpiralNet++ architecture for mesh-based deep learning. |
| [`StyleGAN<T>`](/docs/reference/wiki/neuralnetworks/stylegan/) | Represents a StyleGAN (Style-Based Generator Architecture for GANs) that generates high-quality images with fine-grained control over image style at different levels. |
| [`SuperNet<T>`](/docs/reference/wiki/neuralnetworks/supernet/) | SuperNet implementation for gradient-based neural architecture search (DARTS). |
| [`TVAEGenerator<T>`](/docs/reference/wiki/neuralnetworks/tvaegenerator/) | Tabular Variational Autoencoder (TVAE) for generating synthetic tabular data. |
| [`TabDDPMGenerator<T>`](/docs/reference/wiki/neuralnetworks/tabddpmgenerator/) | TabDDPM (Tabular Denoising Diffusion Probabilistic Model) for generating synthetic tabular data. |
| [`TabDPTClassifier<T>`](/docs/reference/wiki/neuralnetworks/tabdptclassifier/) | TabDPT implementation for classification tasks. |
| [`TabDPTNetwork<T>`](/docs/reference/wiki/neuralnetworks/tabdptnetwork/) | TabDPT (Tabular Data Pre-Training) neural network for tabular data. |
| [`TabDPTRegression<T>`](/docs/reference/wiki/neuralnetworks/tabdptregression/) | TabDPT implementation for regression tasks. |
| [`TabFlowGenerator<T>`](/docs/reference/wiki/neuralnetworks/tabflowgenerator/) | TabFlow generator using flow matching with optimal transport conditional paths for high-quality, fast synthetic tabular data generation. |
| [`TabLLMGenGenerator<T>`](/docs/reference/wiki/neuralnetworks/tabllmgengenerator/) | TabLLM-Gen generator that uses LLM-style schema-aware tokenization and autoregressive transformers to generate realistic tabular data. |
| [`TabMClassifier<T>`](/docs/reference/wiki/neuralnetworks/tabmclassifier/) | TabM implementation for classification tasks. |
| [`TabMNetwork<T>`](/docs/reference/wiki/neuralnetworks/tabmnetwork/) | TabM (Parameter-Efficient Ensemble) neural network for tabular data. |
| [`TabMRegression<T>`](/docs/reference/wiki/neuralnetworks/tabmregression/) | TabM implementation for regression tasks. |
| [`TabNetClassifier<T>`](/docs/reference/wiki/neuralnetworks/tabnetclassifier/) | TabNet implementation for classification tasks. |
| [`TabNetNetwork<T>`](/docs/reference/wiki/neuralnetworks/tabnetnetwork/) | TabNet neural network for interpretable tabular learning. |
| [`TabNetRegression<T>`](/docs/reference/wiki/neuralnetworks/tabnetregression/) | TabNet implementation for regression tasks. |
| [`TabPFNClassifier<T>`](/docs/reference/wiki/neuralnetworks/tabpfnclassifier/) | TabPFN implementation for classification tasks. |
| [`TabPFNNetwork<T>`](/docs/reference/wiki/neuralnetworks/tabpfnnetwork/) | TabPFN (Prior-Fitted Network) neural network for tabular data. |
| [`TabPFNRegression<T>`](/docs/reference/wiki/neuralnetworks/tabpfnregression/) | TabPFN implementation for regression tasks. |
| [`TabRClassifier<T>`](/docs/reference/wiki/neuralnetworks/tabrclassifier/) | TabR implementation for classification tasks. |
| [`TabRNetwork<T>`](/docs/reference/wiki/neuralnetworks/tabrnetwork/) | TabR (Retrieval-Augmented) neural network for tabular data. |
| [`TabRRegression<T>`](/docs/reference/wiki/neuralnetworks/tabrregression/) | TabR implementation for regression tasks. |
| [`TabSynGenerator<T>`](/docs/reference/wiki/neuralnetworks/tabsyngenerator/) | TabSyn generator combining VAE pretraining with latent diffusion for state-of-the-art synthetic tabular data generation. |
| [`TabTransformerClassifier<T>`](/docs/reference/wiki/neuralnetworks/tabtransformerclassifier/) | TabTransformer implementation for classification tasks. |
| [`TabTransformerGenGenerator<T>`](/docs/reference/wiki/neuralnetworks/tabtransformergengenerator/) | TabTransformer-Gen generator that uses column-wise contextual embeddings and masked prediction to generate realistic tabular data. |
| [`TabTransformerNetwork<T>`](/docs/reference/wiki/neuralnetworks/tabtransformernetwork/) | TabTransformer neural network for tabular data with categorical features. |
| [`TabTransformerRegression<T>`](/docs/reference/wiki/neuralnetworks/tabtransformerregression/) | TabTransformer implementation for regression tasks. |
| [`TableGANGenerator<T>`](/docs/reference/wiki/neuralnetworks/tablegangenerator/) | TableGAN generator using a DCGAN-style architecture with classification and information loss for high-quality synthetic tabular data generation. |
| [`TabularDataTransformer<T>`](/docs/reference/wiki/neuralnetworks/tabulardatatransformer/) | Transforms tabular data using Variational Gaussian Mixture (VGM) mode-specific normalization for continuous columns and one-hot encoding for categorical columns. |
| [`TimeGANGenerator<T>`](/docs/reference/wiki/neuralnetworks/timegangenerator/) | TimeGAN generator for synthesizing realistic time-series tabular data while preserving temporal dynamics using an embedding-supervisor-adversarial training framework. |
| [`TransformerArchitecture<T>`](/docs/reference/wiki/neuralnetworks/transformerarchitecture/) | Defines the architecture configuration for a Transformer neural network. |
| [`TransformerEmbeddingNetwork<T>`](/docs/reference/wiki/neuralnetworks/transformerembeddingnetwork/) | A customizable Transformer-based embedding network. |
| [`Transformer<T>`](/docs/reference/wiki/neuralnetworks/transformer/) | Represents a Transformer neural network architecture, which is particularly effective for sequence-based tasks like natural language processing. |
| [`TripleStreamArchitecture<T>`](/docs/reference/wiki/neuralnetworks/triplestreamarchitecture/) | A neural network architecture for three-stream models (e.g., perceiver/abstractor/resampler-style generative VLMs and cross-modality fusion encoders) that hosts a vision encoder, an auxiliary stream (perceiver / abstractor / resampler / cro… |
| [`UNet3D<T>`](/docs/reference/wiki/neuralnetworks/unet3d/) | Represents a 3D U-Net neural network for volumetric semantic segmentation. |
| [`UnifiedMultimodalNetwork<T>`](/docs/reference/wiki/neuralnetworks/unifiedmultimodalnetwork/) | Unified multimodal network that handles text, images, audio, and video in a single architecture with cross-modal attention and any-to-any generation. |
| [`VGGNetwork<T>`](/docs/reference/wiki/neuralnetworks/vggnetwork/) | Represents a VGG (Visual Geometry Group) neural network architecture for image classification. |
| [`VariationalAutoencoder<T>`](/docs/reference/wiki/neuralnetworks/variationalautoencoder/) | Represents a Variational Autoencoder (VAE) neural network architecture, which is used for  generating new data similar to the training data and learning compressed representations. |
| [`VideoCLIPNeuralNetwork<T>`](/docs/reference/wiki/neuralnetworks/videoclipneuralnetwork/) | VideoCLIP neural network for video-text alignment and temporal understanding. |
| [`VisionMambaModel<T>`](/docs/reference/wiki/neuralnetworks/visionmambamodel/) | Implements the Vision Mamba (Vim) model: PatchEmbed + scan pattern + bidirectional Mamba + classifier. |
| [`VisionTransformer<T>`](/docs/reference/wiki/neuralnetworks/visiontransformer/) | Implements the Vision Transformer (ViT) architecture for image classification tasks. |
| [`VoxelCNN<T>`](/docs/reference/wiki/neuralnetworks/voxelcnn/) | Represents a Voxel-based 3D Convolutional Neural Network for processing volumetric data. |
| [`WGANGP<T>`](/docs/reference/wiki/neuralnetworks/wgangp/) | Represents a Wasserstein GAN with Gradient Penalty (WGAN-GP), an improved version of WGAN that uses gradient penalty instead of weight clipping to enforce the Lipschitz constraint. |
| [`WGAN<T>`](/docs/reference/wiki/neuralnetworks/wgan/) | Represents a Wasserstein Generative Adversarial Network (WGAN), which uses the Wasserstein distance (Earth Mover's distance) to measure the difference between the generated and real data distributions. |
| [`Word2Vec<T>`](/docs/reference/wiki/neuralnetworks/word2vec/) | Word2Vec neural network implementation supporting both Skip-Gram and CBOW architectures. |
| [`XLSTMLanguageModel<T>`](/docs/reference/wiki/neuralnetworks/xlstmlanguagemodel/) | Implements a full xLSTM language model: token embedding + N ExtendedLSTMLayer blocks + RMS normalization + LM head. |
| [`Zamba2LanguageModel<T>`](/docs/reference/wiki/neuralnetworks/zamba2languagemodel/) | Implements a Zamba2 language model: embedding + HybridBlockScheduler (Mamba2 + shared attention with LoRA) + RMS norm + LM head. |
| [`ZambaLanguageModel<T>`](/docs/reference/wiki/neuralnetworks/zambalanguagemodel/) | Implements a Zamba language model: embedding + HybridBlockScheduler (Mamba + shared attention) + RMS norm + LM head. |

## Layers (206)

| Type | Summary |
|:-----|:--------|
| [`ABCLayer<T>`](/docs/reference/wiki/neuralnetworks/abclayer/) | Implements the ABC (Attention with Bounded-memory Control) layer from Peng et al., 2022. |
| [`ActivationLayer<T>`](/docs/reference/wiki/neuralnetworks/activationlayer/) | A layer that applies an activation function to transform the input data. |
| [`AdaptiveAveragePoolingLayer<T>`](/docs/reference/wiki/neuralnetworks/adaptiveaveragepoolinglayer/) | Implements adaptive average pooling that outputs a fixed spatial size regardless of input dimensions. |
| [`AddLayer<T>`](/docs/reference/wiki/neuralnetworks/addlayer/) | A layer that adds multiple input tensors element-wise and optionally applies an activation function. |
| [`AnomalyDetectorLayer<T>`](/docs/reference/wiki/neuralnetworks/anomalydetectorlayer/) | Represents a layer that detects anomalies by comparing predictions with actual inputs. |
| [`AttentionLayer<T>`](/docs/reference/wiki/neuralnetworks/attentionlayer/) | Represents an Attention Layer for focusing on relevant parts of input sequences. |
| [`AttentiveTransformerLayer<T>`](/docs/reference/wiki/neuralnetworks/attentivetransformerlayer/) | Implements the Attentive Transformer block used in TabNet architecture for feature selection. |
| [`AveragePoolingLayer<T>`](/docs/reference/wiki/neuralnetworks/averagepoolinglayer/) | Implements an average pooling layer for neural networks, which reduces the spatial dimensions of the input by taking the average value in each pooling window. |
| [`BASEDLayer<T>`](/docs/reference/wiki/neuralnetworks/basedlayer/) | Implements the BASED (Bidirectional Attention with Sliding-window and Expanded features) layer from "Simple Linear Attention Language Models Balance the Recall-Throughput Tradeoff" (Arora et al., 2024). |
| [`BasicBlock<T>`](/docs/reference/wiki/neuralnetworks/basicblock/) | Implements the BasicBlock used in ResNet18 and ResNet34 architectures. |
| [`BatchEnsembleLayer<T>`](/docs/reference/wiki/neuralnetworks/batchensemblelayer/) | Implements a BatchEnsemble layer that provides parameter-efficient ensembling. |
| [`BatchNormalizationLayer<T>`](/docs/reference/wiki/neuralnetworks/batchnormalizationlayer/) | Implements batch normalization for neural networks, which normalizes the inputs across a mini-batch. |
| [`BidirectionalLayer<T>`](/docs/reference/wiki/neuralnetworks/bidirectionallayer/) | Represents a bidirectional layer that processes input sequences in both forward and backward directions. |
| [`BilinearGLUFeedForwardLayer<T>`](/docs/reference/wiki/neuralnetworks/bilinearglufeedforwardlayer/) |  |
| [`BottleneckBlock<T>`](/docs/reference/wiki/neuralnetworks/bottleneckblock/) | Implements the BottleneckBlock used in ResNet50, ResNet101, and ResNet152 architectures. |
| [`CapsuleLayer<T>`](/docs/reference/wiki/neuralnetworks/capsulelayer/) | Represents a capsule neural network layer that encapsulates groups of neurons to better preserve spatial information. |
| [`CifAlignmentLayer<T>`](/docs/reference/wiki/neuralnetworks/cifalignmentlayer/) | Continuous Integrate-and-Fire (CIF) alignment layer per Gao et al. |
| [`ConcatenateLayer<T>`](/docs/reference/wiki/neuralnetworks/concatenatelayer/) | Represents a neural network layer that concatenates multiple inputs along a specified axis. |
| [`ConditionalRandomFieldLayer<T>`](/docs/reference/wiki/neuralnetworks/conditionalrandomfieldlayer/) | Represents a Conditional Random Field (CRF) layer for sequence labeling tasks. |
| [`ConstantScaleLayer<T>`](/docs/reference/wiki/neuralnetworks/constantscalelayer/) | Multiplies its input by a fixed (non-trainable) scalar. |
| [`ContinuumMemorySystemLayer<T>`](/docs/reference/wiki/neuralnetworks/continuummemorysystemlayer/) | Continuum Memory System (CMS) layer for neural networks. |
| [`Conv1DLayer<T>`](/docs/reference/wiki/neuralnetworks/conv1dlayer/) | 1D convolutional layer for sequence / waveform data, with optional dilation. |
| [`Conv1DTransposeLayer<T>`](/docs/reference/wiki/neuralnetworks/conv1dtransposelayer/) | 1D transposed convolution ("deconvolution") for sequence / waveform data — the learnable temporal-upsampling primitive used by HiFi-GAN (Kong et al. |
| [`Conv3DLayer<T>`](/docs/reference/wiki/neuralnetworks/conv3dlayer/) | Represents a 3D convolutional layer for processing volumetric data like voxel grids. |
| [`ConvLSTMLayer<T>`](/docs/reference/wiki/neuralnetworks/convlstmlayer/) | Implements a Convolutional Long Short-Term Memory (ConvLSTM) layer for processing sequential spatial data. |
| [`ConvolutionalLayer<T>`](/docs/reference/wiki/neuralnetworks/convolutionallayer/) | Represents a convolutional layer in a neural network that applies filters to input data. |
| [`CroppingLayer<T>`](/docs/reference/wiki/neuralnetworks/croppinglayer/) | Represents a cropping layer that removes portions of input tensors from the edges. |
| [`CrossAttentionLayer<T>`](/docs/reference/wiki/neuralnetworks/crossattentionlayer/) | Implements cross-attention for conditioning diffusion models on text or other context. |
| [`DecoderLayer<T>`](/docs/reference/wiki/neuralnetworks/decoderlayer/) | Represents a Decoder Layer in a Transformer architecture. |
| [`DeconvolutionalLayer<T>`](/docs/reference/wiki/neuralnetworks/deconvolutionallayer/) | Represents a deconvolutional layer (also known as transposed convolution) in a neural network. |
| [`DeformableConvolutionalLayer<T>`](/docs/reference/wiki/neuralnetworks/deformableconvolutionallayer/) | Deformable Convolutional Layer that learns spatial sampling offsets. |
| [`DeltaFormerLayer<T>`](/docs/reference/wiki/neuralnetworks/deltaformerlayer/) | Implements the DeltaFormer layer from "An Associative Memory Perspective on Transformers and DeltaNet" (Li and Papailiopoulos, 2025, arXiv:2505.19488). |
| [`DeltaNetLayer<T>`](/docs/reference/wiki/neuralnetworks/deltanetlayer/) | Implements the DeltaNet layer from "Linear Transformers with Learnable Kernel Functions" (Yang et al., 2024). |
| [`DeltaProductLayer<T>`](/docs/reference/wiki/neuralnetworks/deltaproductlayer/) | Implements the DeltaProduct layer from "DeltaProduct: Increasing the Expressivity of DeltaNet Through Products of Householders" (Siems et al., 2025). |
| [`DenseBlock<T>`](/docs/reference/wiki/neuralnetworks/denseblock/) | Implements a Dense Block from the DenseNet architecture. |
| [`DenseLayer<T>`](/docs/reference/wiki/neuralnetworks/denselayer/) | Represents a fully connected (dense) layer in a neural network. |
| [`DepthwiseSeparableConvolutionalLayer<T>`](/docs/reference/wiki/neuralnetworks/depthwiseseparableconvolutionallayer/) | Represents a depthwise separable convolutional layer that performs convolution as two separate operations. |
| [`DiffusionConvLayer<T>`](/docs/reference/wiki/neuralnetworks/diffusionconvlayer/) | Implements diffusion convolution for mesh surface processing using the heat diffusion equation. |
| [`DigitCapsuleLayer<T>`](/docs/reference/wiki/neuralnetworks/digitcapsulelayer/) | Represents a digit capsule layer that implements the dynamic routing algorithm between capsules. |
| [`DilatedConvolutionalLayer<T>`](/docs/reference/wiki/neuralnetworks/dilatedconvolutionallayer/) | Represents a dilated convolutional layer for neural networks that applies filters with gaps between filter elements. |
| [`DirectionalGraphLayer<T>`](/docs/reference/wiki/neuralnetworks/directionalgraphlayer/) | Implements Directional Graph Networks for directed graph processing with separate in/out aggregations. |
| [`DropoutLayer<T>`](/docs/reference/wiki/neuralnetworks/dropoutlayer/) | Implements a dropout layer for neural networks to prevent overfitting. |
| [`DuelingCombinationLayer<T>`](/docs/reference/wiki/neuralnetworks/duelingcombinationlayer/) | Dueling DQN combination head (Wang et al. |
| [`EdgeConditionalConvolutionalLayer<T>`](/docs/reference/wiki/neuralnetworks/edgeconditionalconvolutionallayer/) | Implements Edge-Conditioned Convolution for incorporating edge features in graph convolutions. |
| [`EmbeddingLayer<T>`](/docs/reference/wiki/neuralnetworks/embeddinglayer/) | Represents an embedding layer that converts discrete token indices into dense vector representations. |
| [`ExpertLayer<T>`](/docs/reference/wiki/neuralnetworks/expertlayer/) | Represents an expert module in a Mixture-of-Experts architecture, containing a sequence of layers. |
| [`ExtendedLSTMLayer<T>`](/docs/reference/wiki/neuralnetworks/extendedlstmlayer/) | Implements the Extended LSTM (xLSTM) layer from Hochreiter et al., 2024. |
| [`FeatureTokenizerLayer<T>`](/docs/reference/wiki/neuralnetworks/featuretokenizerlayer/) | Feature tokenizer for tabular transformers: embeds each scalar input feature into its OWN learnable embedding vector, producing a `[features, embedding]` token sequence that a transformer encoder can attend over. |
| [`FeatureTransformerLayer<T>`](/docs/reference/wiki/neuralnetworks/featuretransformerlayer/) | Implements the Feature Transformer block used in TabNet architecture. |
| [`FeedForwardLayer<T>`](/docs/reference/wiki/neuralnetworks/feedforwardlayer/) | Represents a fully connected (dense) feed-forward layer in a neural network. |
| [`FlashAttentionLayer<T>`](/docs/reference/wiki/neuralnetworks/flashattentionlayer/) | A multi-head attention layer using the Flash Attention algorithm for memory-efficient computation. |
| [`FlattenLayer<T>`](/docs/reference/wiki/neuralnetworks/flattenlayer/) | Represents a flatten layer that reshapes multi-dimensional input data into a 1D vector. |
| [`FullyConnectedLayer<T>`](/docs/reference/wiki/neuralnetworks/fullyconnectedlayer/) | Represents a fully connected layer in a neural network where every input neuron connects to every output neuron. |
| [`GRULayer<T>`](/docs/reference/wiki/neuralnetworks/grulayer/) | Represents a Gated Recurrent Unit (GRU) layer for processing sequential data. |
| [`GandalfGFLULayer<T>`](/docs/reference/wiki/neuralnetworks/gandalfgflulayer/) | GANDALF feature backbone: a stack of Gated Feature Learning Units (GFLUs), the defining component of GANDALF (Joseph & Raj 2022, "GANDALF: Gated Adaptive Network for Deep Automated Learning of Features"). |
| [`GatedDeltaNetLayer<T>`](/docs/reference/wiki/neuralnetworks/gateddeltanetlayer/) | Implements the GatedDeltaNet layer from NVIDIA, ICLR 2025. |
| [`GatedDeltaProductLayer<T>`](/docs/reference/wiki/neuralnetworks/gateddeltaproductlayer/) | Implements the Gated DeltaProduct layer from "DeltaProduct: Increasing the Expressivity of DeltaNet Through Products of Householders" (Siems et al., 2025). |
| [`GatedFeatureLearningUnitLayer<T>`](/docs/reference/wiki/neuralnetworks/gatedfeaturelearningunitlayer/) | Gated Feature Learning Unit (GFLU) for GANDALF architecture. |
| [`GatedLinearUnitLayer<T>`](/docs/reference/wiki/neuralnetworks/gatedlinearunitlayer/) | Represents a Gated Linear Unit (GLU) layer in a neural network that combines linear transformation with multiplicative gating. |
| [`GatedSlotAttentionLayer<T>`](/docs/reference/wiki/neuralnetworks/gatedslotattentionlayer/) | Implements the Gated Slot Attention (GSA) layer from Li et al., 2024. |
| [`GaussianNoiseLayer<T>`](/docs/reference/wiki/neuralnetworks/gaussiannoiselayer/) | A neural network layer that adds random Gaussian noise to inputs during training. |
| [`GeGLUFeedForwardLayer<T>`](/docs/reference/wiki/neuralnetworks/geglufeedforwardlayer/) |  |
| [`GlobalPoolingLayer<T>`](/docs/reference/wiki/neuralnetworks/globalpoolinglayer/) | Represents a global pooling layer that reduces spatial dimensions to a single value per channel. |
| [`GraphAttentionLayer<T>`](/docs/reference/wiki/neuralnetworks/graphattentionlayer/) | Implements Graph Attention Network (GAT) layer for processing graph-structured data with attention mechanisms. |
| [`GraphConvolutionalLayer<T>`](/docs/reference/wiki/neuralnetworks/graphconvolutionallayer/) | Represents a Graph Convolutional Network (GCN) layer for processing graph-structured data. |
| [`GraphIsomorphismLayer<T>`](/docs/reference/wiki/neuralnetworks/graphisomorphismlayer/) | Implements Graph Isomorphism Network (GIN) layer for powerful graph representation learning. |
| [`GraphSAGELayer<T>`](/docs/reference/wiki/neuralnetworks/graphsagelayer/) | Implements GraphSAGE (Graph Sample and Aggregate) layer for inductive learning on graphs. |
| [`GraphTransformerLayer<T>`](/docs/reference/wiki/neuralnetworks/graphtransformerlayer/) | Implements Graph Transformer layer using self-attention mechanisms on graph-structured data. |
| [`GroupNormalizationLayer<T>`](/docs/reference/wiki/neuralnetworks/groupnormalizationlayer/) | Represents a Group Normalization layer that normalizes inputs across groups of channels. |
| [`HGRN2Layer<T>`](/docs/reference/wiki/neuralnetworks/hgrn2layer/) | Implements the HGRN2 layer from "HGRN2: Gated Linear RNNs with State Expansion" (Qin et al., 2024). |
| [`HGRNLayer<T>`](/docs/reference/wiki/neuralnetworks/hgrnlayer/) | Implements the Hierarchically Gated Recurrent Neural Network (HGRN) layer from NeurIPS 2023. |
| [`HedgehogLayer<T>`](/docs/reference/wiki/neuralnetworks/hedgehoglayer/) | Implements the Hedgehog layer from "The Hedgehog and the Porcupine: Expressive Linear Attentions with Softmax Mimicry" (Zhang et al., 2024, ICLR 2024). |
| [`HeterogeneousGraphLayer<T>`](/docs/reference/wiki/neuralnetworks/heterogeneousgraphlayer/) | Implements Heterogeneous Graph Neural Network layer for graphs with multiple node and edge types. |
| [`HeterogeneousGraphMetadata`](/docs/reference/wiki/neuralnetworks/heterogeneousgraphmetadata/) | Represents metadata for heterogeneous graphs with multiple node and edge types. |
| [`HiFiGANResBlockLayer<T>`](/docs/reference/wiki/neuralnetworks/hifiganresblocklayer/) | HiFi-GAN Multi-Receptive Field (MRF) fusion module (Kong et al. |
| [`HighwayLayer<T>`](/docs/reference/wiki/neuralnetworks/highwaylayer/) | Represents a Highway Neural Network layer that allows information to flow unchanged through the network. |
| [`HybridBlockScheduler<T>`](/docs/reference/wiki/neuralnetworks/hybridblockscheduler/) | Implements a composable hybrid block that schedules SSM and attention layers according to configurable patterns used in modern hybrid architectures (Jamba, Zamba, Samba). |
| [`HyenaLayer<T>`](/docs/reference/wiki/neuralnetworks/hyenalayer/) | Implements the Hyena layer from "Hyena Hierarchy: Towards Larger Convolutional Language Models" (Poli et al., 2023, arXiv:2302.10866). |
| [`HyperbolicLinearLayer<T>`](/docs/reference/wiki/neuralnetworks/hyperboliclinearlayer/) | Represents a fully connected layer operating in hyperbolic (Poincare ball) space. |
| [`InputLayer<T>`](/docs/reference/wiki/neuralnetworks/inputlayer/) | Represents an input layer that passes input data through unchanged to the next layer in the neural network. |
| [`InstanceNormalizationLayer<T>`](/docs/reference/wiki/neuralnetworks/instancenormalizationlayer/) | Represents an Instance Normalization layer that normalizes each channel independently across spatial dimensions. |
| [`InteractingLayer<T>`](/docs/reference/wiki/neuralnetworks/interactinglayer/) | Interacting Layer for AutoInt architecture. |
| [`IntersampleAttentionLayer<T>`](/docs/reference/wiki/neuralnetworks/intersampleattentionlayer/) | Intersample (Row) Attention for SAINT architecture. |
| [`InvertedResidualBlock<T>`](/docs/reference/wiki/neuralnetworks/invertedresidualblock/) | Implements an Inverted Residual Block (MBConv) used in MobileNetV2 and MobileNetV3. |
| [`KairosMultiSizePatchLayer<T>`](/docs/reference/wiki/neuralnetworks/kairosmultisizepatchlayer/) | Kairos Mixture-of-Size patch embedder: emits N parallel patch-size paths through the SAME transformer backbone shape (numPatches varies per path but hiddenDim is fixed), then combines them via a learned router that weights each path per-inp… |
| [`KimiLinearAttentionLayer<T>`](/docs/reference/wiki/neuralnetworks/kimilinearattentionlayer/) | Implements the Kimi KDA (Key-Value Driven Gated Linear Attention) layer from the "Kimi-VL Technical Report" (Kimi Team, 2025, arXiv:2510.26692). |
| [`LSTMLayer<T>`](/docs/reference/wiki/neuralnetworks/lstmlayer/) | Represents a Long Short-Term Memory (LSTM) layer for processing sequential data. |
| [`LambdaLayer<T>`](/docs/reference/wiki/neuralnetworks/lambdalayer/) | Represents a customizable layer that applies user-defined functions for both forward and backward passes. |
| [`LayerBase<T>`](/docs/reference/wiki/neuralnetworks/layerbase/) | Represents the base class for all neural network layers, providing common functionality and interfaces. |
| [`LayerNormalizationLayer<T>`](/docs/reference/wiki/neuralnetworks/layernormalizationlayer/) | Represents a Layer Normalization layer that normalizes inputs across the feature dimension. |
| [`LayerPort`](/docs/reference/wiki/neuralnetworks/layerport/) | Declares a named input or output port on a layer. |
| [`LinearRecurrentUnitLayer<T>`](/docs/reference/wiki/neuralnetworks/linearrecurrentunitlayer/) | Implements the Linear Recurrent Unit (LRU) layer from Orvieto et al., 2023. |
| [`LocallyConnectedLayer<T>`](/docs/reference/wiki/neuralnetworks/locallyconnectedlayer/) | Represents a Locally Connected layer which applies different filters to different regions of the input, unlike a convolutional layer which shares filters. |
| [`LogLinearAttentionLayer<T>`](/docs/reference/wiki/neuralnetworks/loglinearattentionlayer/) | Implements the Log-Linear Attention layer from Zhang et al., 2025. |
| [`LogVarianceLayer<T>`](/docs/reference/wiki/neuralnetworks/logvariancelayer/) | Represents a layer that computes the logarithm of variance along a specified axis in the input tensor. |
| [`LonghornLayer<T>`](/docs/reference/wiki/neuralnetworks/longhornlayer/) | Implements the Longhorn layer from "Longhorn: State Space Models are Amortized Online Learners" (Liu et al., 2024). |
| [`MEGALayer<T>`](/docs/reference/wiki/neuralnetworks/megalayer/) | Implements the MEGA (Moving Average Equipped Gated Attention) layer from Ma et al., 2023. |
| [`MLPMixerBlockLayer<T>`](/docs/reference/wiki/neuralnetworks/mlpmixerblocklayer/) | A single MLP-Mixer block: temporal-axis MLP + channel-axis MLP, each wrapped with pre-norm and residual, per Tolstikhin et al. |
| [`Mamba2Block<T>`](/docs/reference/wiki/neuralnetworks/mamba2block/) | Implements the Mamba-2 block using the State Space Duality (SSD) framework from Dao and Gu, 2024. |
| [`MaskingLayer<T>`](/docs/reference/wiki/neuralnetworks/maskinglayer/) | Represents a layer that masks specified values in the input tensor, typically used to ignore padding in sequential data. |
| [`MaxPool3DLayer<T>`](/docs/reference/wiki/neuralnetworks/maxpool3dlayer/) | Represents a 3D max pooling layer for downsampling volumetric data. |
| [`MaxPoolingLayer<T>`](/docs/reference/wiki/neuralnetworks/maxpoolinglayer/) | Implements a max pooling layer for neural networks, which reduces the spatial dimensions of the input by taking the maximum value in each pooling window. |
| [`MeanLayer<T>`](/docs/reference/wiki/neuralnetworks/meanlayer/) | Represents a layer that computes the mean (average) of input values along a specified axis. |
| [`MeasurementLayer<T>`](/docs/reference/wiki/neuralnetworks/measurementlayer/) | Represents a layer that performs quantum measurement operations on complex-valued input tensors. |
| [`MegalodonLayer<T>`](/docs/reference/wiki/neuralnetworks/megalodonlayer/) | Implements the Megalodon layer from "Megalodon: Efficient LLM Pretraining and Inference with Unlimited Context Length" (Ma et al., 2024, arXiv:2404.08801). |
| [`MemoryReadLayer<T>`](/docs/reference/wiki/neuralnetworks/memoryreadlayer/) | Represents a layer that reads from a memory tensor using an attention mechanism. |
| [`MemoryWriteLayer<T>`](/docs/reference/wiki/neuralnetworks/memorywritelayer/) | Represents a layer that writes to a memory tensor using an attention mechanism. |
| [`MesaNetLayer<T>`](/docs/reference/wiki/neuralnetworks/mesanetlayer/) | Implements the MesaNet layer from Grazzi et al., 2025. |
| [`MeshEdgeConvLayer<T>`](/docs/reference/wiki/neuralnetworks/meshedgeconvlayer/) | Implements edge convolution for mesh-based neural networks (MeshCNN style). |
| [`MeshPoolLayer<T>`](/docs/reference/wiki/neuralnetworks/meshpoollayer/) | Implements mesh pooling via edge collapse for MeshCNN-style networks. |
| [`MessagePassingLayer<T>`](/docs/reference/wiki/neuralnetworks/messagepassinglayer/) | Implements a general Message Passing Neural Network (MPNN) layer. |
| [`MinGRULayer<T>`](/docs/reference/wiki/neuralnetworks/mingrulayer/) | Implements the minGRU layer from "Were RNNs All We Needed?" (Feng et al., 2024). |
| [`MinLSTMLayer<T>`](/docs/reference/wiki/neuralnetworks/minlstmlayer/) | Implements the minLSTM layer from "Were RNNs All We Needed?" (Feng et al., 2024). |
| [`MixtureOfExpertsBuilder<T>`](/docs/reference/wiki/neuralnetworks/mixtureofexpertsbuilder/) | A builder class that helps create and configure Mixture-of-Experts layers with sensible defaults. |
| [`MixtureOfExpertsLayer<T>`](/docs/reference/wiki/neuralnetworks/mixtureofexpertslayer/) | Implements a Mixture-of-Experts (MoE) layer that routes inputs through multiple expert networks. |
| [`MixtureOfMambaLayer<T>`](/docs/reference/wiki/neuralnetworks/mixtureofmambalayer/) | Implements the Mixture-of-Mamba layer from Jiang et al., 2025 (arXiv:2501.16295). |
| [`MixtureOfMemoriesLayer<T>`](/docs/reference/wiki/neuralnetworks/mixtureofmemorieslayer/) | Implements the Mixture of Memories (MoM) layer from Chou et al., 2025. |
| [`MultiHeadAttentionLayer<T>`](/docs/reference/wiki/neuralnetworks/multiheadattentionlayer/) | Implements a multi-head attention layer for neural networks, a key component in transformer architectures. |
| [`MultiLatentAttentionLayer<T>`](/docs/reference/wiki/neuralnetworks/multilatentattentionlayer/) | Implements the Multi-Latent Attention (MLA) layer from DeepSeek-V2 (Aixin Liu et al., 2024). |
| [`MultiplyLayer<T>`](/docs/reference/wiki/neuralnetworks/multiplylayer/) | Represents a layer that performs element-wise multiplication of multiple input tensors. |
| [`NodeEnsembleLayer<T>`](/docs/reference/wiki/neuralnetworks/nodeensemblelayer/) | NODE ensemble: a set of differentiable oblivious decision trees run in PARALLEL on the same input, with their outputs concatenated (Popov et al. |
| [`NoisyDenseLayer<T>`](/docs/reference/wiki/neuralnetworks/noisydenselayer/) | Noisy linear layer for exploration in reinforcement learning (Fortunato et al. |
| [`ObliviousDecisionTreeLayer<T>`](/docs/reference/wiki/neuralnetworks/obliviousdecisiontreelayer/) | Oblivious Decision Tree (ODT) for NODE architecture. |
| [`OctonionLinearLayer<T>`](/docs/reference/wiki/neuralnetworks/octonionlinearlayer/) | Represents a fully connected layer using octonion-valued weights and inputs. |
| [`PReLULayer<T>`](/docs/reference/wiki/neuralnetworks/prelulayer/) | Implements a Parametric ReLU (PReLU) layer with learnable negative-slope coefficients. |
| [`PaTHAttentionLayer<T>`](/docs/reference/wiki/neuralnetworks/pathattentionlayer/) | Implements the PaTH Attention (Positional-aware Transformer via Householder) layer from Mao et al., 2025 (arXiv:2505.16381). |
| [`PaddingLayer<T>`](/docs/reference/wiki/neuralnetworks/paddinglayer/) | Represents a layer that adds padding to the input tensor. |
| [`ParallelStreamsLayer<T>`](/docs/reference/wiki/neuralnetworks/parallelstreamslayer/) | Splits input along the feature axis into two equal halves, processes each half through its own independent sub-network (stream), and concatenates the two stream outputs. |
| [`PatchEmbeddingLayer<T>`](/docs/reference/wiki/neuralnetworks/patchembeddinglayer/) | Implements a patch embedding layer for Vision Transformer (ViT) architecture. |
| [`PiecewiseLinearEncodingLayer<T>`](/docs/reference/wiki/neuralnetworks/piecewiselinearencodinglayer/) | Piecewise Linear Encoding for numerical features in tabular models like TabM. |
| [`PixelShuffleLayer<T>`](/docs/reference/wiki/neuralnetworks/pixelshufflelayer/) | Pixel shuffle (sub-pixel convolution) layer for efficient spatial upsampling. |
| [`PoolingLayer<T>`](/docs/reference/wiki/neuralnetworks/poolinglayer/) | Represents a layer that performs pooling operations on input tensors. |
| [`PositionalEncodingLayer<T>`](/docs/reference/wiki/neuralnetworks/positionalencodinglayer/) | Represents a layer that adds positional encodings to input sequences. |
| [`PreLNTransformerBlock<T>`](/docs/reference/wiki/neuralnetworks/prelntransformerblock/) | Pre-Layer-Normalization transformer block with RMSNorm and a caller-supplied self-attention sublayer. |
| [`PrependCLSTokenLayer<T>`](/docs/reference/wiki/neuralnetworks/prependclstokenlayer/) | Prepends a learnable `[CLS]` token to a sequence-of-embeddings input, as introduced by BERT (Devlin et al. |
| [`PrimaryCapsuleLayer<T>`](/docs/reference/wiki/neuralnetworks/primarycapsulelayer/) | Represents a primary capsule layer for capsule networks. |
| [`PrincipalNeighbourhoodAggregationLayer<T>`](/docs/reference/wiki/neuralnetworks/principalneighbourhoodaggregationlayer/) | Implements Principal Neighbourhood Aggregation (PNA) layer for powerful graph representation learning. |
| [`PrototypeAlignmentLayer<T>`](/docs/reference/wiki/neuralnetworks/prototypealignmentlayer/) | A learned prototype-alignment layer per Sun et al. |
| [`QuantumLayer<T>`](/docs/reference/wiki/neuralnetworks/quantumlayer/) | Represents a neural network layer that uses quantum computing principles for processing inputs. |
| [`RBFLayer<T>`](/docs/reference/wiki/neuralnetworks/rbflayer/) | Represents a Radial Basis Function (RBF) layer for neural networks. |
| [`RBMLayer<T>`](/docs/reference/wiki/neuralnetworks/rbmlayer/) | Represents a Restricted Boltzmann Machine (RBM) layer for neural networks. |
| [`RMSNormalizationLayer<T>`](/docs/reference/wiki/neuralnetworks/rmsnormalizationlayer/) | Root-Mean-Square Layer Normalization (Zhang & Sennrich 2019). |
| [`RRDBLayer<T>`](/docs/reference/wiki/neuralnetworks/rrdblayer/) | Residual in Residual Dense Block (RRDB) - the core building block of ESRGAN and Real-ESRGAN generators. |
| [`RRDBNetGenerator<T>`](/docs/reference/wiki/neuralnetworks/rrdbnetgenerator/) | RRDBNet Generator - the full generator architecture from ESRGAN and Real-ESRGAN. |
| [`RWKV7Block<T>`](/docs/reference/wiki/neuralnetworks/rwkv7block/) | Implements a single RWKV-7 "Goose" block with the WKV-7 kernel featuring dynamic state evolution. |
| [`RWKVLayer<T>`](/docs/reference/wiki/neuralnetworks/rwkvlayer/) | Implements the RWKV (Receptance Weighted Key Value) layer, a linear attention RNN from Peng et al., 2024. |
| [`ReGLUFeedForwardLayer<T>`](/docs/reference/wiki/neuralnetworks/reglufeedforwardlayer/) |  |
| [`ReadoutLayer<T>`](/docs/reference/wiki/neuralnetworks/readoutlayer/) | Represents a readout layer that performs the final mapping from features to output in a neural network. |
| [`RealGatedLinearRecurrenceLayer<T>`](/docs/reference/wiki/neuralnetworks/realgatedlinearrecurrencelayer/) | Implements the Real-Gated Linear Recurrence Unit (RG-LRU) from Google DeepMind's Griffin architecture. |
| [`RebasedLayer<T>`](/docs/reference/wiki/neuralnetworks/rebasedlayer/) | Implements the ReBased linear attention layer from "Linearizing Large Language Models" (Bick et al., 2024). |
| [`ReconstructionLayer<T>`](/docs/reference/wiki/neuralnetworks/reconstructionlayer/) | Represents a reconstruction layer that uses multiple fully connected layers to transform inputs into outputs. |
| [`RecurrentLayer<T>`](/docs/reference/wiki/neuralnetworks/recurrentlayer/) | Represents a recurrent neural network layer that processes sequential data by maintaining a hidden state. |
| [`RepParameterizationLayer<T>`](/docs/reference/wiki/neuralnetworks/repparameterizationlayer/) | Represents a reparameterization layer used in variational autoencoders (VAEs) to enable backpropagation through random sampling. |
| [`ReservoirLayer<T>`](/docs/reference/wiki/neuralnetworks/reservoirlayer/) | Represents a reservoir layer used in Echo State Networks (ESNs) for processing sequential data with fixed random weights. |
| [`ReshapeLayer<T>`](/docs/reference/wiki/neuralnetworks/reshapelayer/) | Represents a reshape layer that transforms the dimensions of input data without changing its content. |
| [`ResidualDenseBlock<T>`](/docs/reference/wiki/neuralnetworks/residualdenseblock/) | Residual Dense Block (RDB) as used in ESRGAN and Real-ESRGAN generators. |
| [`ResidualLayer<T>`](/docs/reference/wiki/neuralnetworks/residuallayer/) | Represents a residual layer that adds the identity mapping (input) to the output of an inner layer. |
| [`RetNetLayer<T>`](/docs/reference/wiki/neuralnetworks/retnetlayer/) | Implements the RetNet (Retentive Network) layer from Sun et al., 2023. |
| [`RodimusLayer<T>`](/docs/reference/wiki/neuralnetworks/rodimuslayer/) | Implements the Rodimus layer from "Rodimus: Breaking the Accuracy-Efficiency Trade-Off with Efficient Attentions" (He et al., 2025). |
| [`S4DLayer<T>`](/docs/reference/wiki/neuralnetworks/s4dlayer/) | Implements a Diagonal State Space (S4D) layer from Gu et al., 2022. |
| [`S5Layer<T>`](/docs/reference/wiki/neuralnetworks/s5layer/) | Implements the Simplified State Space (S5) layer from Smith et al., 2023. |
| [`S6Scan<T>`](/docs/reference/wiki/neuralnetworks/s6scan/) | Provides reusable S6 (Selective Structured State Space Sequence) scan operations for Mamba-family architectures. |
| [`SSMQuantizationHelper<T>`](/docs/reference/wiki/neuralnetworks/ssmquantizationhelper/) | Provides SSM-specific quantization utilities for reducing memory and accelerating inference. |
| [`SSMStateCache<T>`](/docs/reference/wiki/neuralnetworks/ssmstatecache/) | Manages hidden state caching across autoregressive inference steps for SSM models. |
| [`ScanPatterns<T>`](/docs/reference/wiki/neuralnetworks/scanpatterns/) | Provides scanning pattern functions for Vision SSM architectures that process 2D spatial data. |
| [`SelfAttentionLayer<T>`](/docs/reference/wiki/neuralnetworks/selfattentionlayer/) | Represents a self-attention layer that allows a sequence to attend to itself, capturing relationships between elements. |
| [`SeparableConvolutionalLayer<T>`](/docs/reference/wiki/neuralnetworks/separableconvolutionallayer/) | Represents a separable convolutional layer that decomposes standard convolution into depthwise and pointwise operations. |
| [`SequenceLastLayer<T>`](/docs/reference/wiki/neuralnetworks/sequencelastlayer/) | A layer that extracts the last timestep from a sequence. |
| [`SequenceTokenSliceLayer<T>`](/docs/reference/wiki/neuralnetworks/sequencetokenslicelayer/) | Collapses a transformer encoder's `[batch, seq, dim]` hidden states down to `[batch, dim]` by selecting a single position (last, first, or a fixed middle index). |
| [`SoftTreeLayer<T>`](/docs/reference/wiki/neuralnetworks/softtreelayer/) | A differentiable soft decision tree layer for GANDALF and similar architectures. |
| [`SparseLinearLayer<T>`](/docs/reference/wiki/neuralnetworks/sparselinearlayer/) | Represents a fully connected layer with sparse weight matrix for efficient computation. |
| [`SpatialPoolerLayer<T>`](/docs/reference/wiki/neuralnetworks/spatialpoolerlayer/) | Represents a spatial pooler layer inspired by hierarchical temporal memory (HTM) principles. |
| [`SpatialTransformerLayer<T>`](/docs/reference/wiki/neuralnetworks/spatialtransformerlayer/) | Represents a spatial transformer layer that enables spatial manipulations of data via a learnable transformation. |
| [`SpectralNormalizationLayer<T>`](/docs/reference/wiki/neuralnetworks/spectralnormalizationlayer/) | Represents a spectral normalization layer that normalizes the weights of a layer by their spectral norm. |
| [`SpikingLayer<T>`](/docs/reference/wiki/neuralnetworks/spikinglayer/) | Represents a layer of spiking neurons that model the biological dynamics of neural activity. |
| [`SpiralConvLayer<T>`](/docs/reference/wiki/neuralnetworks/spiralconvlayer/) | Implements spiral convolution for mesh vertex processing. |
| [`SplitLayer<T>`](/docs/reference/wiki/neuralnetworks/splitlayer/) | Represents a layer that splits the input tensor along a specific dimension into multiple equal parts. |
| [`SpyNetLayer<T>`](/docs/reference/wiki/neuralnetworks/spynetlayer/) | SPyNet (Spatial Pyramid Network) layer for optical flow estimation. |
| [`SqueezeAndExcitationLayer<T>`](/docs/reference/wiki/neuralnetworks/squeezeandexcitationlayer/) | Represents a Squeeze-and-Excitation layer that recalibrates channel-wise feature responses adaptively. |
| [`SubpixelConvolutionalLayer<T>`](/docs/reference/wiki/neuralnetworks/subpixelconvolutionallayer/) | Represents a subpixel convolutional layer that performs convolution followed by pixel shuffling for upsampling. |
| [`SwiGLUFeedForwardLayer<T>`](/docs/reference/wiki/neuralnetworks/swiglufeedforwardlayer/) |  |
| [`SwinPatchEmbeddingLayer<T>`](/docs/reference/wiki/neuralnetworks/swinpatchembeddinglayer/) | Patch embedding layer for Swin Transformer that converts images to patch sequences. |
| [`SwinPatchMergingLayer<T>`](/docs/reference/wiki/neuralnetworks/swinpatchmerginglayer/) | Patch merging layer for Swin Transformer that performs downsampling between stages. |
| [`SwinTransformerBlockLayer<T>`](/docs/reference/wiki/neuralnetworks/swintransformerblocklayer/) | Swin Transformer block layer with windowed multi-head self-attention. |
| [`SynapticPlasticityLayer<T>`](/docs/reference/wiki/neuralnetworks/synapticplasticitylayer/) | Represents a synaptic plasticity layer that models biological learning mechanisms through spike-timing-dependent plasticity. |
| [`T5RelativeBiasAttentionLayer<T>`](/docs/reference/wiki/neuralnetworks/t5relativebiasattentionlayer/) | T5-style multi-head self-attention with learned relative position bias (Raffel et al., "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer", JMLR 2020). |
| [`TTTLayer<T>`](/docs/reference/wiki/neuralnetworks/tttlayer/) | Implements the TTT (Test-Time Training) layer from Sun et al., 2024. |
| [`TabMEnsembleLayer<T>`](/docs/reference/wiki/neuralnetworks/tabmensemblelayer/) | TabM ensemble MLP (Gorishniy et al. |
| [`TabNetEncoderLayer<T>`](/docs/reference/wiki/neuralnetworks/tabnetencoderlayer/) | TabNet encoder (Arik & Pfister 2019): a sequential-attention block that produces an aggregated decision representation through several decision steps, each of which selects a sparse subset of features via a learnable mask. |
| [`TemporalMemoryLayer<T>`](/docs/reference/wiki/neuralnetworks/temporalmemorylayer/) | Represents a temporal memory layer that models sequence learning through hierarchical temporal memory concepts. |
| [`TimeDistributedLayer<T>`](/docs/reference/wiki/neuralnetworks/timedistributedlayer/) | Represents a wrapper layer that applies an inner layer to each time step of a sequence independently. |
| [`TimeEmbeddingLayer<T>`](/docs/reference/wiki/neuralnetworks/timeembeddinglayer/) | Represents a time embedding layer that encodes timesteps using sinusoidal embeddings for diffusion models. |
| [`TimeMoEBlockLayer<T>`](/docs/reference/wiki/neuralnetworks/timemoeblocklayer/) | A single Time-MoE transformer block: multi-head self-attention + Mixture-of-Experts FFN, each wrapped in pre-norm + residual. |
| [`TimeSformerBlockLayer<T>`](/docs/reference/wiki/neuralnetworks/timesformerblocklayer/) | TimeSformer encoder block with divided space-time attention. |
| [`TransNormerLLMLayer<T>`](/docs/reference/wiki/neuralnetworks/transnormerllmlayer/) | Implements the TransNormerLLM layer from "TransNormerLLM: A Faster and Better LLM" (Qin et al., 2023). |
| [`TransformerDecoderBlock<T>`](/docs/reference/wiki/neuralnetworks/transformerdecoderblock/) | Pre-Layer-Normalization transformer decoder block — self-attention, a second ("cross") attention, and a position-wise feed-forward network, each wrapped in a residual (skip) connection with layer normalization applied BEFORE the sublayer. |
| [`TransformerDecoderLayer<T>`](/docs/reference/wiki/neuralnetworks/transformerdecoderlayer/) | Represents a transformer decoder layer that processes sequences using self-attention, cross-attention, and feed-forward networks. |
| [`TransformerEncoderBlock<T>`](/docs/reference/wiki/neuralnetworks/transformerencoderblock/) | Pre-Layer-Normalization transformer encoder block — multi-head self-attention and a position-wise feed-forward network, each wrapped in a residual (skip) connection with layer normalization applied BEFORE the sublayer (Pre-LN). |
| [`TransformerEncoderLayer<T>`](/docs/reference/wiki/neuralnetworks/transformerencoderlayer/) | Represents a transformer encoder layer that processes sequences using self-attention and feed-forward networks. |
| [`TransitionLayer<T>`](/docs/reference/wiki/neuralnetworks/transitionlayer/) | Implements a Transition Layer from the DenseNet architecture. |
| [`TransposeLayer<T>`](/docs/reference/wiki/neuralnetworks/transposelayer/) | Reorders the axes of the input tensor according to a fixed permutation. |
| [`UNetDiscriminator<T>`](/docs/reference/wiki/neuralnetworks/unetdiscriminator/) | U-Net Discriminator as used in Real-ESRGAN for improved perceptual quality. |
| [`Upsample3DLayer<T>`](/docs/reference/wiki/neuralnetworks/upsample3dlayer/) | Represents a 3D upsampling layer that increases the spatial dimensions of volumetric data using nearest-neighbor interpolation. |
| [`UpsamplingLayer<T>`](/docs/reference/wiki/neuralnetworks/upsamplinglayer/) | Represents an upsampling layer that increases the spatial dimensions of input tensors using nearest-neighbor interpolation. |
| [`WaveNetResidualBlockLayer<T>`](/docs/reference/wiki/neuralnetworks/wavenetresidualblocklayer/) | A single WaveNet / Parallel WaveGAN residual block (van den Oord et al. |
| [`WordCharEmbeddingLayer<T>`](/docs/reference/wiki/neuralnetworks/wordcharembeddinglayer/) | Paper-faithful word + character embedding front-end for sequence-labeling NER models (Lample et al., NAACL 2016, "Neural Architectures for Named Entity Recognition", §3). |

## Base Classes (15)

| Type | Summary |
|:-----|:--------|
| [`AutoIntBase<T>`](/docs/reference/wiki/neuralnetworks/autointbase/) | Base class for AutoInt (Automatic Feature Interaction Learning). |
| [`DualEncoderArchitecture<T>`](/docs/reference/wiki/neuralnetworks/dualencoderarchitecture/) | Modality-agnostic base for two-encoder neural network architectures. |
| [`FTTransformerBase<T>`](/docs/reference/wiki/neuralnetworks/fttransformerbase/) | Base implementation of FT-Transformer (Feature Tokenizer + Transformer) for tabular data. |
| [`GANDALFBase<T>`](/docs/reference/wiki/neuralnetworks/gandalfbase/) | Base implementation of GANDALF (Gated Additive Neural Decision Forest). |
| [`MambularBase<T>`](/docs/reference/wiki/neuralnetworks/mambularbase/) | Base class for Mambular (State Space Models for Tabular Data). |
| [`NODEBase<T>`](/docs/reference/wiki/neuralnetworks/nodebase/) | Base class for NODE (Neural Oblivious Decision Ensembles). |
| [`NeuralNetworkBase<T>`](/docs/reference/wiki/neuralnetworks/neuralnetworkbase/) | Base class for all neural network implementations in AiDotNet. |
| [`SAINTBase<T>`](/docs/reference/wiki/neuralnetworks/saintbase/) | Base class for SAINT (Self-Attention and Intersample Attention Transformer). |
| [`SyntheticTabularGeneratorBase<T>`](/docs/reference/wiki/neuralnetworks/synthetictabulargeneratorbase/) | Abstract base class for synthetic tabular data generators, providing common infrastructure for fitting models on real data and generating synthetic rows. |
| [`TabDPTBase<T>`](/docs/reference/wiki/neuralnetworks/tabdptbase/) | Base class for TabDPT (Tabular Data Pre-Training) foundation model. |
| [`TabMBase<T>`](/docs/reference/wiki/neuralnetworks/tabmbase/) | Base implementation of TabM, a parameter-efficient ensemble model for tabular data. |
| [`TabNetBase<T>`](/docs/reference/wiki/neuralnetworks/tabnetbase/) | Base implementation of the TabNet architecture for attentive interpretable tabular learning. |
| [`TabPFNBase<T>`](/docs/reference/wiki/neuralnetworks/tabpfnbase/) | Base class for TabPFN (Prior-Fitted Networks) for tabular data. |
| [`TabRBase<T>`](/docs/reference/wiki/neuralnetworks/tabrbase/) | Base implementation of TabR, a retrieval-augmented model for tabular data. |
| [`TabTransformerBase<T>`](/docs/reference/wiki/neuralnetworks/tabtransformerbase/) | Base implementation of TabTransformer for tabular data. |

## Enums (12)

| Type | Summary |
|:-----|:--------|
| [`ColumnDataType`](/docs/reference/wiki/neuralnetworks/columndatatype/) | Specifies the data type of a column in a tabular dataset. |
| [`EmbeddingInputMode`](/docs/reference/wiki/neuralnetworks/embeddinginputmode/) |  |
| [`FlashAttentionPrecision`](/docs/reference/wiki/neuralnetworks/flashattentionprecision/) | Precision modes for Flash Attention computation. |
| [`GLUGateType`](/docs/reference/wiki/neuralnetworks/glugatetype/) |  |
| [`GraphPooling<T>`](/docs/reference/wiki/neuralnetworks/graphpooling/) | Graph pooling methods for aggregating node embeddings. |
| [`HybridSchedulePattern`](/docs/reference/wiki/neuralnetworks/hybridschedulepattern/) | Defines the scheduling pattern for hybrid SSM/attention architectures. |
| [`LinkPredictionDecoder<T>`](/docs/reference/wiki/neuralnetworks/linkpredictiondecoder/) | Decoder types for combining node embeddings into edge scores. |
| [`PoolingStrategy<T>`](/docs/reference/wiki/neuralnetworks/poolingstrategy/) | Defines the available pooling strategies for creating a single sentence embedding. |
| [`Position<T>`](/docs/reference/wiki/neuralnetworks/position/) | Which sequence position the slice selects. |
| [`SpatialTransformerDataFormat`](/docs/reference/wiki/neuralnetworks/spatialtransformerdataformat/) |  |
| [`SymbolicShapeMode`](/docs/reference/wiki/neuralnetworks/symbolicshapemode/) | Strategy for how `CompiledModelHost` keys the compile cache. |
| [`VisionScanPattern`](/docs/reference/wiki/neuralnetworks/visionscanpattern/) | Defines the scan pattern used by the Vision Mamba model to convert 2D patch grids into 1D sequences. |

## Delegates (3)

| Type | Summary |
|:-----|:--------|
| [`AggregationFunction<T>`](/docs/reference/wiki/neuralnetworks/aggregationfunction/) | Defines the aggregation function type for combining messages. |
| [`MessageFunction<T>`](/docs/reference/wiki/neuralnetworks/messagefunction/) | Defines the message function type for message passing neural networks. |
| [`UpdateFunction<T>`](/docs/reference/wiki/neuralnetworks/updatefunction/) | Defines the update function type for updating node features. |

## Options & Configuration (110)

| Type | Summary |
|:-----|:--------|
| [`ACGANOptions`](/docs/reference/wiki/neuralnetworks/acganoptions/) | Configuration options for the ACGAN neural network. |
| [`AttentionNetworkOptions`](/docs/reference/wiki/neuralnetworks/attentionnetworkoptions/) | Configuration options for the AttentionNetwork. |
| [`AudioVisualCorrespondenceOptions`](/docs/reference/wiki/neuralnetworks/audiovisualcorrespondenceoptions/) | Configuration options for the AudioVisualCorrespondenceNetwork. |
| [`AudioVisualEventLocalizationOptions`](/docs/reference/wiki/neuralnetworks/audiovisualeventlocalizationoptions/) | Configuration options for the AudioVisualEventLocalizationNetwork. |
| [`AutoencoderOptions`](/docs/reference/wiki/neuralnetworks/autoencoderoptions/) | Configuration options for the Autoencoder neural network. |
| [`BGEOptions`](/docs/reference/wiki/neuralnetworks/bgeoptions/) | Configuration options for the BGE (BAAI General Embedding) model. |
| [`BigGANOptions`](/docs/reference/wiki/neuralnetworks/bigganoptions/) | Configuration options for the BigGAN neural network. |
| [`Blip2Options`](/docs/reference/wiki/neuralnetworks/blip2options/) | Configuration options for the Blip2NeuralNetwork. |
| [`BlipOptions`](/docs/reference/wiki/neuralnetworks/blipoptions/) | Configuration options for the BlipNeuralNetwork. |
| [`CapsuleNetworkOptions`](/docs/reference/wiki/neuralnetworks/capsulenetworkoptions/) | Configuration options for the CapsuleNetwork. |
| [`ClipModelConfig`](/docs/reference/wiki/neuralnetworks/clipmodelconfig/) | Configuration for a CLIP model variant. |
| [`ClipOptions`](/docs/reference/wiki/neuralnetworks/clipoptions/) | Configuration options for the ClipNeuralNetwork. |
| [`ColBERTOptions`](/docs/reference/wiki/neuralnetworks/colbertoptions/) | Configuration options for the ColBERT model. |
| [`ConditionalGANOptions`](/docs/reference/wiki/neuralnetworks/conditionalganoptions/) | Configuration options for the Conditional GAN model. |
| [`ConvolutionalNeuralNetworkOptions`](/docs/reference/wiki/neuralnetworks/convolutionalneuralnetworkoptions/) | Configuration options for the ConvolutionalNeuralNetwork. |
| [`CycleGANOptions`](/docs/reference/wiki/neuralnetworks/cycleganoptions/) | Configuration options for the CycleGAN neural network. |
| [`DCGANOptions`](/docs/reference/wiki/neuralnetworks/dcganoptions/) | Configuration options for the DCGAN (Deep Convolutional GAN) model. |
| [`DeepBeliefNetworkOptions`](/docs/reference/wiki/neuralnetworks/deepbeliefnetworkoptions/) | Configuration options for the DeepBeliefNetwork. |
| [`DeepBoltzmannMachineOptions`](/docs/reference/wiki/neuralnetworks/deepboltzmannmachineoptions/) | Configuration options for the DeepBoltzmannMachine. |
| [`DeepQNetworkOptions`](/docs/reference/wiki/neuralnetworks/deepqnetworkoptions/) | Configuration options for the DeepQNetwork. |
| [`DenseNetOptions`](/docs/reference/wiki/neuralnetworks/densenetoptions/) | Configuration options for the DenseNetNetwork. |
| [`DifferentiableNeuralComputerOptions`](/docs/reference/wiki/neuralnetworks/differentiableneuralcomputeroptions/) | Configuration options for the DifferentiableNeuralComputer. |
| [`EagleOptions`](/docs/reference/wiki/neuralnetworks/eagleoptions/) | Configuration options for the EagleLanguageModel. |
| [`EchoStateNetworkOptions`](/docs/reference/wiki/neuralnetworks/echostatenetworkoptions/) | Configuration options for the EchoStateNetwork. |
| [`EfficientNetOptions`](/docs/reference/wiki/neuralnetworks/efficientnetoptions/) | Configuration options for the EfficientNetNetwork. |
| [`ExtremeLearningMachineOptions`](/docs/reference/wiki/neuralnetworks/extremelearningmachineoptions/) | Configuration options for the ExtremeLearningMachine. |
| [`FalconMambaOptions`](/docs/reference/wiki/neuralnetworks/falconmambaoptions/) | Configuration options for the FalconMambaLanguageModel. |
| [`FastTextOptions`](/docs/reference/wiki/neuralnetworks/fasttextoptions/) | Configuration options for the FastText model. |
| [`FeedForwardNeuralNetworkOptions`](/docs/reference/wiki/neuralnetworks/feedforwardneuralnetworkoptions/) | Configuration options for the FeedForwardNeuralNetwork. |
| [`FinchOptions`](/docs/reference/wiki/neuralnetworks/finchoptions/) | Configuration options for the FinchLanguageModel. |
| [`FlamingoOptions`](/docs/reference/wiki/neuralnetworks/flamingooptions/) | Configuration options for the FlamingoNeuralNetwork. |
| [`FlashAttentionConfig`](/docs/reference/wiki/neuralnetworks/flashattentionconfig/) | Configuration options for Flash Attention algorithm. |
| [`GLAOptions`](/docs/reference/wiki/neuralnetworks/glaoptions/) | Configuration options for the GLALanguageModel. |
| [`GRUOptions`](/docs/reference/wiki/neuralnetworks/gruoptions/) | Configuration options for the GRUNeuralNetwork. |
| [`GatedDeltaNetOptions`](/docs/reference/wiki/neuralnetworks/gateddeltanetoptions/) | Configuration options for the GatedDeltaNetLanguageModel. |
| [`GenerativeAdversarialNetworkOptions`](/docs/reference/wiki/neuralnetworks/generativeadversarialnetworkoptions/) | Configuration options for the GenerativeAdversarialNetwork. |
| [`GloVeOptions`](/docs/reference/wiki/neuralnetworks/gloveoptions/) | Configuration options for the GloVe model. |
| [`Gpt4VisionOptions`](/docs/reference/wiki/neuralnetworks/gpt4visionoptions/) | Configuration options for the Gpt4VisionNeuralNetwork. |
| [`GraphAttentionNetworkOptions`](/docs/reference/wiki/neuralnetworks/graphattentionnetworkoptions/) | Configuration options for the GraphAttentionNetwork. |
| [`GraphClassificationModelOptions`](/docs/reference/wiki/neuralnetworks/graphclassificationmodeloptions/) | Configuration options for the GraphClassificationModel. |
| [`GraphGenerationModelOptions`](/docs/reference/wiki/neuralnetworks/graphgenerationmodeloptions/) | Configuration options for the GraphGenerationModel. |
| [`GraphIsomorphismNetworkOptions`](/docs/reference/wiki/neuralnetworks/graphisomorphismnetworkoptions/) | Configuration options for the GraphIsomorphismNetwork. |
| [`GraphNeuralNetworkOptions`](/docs/reference/wiki/neuralnetworks/graphneuralnetworkoptions/) | Configuration options for the GraphNeuralNetwork. |
| [`GraphSAGEOptions`](/docs/reference/wiki/neuralnetworks/graphsageoptions/) | Configuration options for the GraphSAGENetwork. |
| [`GriffinOptions`](/docs/reference/wiki/neuralnetworks/griffinoptions/) | Configuration options for the GriffinLanguageModel. |
| [`HTMNetworkOptions`](/docs/reference/wiki/neuralnetworks/htmnetworkoptions/) | Configuration options for the HTMNetwork. |
| [`HawkOptions`](/docs/reference/wiki/neuralnetworks/hawkoptions/) | Configuration options for the HawkLanguageModel. |
| [`HopeNetworkOptions`](/docs/reference/wiki/neuralnetworks/hopenetworkoptions/) | Configuration options for the HopeNetwork. |
| [`HopfieldNetworkOptions`](/docs/reference/wiki/neuralnetworks/hopfieldnetworkoptions/) | Configuration options for the HopfieldNetwork. |
| [`HyperbolicNeuralNetworkOptions`](/docs/reference/wiki/neuralnetworks/hyperbolicneuralnetworkoptions/) | Configuration options for the HyperbolicNeuralNetwork. |
| [`ImageBindOptions`](/docs/reference/wiki/neuralnetworks/imagebindoptions/) | Configuration options for the ImageBindNeuralNetwork. |
| [`InferenceArenaSettings`](/docs/reference/wiki/neuralnetworks/inferencearenasettings/) | Global opt-in switch for the inference forward-caching allocator (#1661 / Tensors #661). |
| [`InfoGANOptions`](/docs/reference/wiki/neuralnetworks/infoganoptions/) | Configuration options for the InfoGAN neural network. |
| [`InstructorEmbeddingOptions`](/docs/reference/wiki/neuralnetworks/instructorembeddingoptions/) | Configuration options for the Instructor Embedding model. |
| [`JambaOptions`](/docs/reference/wiki/neuralnetworks/jambaoptions/) | Configuration options for the JambaLanguageModel. |
| [`LLaVAOptions`](/docs/reference/wiki/neuralnetworks/llavaoptions/) | Configuration options for the LLaVANeuralNetwork. |
| [`LSTMOptions`](/docs/reference/wiki/neuralnetworks/lstmoptions/) | Configuration options for the LSTMNeuralNetwork. |
| [`LinkPredictionModelOptions`](/docs/reference/wiki/neuralnetworks/linkpredictionmodeloptions/) | Configuration options for the LinkPredictionModel. |
| [`LiquidStateMachineOptions`](/docs/reference/wiki/neuralnetworks/liquidstatemachineoptions/) | Configuration options for the LiquidStateMachine. |
| [`Mamba2Options`](/docs/reference/wiki/neuralnetworks/mamba2options/) | Configuration options for the Mamba2LanguageModel. |
| [`MambaOptions`](/docs/reference/wiki/neuralnetworks/mambaoptions/) | Configuration options for the MambaLanguageModel. |
| [`MatryoshkaEmbeddingOptions`](/docs/reference/wiki/neuralnetworks/matryoshkaembeddingoptions/) | Configuration options for the Matryoshka Embedding model. |
| [`MemoryNetworkOptions`](/docs/reference/wiki/neuralnetworks/memorynetworkoptions/) | Configuration options for the MemoryNetwork. |
| [`MobileNetV2Options`](/docs/reference/wiki/neuralnetworks/mobilenetv2options/) | Configuration options for the MobileNetV2Network. |
| [`MobileNetV3Options`](/docs/reference/wiki/neuralnetworks/mobilenetv3options/) | Configuration options for the MobileNetV3Network. |
| [`NEATOptions`](/docs/reference/wiki/neuralnetworks/neatoptions/) | Configuration options for the NEAT neural network. |
| [`NeuralNetworkDefaultOptions`](/docs/reference/wiki/neuralnetworks/neuralnetworkdefaultoptions/) | Configuration options for the NeuralNetwork. |
| [`NeuralTuringMachineOptions`](/docs/reference/wiki/neuralnetworks/neuralturingmachineoptions/) | Configuration options for the NeuralTuringMachine. |
| [`NodeClassificationModelOptions`](/docs/reference/wiki/neuralnetworks/nodeclassificationmodeloptions/) | Configuration options for the NodeClassificationModel. |
| [`OccupancyNeuralNetworkOptions`](/docs/reference/wiki/neuralnetworks/occupancyneuralnetworkoptions/) | Configuration options for the OccupancyNeuralNetwork. |
| [`OctonionNeuralNetworkOptions`](/docs/reference/wiki/neuralnetworks/octonionneuralnetworkoptions/) | Configuration options for the OctonionNeuralNetwork. |
| [`Pix2PixOptions`](/docs/reference/wiki/neuralnetworks/pix2pixoptions/) | Configuration options for the Pix2Pix neural network. |
| [`ProgressiveGANOptions`](/docs/reference/wiki/neuralnetworks/progressiveganoptions/) | Configuration options for the ProgressiveGAN. |
| [`QuantumNeuralNetworkOptions`](/docs/reference/wiki/neuralnetworks/quantumneuralnetworkoptions/) | Configuration options for the QuantumNeuralNetwork. |
| [`RWKV4Options`](/docs/reference/wiki/neuralnetworks/rwkv4options/) | Configuration options for the RWKV4LanguageModel. |
| [`RWKV7Options`](/docs/reference/wiki/neuralnetworks/rwkv7options/) | Configuration options for the RWKV7LanguageModel. |
| [`RadialBasisFunctionNetworkOptions`](/docs/reference/wiki/neuralnetworks/radialbasisfunctionnetworkoptions/) | Configuration options for the RadialBasisFunctionNetwork. |
| [`RecurrentGemmaOptions`](/docs/reference/wiki/neuralnetworks/recurrentgemmaoptions/) | Configuration options for the RecurrentGemmaLanguageModel. |
| [`RecurrentNeuralNetworkOptions`](/docs/reference/wiki/neuralnetworks/recurrentneuralnetworkoptions/) | Configuration options for the RecurrentNeuralNetwork. |
| [`ResNetOptions`](/docs/reference/wiki/neuralnetworks/resnetoptions/) | Configuration options for the ResNetNetwork. |
| [`ResidualNeuralNetworkOptions`](/docs/reference/wiki/neuralnetworks/residualneuralnetworkoptions/) | Configuration options for the ResidualNeuralNetwork. |
| [`RestrictedBoltzmannMachineOptions`](/docs/reference/wiki/neuralnetworks/restrictedboltzmannmachineoptions/) | Configuration options for the RestrictedBoltzmannMachine. |
| [`SAGANOptions`](/docs/reference/wiki/neuralnetworks/saganoptions/) | Configuration options for the SAGAN neural network. |
| [`SGPTOptions`](/docs/reference/wiki/neuralnetworks/sgptoptions/) | Configuration options for the SGPT model. |
| [`SPLADEOptions`](/docs/reference/wiki/neuralnetworks/spladeoptions/) | Configuration options for the SPLADE model. |
| [`SambaOptions`](/docs/reference/wiki/neuralnetworks/sambaoptions/) | Configuration options for the SambaLanguageModel. |
| [`SamplingOptions`](/docs/reference/wiki/neuralnetworks/samplingoptions/) | Decoding/sampling configuration for autoregressive text generation (#1632 / #95). |
| [`SelfOrganizingMapNNOptions`](/docs/reference/wiki/neuralnetworks/selforganizingmapnnoptions/) | Configuration options for the SelfOrganizingMap neural network. |
| [`SiameseNetworkOptions`](/docs/reference/wiki/neuralnetworks/siamesenetworkoptions/) | Configuration options for the SiameseNetwork. |
| [`SiameseNeuralNetworkOptions`](/docs/reference/wiki/neuralnetworks/siameseneuralnetworkoptions/) | Configuration options for the SiameseNeuralNetwork. |
| [`SimCSEOptions`](/docs/reference/wiki/neuralnetworks/simcseoptions/) | Configuration options for the SimCSE model. |
| [`SparseNeuralNetworkOptions`](/docs/reference/wiki/neuralnetworks/sparseneuralnetworkoptions/) | Configuration options for the SparseNeuralNetwork. |
| [`SpikingNeuralNetworkOptions`](/docs/reference/wiki/neuralnetworks/spikingneuralnetworkoptions/) | Configuration options for the SpikingNeuralNetwork. |
| [`StyleGANOptions`](/docs/reference/wiki/neuralnetworks/styleganoptions/) | Configuration options for the StyleGAN neural network. |
| [`TransformerEmbeddingOptions`](/docs/reference/wiki/neuralnetworks/transformerembeddingoptions/) | Configuration options for the TransformerEmbeddingNetwork. |
| [`TransformerOptions`](/docs/reference/wiki/neuralnetworks/transformeroptions/) | Configuration options for the Transformer neural network. |
| [`UNet3DOptions`](/docs/reference/wiki/neuralnetworks/unet3doptions/) | Configuration options for the UNet3D neural network. |
| [`UnifiedMultimodalNetworkOptions`](/docs/reference/wiki/neuralnetworks/unifiedmultimodalnetworkoptions/) | Configuration options for the UnifiedMultimodalNetwork. |
| [`VGGOptions`](/docs/reference/wiki/neuralnetworks/vggoptions/) | Configuration options for the VGGNetwork. |
| [`VariationalAutoencoderOptions`](/docs/reference/wiki/neuralnetworks/variationalautoencoderoptions/) | Configuration options for the VariationalAutoencoder. |
| [`VideoCLIPOptions`](/docs/reference/wiki/neuralnetworks/videoclipoptions/) | Configuration options for the VideoCLIPNeuralNetwork. |
| [`VisionMambaOptions`](/docs/reference/wiki/neuralnetworks/visionmambaoptions/) | Configuration options for the VisionMambaLanguageModel. |
| [`VisionTransformerOptions`](/docs/reference/wiki/neuralnetworks/visiontransformeroptions/) | Configuration options for the VisionTransformer. |
| [`VoxelCNNOptions`](/docs/reference/wiki/neuralnetworks/voxelcnnoptions/) | Configuration options for the VoxelCNN neural network. |
| [`WGANGPOptions`](/docs/reference/wiki/neuralnetworks/wgangpoptions/) | Configuration options for the WGANGP neural network. |
| [`WGANOptions`](/docs/reference/wiki/neuralnetworks/wganoptions/) | Configuration options for the WGAN neural network. |
| [`Word2VecOptions`](/docs/reference/wiki/neuralnetworks/word2vecoptions/) | Configuration options for the Word2Vec model. |
| [`XLSTMOptions`](/docs/reference/wiki/neuralnetworks/xlstmoptions/) | Configuration options for the XLSTMLanguageModel. |
| [`Zamba2Options`](/docs/reference/wiki/neuralnetworks/zamba2options/) | Configuration options for the Zamba2LanguageModel. |
| [`ZambaOptions`](/docs/reference/wiki/neuralnetworks/zambaoptions/) | Configuration options for the ZambaLanguageModel. |

## Helpers & Utilities (3)

| Type | Summary |
|:-----|:--------|
| [`AutoregressiveDecoder<T>`](/docs/reference/wiki/neuralnetworks/autoregressivedecoder/) | Generic autoregressive decode loop (#1632 / #95): the reusable "generate" driver the codebase lacked — GPT4Vision / Blip / Flamingo each hand-rolled this loop. |
| [`ClipModelLoader`](/docs/reference/wiki/neuralnetworks/clipmodelloader/) | Loads CLIP models from HuggingFace Hub or local directories. |
| [`SegmentationMetrics<T>`](/docs/reference/wiki/neuralnetworks/segmentationmetrics/) | Provides metrics for evaluating segmentation and classification tasks. |

