namespace AiDotNet.Enums;

/// <summary>
/// Defines the algorithm family or category that a machine learning model belongs to.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This tells you what kind of algorithm a model uses under the hood.
/// For example, a neural network learns by adjusting many interconnected weights, while
/// a decision tree learns by splitting data into branches based on feature values.
/// Understanding the category helps you choose the right tool for your problem.
/// </para>
/// </remarks>
public enum ModelCategory
{
    /// <summary>
    /// A feedforward or general-purpose neural network.
    /// Learns patterns through layers of interconnected neurons with adjustable weights.
    /// </summary>
    NeuralNetwork,

    /// <summary>
    /// A regression model that predicts continuous numeric values.
    /// Examples: linear regression, polynomial regression, ridge regression.
    /// </summary>
    Regression,

    /// <summary>
    /// A classification model that assigns inputs to discrete categories.
    /// Examples: logistic regression, naive Bayes, SVM, decision tree classifiers.
    /// </summary>
    Classifier,

    /// <summary>
    /// A clustering model that groups similar data points together without labels.
    /// Examples: K-Means, DBSCAN, hierarchical clustering.
    /// </summary>
    Clustering,

    /// <summary>
    /// A Generative Adversarial Network that learns by training a generator against a discriminator.
    /// Used for generating realistic images, data augmentation, and style transfer.
    /// </summary>
    GAN,

    /// <summary>
    /// A diffusion model that generates data by learning to reverse a noise-adding process.
    /// Achieves state-of-the-art quality in image, audio, and video generation.
    /// </summary>
    Diffusion,

    /// <summary>
    /// A transformer model based on self-attention mechanisms.
    /// The foundation of modern language models and increasingly used in vision and audio.
    /// </summary>
    Transformer,

    /// <summary>
    /// A reinforcement learning agent that learns through trial-and-error interaction with an environment.
    /// Examples: DQN, PPO, A2C, SAC agents.
    /// </summary>
    ReinforcementLearningAgent,

    /// <summary>
    /// A Gaussian process model that provides probabilistic predictions with uncertainty estimates.
    /// Non-parametric Bayesian approach ideal for small datasets and active learning.
    /// </summary>
    GaussianProcess,

    /// <summary>
    /// An ensemble model that combines multiple base models for better predictions.
    /// Examples: random forests, boosting, bagging, stacking.
    /// </summary>
    Ensemble,

    /// <summary>
    /// A Bayesian model that incorporates prior knowledge and provides posterior distributions.
    /// Naturally handles uncertainty and works well with limited data.
    /// </summary>
    Bayesian,

    /// <summary>
    /// A survival analysis model that predicts time-to-event outcomes.
    /// Examples: Cox proportional hazards, Kaplan-Meier, accelerated failure time models.
    /// </summary>
    SurvivalModel,

    /// <summary>
    /// A causal inference model that estimates cause-and-effect relationships.
    /// Examples: treatment effect estimators, causal forests, propensity score methods.
    /// </summary>
    CausalModel,

    /// <summary>
    /// A time series model specialized for sequential, time-dependent data.
    /// Examples: ARIMA, exponential smoothing, temporal convolutional networks.
    /// </summary>
    TimeSeriesModel,

    /// <summary>
    /// An autoencoder that learns compressed representations by encoding and decoding data.
    /// Used for dimensionality reduction, anomaly detection, and feature learning.
    /// </summary>
    Autoencoder,

    /// <summary>
    /// A recurrent neural network that processes sequential data by maintaining hidden state.
    /// Examples: LSTM, GRU, bidirectional RNNs.
    /// </summary>
    RecurrentNetwork,

    /// <summary>
    /// A convolutional neural network that learns spatial patterns using learnable filters.
    /// The standard architecture for image and signal processing tasks.
    /// </summary>
    ConvolutionalNetwork,

    /// <summary>
    /// A graph neural network that operates on graph-structured data.
    /// Learns node, edge, and graph-level representations through message passing.
    /// </summary>
    GraphNetwork,

    /// <summary>
    /// A model that produces dense vector representations (embeddings) of inputs.
    /// Used for similarity search, retrieval, and as features for downstream tasks.
    /// </summary>
    EmbeddingModel,

    /// <summary>
    /// A large-scale foundation model pretrained on massive datasets.
    /// Can be fine-tuned or used zero-shot for many downstream tasks.
    /// </summary>
    FoundationModel,

    /// <summary>
    /// A meta-learning model that learns how to learn from few examples.
    /// Examples: MAML, prototypical networks, matching networks.
    /// </summary>
    MetaLearning,

    /// <summary>
    /// A model specifically optimized for structured tabular data.
    /// Examples: TabNet, SAINT, FT-Transformer, NODE.
    /// </summary>
    TabularModel,

    /// <summary>
    /// A model that generates synthetic data mimicking real dataset properties.
    /// Examples: CTGAN, CopulaGAN, Bayesian network synthesizers.
    /// </summary>
    SyntheticDataGenerator,

    /// <summary>
    /// A physics-informed model that incorporates physical laws as constraints.
    /// Combines data-driven learning with domain knowledge from physics equations.
    /// </summary>
    PhysicsInformed,

    /// <summary>
    /// A neural operator that learns mappings between function spaces.
    /// Examples: Fourier Neural Operator, DeepONet for solving PDEs.
    /// </summary>
    NeuralOperator,

    /// <summary>
    /// An autonomous agent that uses AI models to take actions and make decisions.
    /// Examples: LLM-based agents, planning agents, tool-using agents.
    /// </summary>
    Agent,

    /// <summary>
    /// A classical signal processing method that uses mathematical transforms and filters.
    /// Examples: spectral subtraction, Wiener filtering, beamforming, parametric EQ.
    /// </summary>
    SignalProcessing
}
