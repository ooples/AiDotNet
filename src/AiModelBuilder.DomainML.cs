using AiDotNet.Clustering.Interfaces;
using AiDotNet.ContinualLearning.Interfaces;
using AiDotNet.Document.Interfaces;
using AiDotNet.Finance.Interfaces;
using AiDotNet.Interfaces;
using AiDotNet.PointCloud.Interfaces;
using AiDotNet.Scoring;
using AiDotNet.Video.Interfaces;

namespace AiDotNet;

/// <summary>
/// Domain-specific machine learning configuration extensions for AiModelBuilder.
/// Provides Configure methods for audio, speech, diffusion, active/continual/online learning,
/// causal inference, drift detection, survival analysis, video, point cloud, document processing,
/// finance, radial basis functions, distance metrics, embeddings, scoring, explainability, and RL agents.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input type.</typeparam>
/// <typeparam name="TOutput">The output type.</typeparam>
public partial class AiModelBuilder<T, TInput, TOutput>
{
    private IAudioEffect<T>? _configuredAudioEffect;
    private ISpeechRecognizer<T>? _configuredSpeechRecognizer;
    private ITextToSpeech<T>? _configuredTextToSpeech;
    private IDiffusionModel<T>? _configuredDiffusionModel;
    private IActiveLearningStrategy<T>? _configuredActiveLearningStrategy;
    private IContinualLearner<T, TInput, TOutput>? _configuredContinualLearner;
    private IOnlineLearningModel<T>? _configuredOnlineLearningModel;
    private ICausalModel<T>? _configuredCausalModel;
    private AiDotNet.DriftDetection.IDriftDetector<T>? _configuredDriftDetector;
    private ISurvivalModel<T>? _configuredSurvivalModel;
    private IVideoModel<T>? _configuredVideoModel;
    private IPointCloudModel<T>? _configuredPointCloudModel;
    private IDocumentModel<T>? _configuredDocumentModel;
    private IFinancialModel<T>? _configuredFinancialModel;
    private IRadialBasisFunction<T>? _configuredRadialBasisFunction;
    private IDistanceMetric<T>? _configuredDistanceMetric;
    private IEmbeddingModel<T>? _configuredEmbeddingModel;
    private IScoringRule<T>? _configuredScoringRule;
    private IModelExplainer<T>? _configuredModelExplainer;
    private IAgent<T>? _configuredRLAgent;

    /// <summary>
    /// Configures an audio effect for audio signal processing pipelines.
    /// </summary>
    /// <param name="audioEffect">The audio effect implementation to use.</param>
    /// <returns>The builder instance for method chaining.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Audio effects transform sound signals, such as applying
    /// reverb, equalization, noise reduction, or pitch shifting. These are useful for audio
    /// preprocessing and augmentation in speech and music ML pipelines.</para>
    /// </remarks>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureAudioEffect(IAudioEffect<T> audioEffect)
    {
        _configuredAudioEffect = audioEffect;
        return this;
    }

    /// <summary>
    /// Configures a speech recognition model for converting spoken audio to text.
    /// </summary>
    /// <param name="recognizer">The speech recognizer implementation to use.</param>
    /// <returns>The builder instance for method chaining.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Speech recognition (ASR) converts spoken words into written text.
    /// This enables voice assistants, transcription services, and voice-controlled applications.
    /// Available models range from simple HMM-based to modern transformer architectures.</para>
    /// </remarks>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureSpeechRecognizer(ISpeechRecognizer<T> recognizer)
    {
        _configuredSpeechRecognizer = recognizer;
        return this;
    }

    /// <summary>
    /// Configures a text-to-speech model for converting written text to spoken audio.
    /// </summary>
    /// <param name="tts">The text-to-speech implementation to use.</param>
    /// <returns>The builder instance for method chaining.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Text-to-speech (TTS) generates natural-sounding audio from text.
    /// Modern TTS systems can produce highly realistic speech with controllable voice characteristics,
    /// useful for accessibility, virtual assistants, and content creation.</para>
    /// </remarks>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureTextToSpeech(ITextToSpeech<T> tts)
    {
        _configuredTextToSpeech = tts;
        return this;
    }

    /// <summary>
    /// Configures a diffusion model for generative tasks (image/audio/video generation).
    /// </summary>
    /// <param name="diffusionModel">The diffusion model implementation to use.</param>
    /// <returns>The builder instance for method chaining.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Diffusion models generate new data (images, audio, etc.) by
    /// learning to gradually remove noise from random data. They power tools like DALL-E and
    /// Stable Diffusion for image generation and can be applied to audio and video as well.</para>
    /// </remarks>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureDiffusionModel(IDiffusionModel<T> diffusionModel)
    {
        _configuredDiffusionModel = diffusionModel;
        return this;
    }

    /// <summary>
    /// Configures an active learning strategy for intelligently selecting training samples.
    /// </summary>
    /// <param name="strategy">The active learning strategy implementation to use.</param>
    /// <returns>The builder instance for method chaining.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Active learning helps your model learn more efficiently by
    /// choosing the most informative samples to label next. Instead of labeling all your data,
    /// the model asks for labels on the examples it's most uncertain about, saving labeling effort.</para>
    /// </remarks>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureActiveLearning(IActiveLearningStrategy<T> strategy)
    {
        _configuredActiveLearningStrategy = strategy;
        return this;
    }

    /// <summary>
    /// Configures a continual learning trainer that can learn new tasks without forgetting old ones.
    /// </summary>
    /// <param name="learner">The continual learner implementation to use.</param>
    /// <returns>The builder instance for method chaining.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Continual learning (lifelong learning) enables your model to
    /// learn new tasks over time without forgetting what it learned before. Traditional neural
    /// networks suffer from "catastrophic forgetting" - continual learning techniques like
    /// EWC, LwF, and GEM prevent this.</para>
    /// </remarks>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureContinualLearning(IContinualLearner<T, TInput, TOutput> learner)
    {
        _configuredContinualLearner = learner;
        return this;
    }

    /// <summary>
    /// Configures an online learning model that updates incrementally with new data.
    /// </summary>
    /// <param name="learner">The online learning model implementation to use.</param>
    /// <returns>The builder instance for method chaining.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Online learning updates the model one sample at a time (or in
    /// small batches) as new data arrives, rather than retraining from scratch. This is essential
    /// for streaming data, real-time systems, and scenarios where data is too large to store.</para>
    /// </remarks>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureOnlineLearning(IOnlineLearningModel<T> learner)
    {
        _configuredOnlineLearningModel = learner;
        return this;
    }

    /// <summary>
    /// Configures a causal inference model for understanding cause-and-effect relationships.
    /// </summary>
    /// <param name="causalModel">The causal model implementation to use.</param>
    /// <returns>The builder instance for method chaining.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Causal inference goes beyond prediction to understand why things
    /// happen. While regular ML finds correlations ("ice cream sales and drownings both increase"),
    /// causal models identify actual causes ("summer causes both"). This is crucial for
    /// decision-making, policy evaluation, and understanding treatment effects.</para>
    /// </remarks>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureCausalInference(ICausalModel<T> causalModel)
    {
        _configuredCausalModel = causalModel;
        return this;
    }

    /// <summary>
    /// Configures a drift detector for monitoring changes in data distribution over time.
    /// </summary>
    /// <param name="driftDetector">The drift detector implementation to use.</param>
    /// <returns>The builder instance for method chaining.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Drift detection monitors whether the data your model sees in
    /// production is changing compared to what it was trained on. If drift is detected, it may
    /// signal that your model needs retraining. Common detectors include DDM, ADWIN, and
    /// Page-Hinkley methods.</para>
    /// </remarks>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureDriftDetection(AiDotNet.DriftDetection.IDriftDetector<T> driftDetector)
    {
        _configuredDriftDetector = driftDetector;
        return this;
    }

    /// <summary>
    /// Configures a survival analysis model for time-to-event prediction.
    /// </summary>
    /// <param name="survivalModel">The survival model implementation to use.</param>
    /// <returns>The builder instance for method chaining.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Survival analysis predicts how long until an event occurs,
    /// such as customer churn, equipment failure, or patient outcomes. Unlike regular regression,
    /// it handles censored data (when the event hasn't occurred yet for some subjects).
    /// Common models include Cox Proportional Hazards and Kaplan-Meier.</para>
    /// </remarks>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureSurvivalAnalysis(ISurvivalModel<T> survivalModel)
    {
        _configuredSurvivalModel = survivalModel;
        return this;
    }

    /// <summary>
    /// Configures a video model for video understanding and generation tasks.
    /// </summary>
    /// <param name="videoModel">The video model implementation to use.</param>
    /// <returns>The builder instance for method chaining.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Video models process sequences of frames for tasks like
    /// action recognition, video classification, temporal segmentation, and video generation.
    /// They capture both spatial (within-frame) and temporal (across-frame) information.</para>
    /// </remarks>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureVideoModel(IVideoModel<T> videoModel)
    {
        _configuredVideoModel = videoModel;
        return this;
    }

    /// <summary>
    /// Configures a point cloud model for 3D data processing.
    /// </summary>
    /// <param name="model">The point cloud model implementation to use.</param>
    /// <returns>The builder instance for method chaining.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Point cloud models process 3D data represented as collections of
    /// points in space (from LiDAR, depth cameras, etc.). They enable tasks like 3D object detection,
    /// segmentation, and scene understanding for autonomous driving and robotics.</para>
    /// </remarks>
    public IAiModelBuilder<T, TInput, TOutput> ConfigurePointCloudModel(IPointCloudModel<T> model)
    {
        _configuredPointCloudModel = model;
        return this;
    }

    /// <summary>
    /// Configures a document model for document understanding and processing.
    /// </summary>
    /// <param name="documentModel">The document model implementation to use.</param>
    /// <returns>The builder instance for method chaining.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Document models understand the layout and content of documents
    /// (PDFs, forms, invoices). They combine text understanding with spatial layout information
    /// to extract structured data, classify documents, and answer questions about document content.</para>
    /// </remarks>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureDocumentModel(IDocumentModel<T> documentModel)
    {
        _configuredDocumentModel = documentModel;
        return this;
    }

    /// <summary>
    /// Configures a financial model for quantitative finance and risk analysis.
    /// </summary>
    /// <param name="financialModel">The financial model implementation to use.</param>
    /// <returns>The builder instance for method chaining.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Financial models apply ML to finance tasks such as portfolio
    /// optimization, risk assessment, option pricing, credit scoring, and algorithmic trading.
    /// They handle specialized requirements like time-series data and risk-adjusted metrics.</para>
    /// </remarks>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureFinancialModel(IFinancialModel<T> financialModel)
    {
        _configuredFinancialModel = financialModel;
        return this;
    }

    /// <summary>
    /// Configures a radial basis function for RBF networks and interpolation.
    /// </summary>
    /// <param name="rbf">The radial basis function implementation to use.</param>
    /// <returns>The builder instance for method chaining.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Radial basis functions measure the distance from a center point
    /// and produce a response that depends only on that distance. They are used in RBF neural networks
    /// for function approximation and in scattered data interpolation. Common types include
    /// Gaussian, Multiquadric, and Thin Plate Spline.</para>
    /// </remarks>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureRadialBasisFunction(IRadialBasisFunction<T> rbf)
    {
        _configuredRadialBasisFunction = rbf;
        return this;
    }

    /// <summary>
    /// Configures a distance metric for measuring similarity between data points.
    /// </summary>
    /// <param name="distanceMetric">The distance metric implementation to use.</param>
    /// <returns>The builder instance for method chaining.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Distance metrics measure how different two data points are.
    /// They are fundamental to clustering, nearest-neighbor search, and many other ML algorithms.
    /// Common metrics include Euclidean, Manhattan, Cosine, and Mahalanobis distance.</para>
    /// </remarks>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureDistanceMetric(IDistanceMetric<T> distanceMetric)
    {
        _configuredDistanceMetric = distanceMetric;
        return this;
    }

    /// <summary>
    /// Configures an embedding model for learning dense vector representations.
    /// </summary>
    /// <param name="embeddingModel">The embedding model implementation to use.</param>
    /// <returns>The builder instance for method chaining.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Embedding models convert high-dimensional or categorical data
    /// into compact, dense vectors that capture semantic meaning. Similar items end up close
    /// together in the embedding space. Used for text (Word2Vec, BERT), images, graphs, and more.</para>
    /// </remarks>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureEmbeddingModel(IEmbeddingModel<T> embeddingModel)
    {
        _configuredEmbeddingModel = embeddingModel;
        return this;
    }

    /// <summary>
    /// Configures a scoring rule for evaluating probabilistic predictions.
    /// </summary>
    /// <param name="scorer">The scoring rule implementation to use.</param>
    /// <returns>The builder instance for method chaining.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Scoring rules evaluate how good probabilistic predictions are.
    /// Unlike simple accuracy, they assess whether predicted probabilities are well-calibrated.
    /// Common scoring rules include Brier Score, Log Loss, and CRPS (Continuous Ranked
    /// Probability Score).</para>
    /// </remarks>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureScoringRule(IScoringRule<T> scorer)
    {
        _configuredScoringRule = scorer;
        return this;
    }

    /// <summary>
    /// Configures a model explainer for understanding model predictions.
    /// </summary>
    /// <param name="explainer">The model explainer implementation to use.</param>
    /// <returns>The builder instance for method chaining.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> Model explainers help you understand why a model makes specific
    /// predictions. This is crucial for trust, debugging, and regulatory compliance. Methods
    /// include SHAP (feature importance), LIME (local explanations), Integrated Gradients,
    /// and Attention visualization.</para>
    /// </remarks>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureModelExplainer(IModelExplainer<T> explainer)
    {
        _configuredModelExplainer = explainer;
        return this;
    }

    /// <summary>
    /// Configures a reinforcement learning agent for learning through interaction with an environment.
    /// </summary>
    /// <param name="agent">The RL agent implementation to use.</param>
    /// <returns>The builder instance for method chaining.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> RL agents learn by trial and error, receiving rewards or penalties
    /// for their actions in an environment. They're used for game playing, robotics, recommendation
    /// systems, and resource optimization. Available agents include DQN, PPO, SAC, and A2C.</para>
    /// </remarks>
    public IAiModelBuilder<T, TInput, TOutput> ConfigureRLAgent(IAgent<T> agent)
    {
        _configuredRLAgent = agent;
        return this;
    }
}
