namespace AiDotNet.Enums;

/// <summary>
/// Defines the application domain or field that a machine learning model is designed for.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This tells you what area or industry a model is best suited for.
/// A model can belong to multiple domains — for example, an image classification model
/// could be used in both Vision and Healthcare (medical imaging).
/// </para>
/// </remarks>
public enum ModelDomain
{
    /// <summary>
    /// General-purpose model not specific to any particular domain.
    /// Suitable for a wide range of applications.
    /// </summary>
    General,

    /// <summary>
    /// Computer vision models that process and understand images and video frames.
    /// Examples: image classification, object detection, segmentation.
    /// </summary>
    Vision,

    /// <summary>
    /// Natural language processing models that work with text and language.
    /// Examples: text classification, translation, summarization, question answering.
    /// </summary>
    Language,

    /// <summary>
    /// Audio processing models that analyze sound, speech, and music.
    /// Examples: speech recognition, music generation, audio classification.
    /// </summary>
    Audio,

    /// <summary>
    /// Video processing models that understand temporal sequences of frames.
    /// Examples: action recognition, video generation, frame interpolation.
    /// </summary>
    Video,

    /// <summary>
    /// Models that combine multiple modalities (text, image, audio, video).
    /// Examples: vision-language models, text-to-image, audio-visual models.
    /// </summary>
    Multimodal,

    /// <summary>
    /// Models designed for healthcare and biomedical applications.
    /// Examples: medical imaging, drug discovery, patient outcome prediction.
    /// </summary>
    Healthcare,

    /// <summary>
    /// Models designed for financial applications and markets.
    /// Examples: stock prediction, risk assessment, fraud detection, algorithmic trading.
    /// </summary>
    Finance,

    /// <summary>
    /// Models designed for scientific research and discovery.
    /// Examples: molecular simulation, climate modeling, physics-informed predictions.
    /// </summary>
    Science,

    /// <summary>
    /// Models designed for robotics and autonomous control systems.
    /// Examples: robot navigation, manipulation planning, motor control.
    /// </summary>
    Robotics,

    /// <summary>
    /// Models that operate on graph-structured data (nodes and edges).
    /// Examples: social network analysis, molecule property prediction, knowledge graphs.
    /// </summary>
    GraphAnalysis,

    /// <summary>
    /// Models that work with 3D data such as point clouds, meshes, and voxels.
    /// Examples: 3D object generation, depth estimation, 3D reconstruction.
    /// </summary>
    ThreeD,

    /// <summary>
    /// Models optimized for structured tabular data (rows and columns).
    /// Examples: customer churn prediction, credit scoring, feature engineering.
    /// </summary>
    Tabular,

    /// <summary>
    /// Models designed for sequential time-dependent data.
    /// Examples: forecasting, anomaly detection in time series, trend analysis.
    /// </summary>
    TimeSeries,

    /// <summary>
    /// Models focused on generating new content (images, text, audio, etc.).
    /// Examples: GANs, diffusion models, variational autoencoders.
    /// </summary>
    Generative,

    /// <summary>
    /// Models that learn through interaction with an environment via rewards.
    /// Examples: game playing, robot control, resource allocation.
    /// </summary>
    ReinforcementLearning,

    /// <summary>
    /// Models designed for causal reasoning and intervention analysis.
    /// Examples: treatment effect estimation, causal discovery, counterfactual analysis.
    /// </summary>
    Causal,

    /// <summary>
    /// Traditional machine learning models for general-purpose prediction tasks.
    /// Examples: classification, regression, clustering on structured/tabular data.
    /// </summary>
    MachineLearning
}
