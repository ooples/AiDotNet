namespace AiDotNet.Enums;

/// <summary>
/// Defines the high-level task family for an AutoML run.
/// </summary>
/// <remarks>
/// <para>
/// AutoML uses the task family to choose sensible defaults for metrics, evaluation protocols, and candidate selection.
/// </para>
/// <para>
/// <b>For Beginners:</b> This is the kind of problem you're solving:
/// <list type="bullet">
/// <item><description><see cref="Regression"/> predicts numbers.</description></item>
/// <item><description><see cref="BinaryClassification"/> predicts one of two outcomes.</description></item>
/// <item><description><see cref="MultiClassClassification"/> predicts one of many outcomes.</description></item>
/// <item><description><see cref="TimeSeriesForecasting"/> predicts future values from past time-ordered values.</description></item>
/// <item><description><see cref="ReinforcementLearning"/> learns by interacting with an environment to maximize reward.</description></item>
/// </list>
/// </para>
/// </remarks>
public enum AutoMLTaskFamily
{
    /// <summary>
    /// Supervised regression (predicting continuous numeric values).
    /// </summary>
    Regression,

    /// <summary>
    /// Binary (two-class) classification.
    /// </summary>
    BinaryClassification,

    /// <summary>
    /// Multi-class (single-label) classification.
    /// </summary>
    MultiClassClassification,

    /// <summary>
    /// Multi-label classification (multiple labels can be true for one sample).
    /// </summary>
    MultiLabelClassification,

    /// <summary>
    /// Time-series forecasting (predicting future values from past time-ordered values).
    /// </summary>
    TimeSeriesForecasting,

    /// <summary>
    /// Time-series anomaly detection (detecting rare/abnormal events in time-ordered data).
    /// </summary>
    TimeSeriesAnomalyDetection,

    /// <summary>
    /// Ranking (ordering items by relevance, e.g., search results).
    /// </summary>
    Ranking,

    /// <summary>
    /// Recommendation (ranking or scoring items for users).
    /// </summary>
    Recommendation,

    /// <summary>
    /// Graph node classification.
    /// </summary>
    GraphNodeClassification,

    /// <summary>
    /// Graph (whole-graph) classification.
    /// </summary>
    GraphClassification,

    /// <summary>
    /// Graph link prediction.
    /// </summary>
    GraphLinkPrediction,

    /// <summary>
    /// Graph generation.
    /// </summary>
    GraphGeneration,

    /// <summary>
    /// Text classification.
    /// </summary>
    TextClassification,

    /// <summary>
    /// Sequence tagging (token-level labels like NER, POS).
    /// </summary>
    SequenceTagging,

    /// <summary>
    /// Machine translation.
    /// </summary>
    Translation,

    /// <summary>
    /// Text generation (language modeling / free-form generation).
    /// </summary>
    TextGeneration,

    /// <summary>
    /// Speech recognition (ASR).
    /// </summary>
    SpeechRecognition,

    /// <summary>
    /// Image classification.
    /// </summary>
    ImageClassification,

    /// <summary>
    /// Object detection.
    /// </summary>
    ObjectDetection,

    /// <summary>
    /// Image segmentation.
    /// </summary>
    ImageSegmentation,

    /// <summary>
    /// Reinforcement learning.
    /// </summary>
    ReinforcementLearning,
    /// <summary>
    /// Point cloud classification (classifying entire point clouds into categories).
    /// </summary>
    /// <remarks>
    /// <para>Used for 3D object recognition, scene classification, and shape classification.</para>
    /// <para>Typical models: PointNet, PointNet++, DGCNN.</para>
    /// </remarks>
    PointCloudClassification,

    /// <summary>
    /// Point cloud segmentation (per-point classification/labeling).
    /// </summary>
    /// <remarks>
    /// <para>Used for semantic segmentation of 3D scenes, part segmentation of objects.</para>
    /// <para>Typical models: PointNet++, DGCNN with segmentation heads.</para>
    /// </remarks>
    PointCloudSegmentation,

    /// <summary>
    /// Point cloud completion (reconstructing missing parts of point clouds).
    /// </summary>
    /// <remarks>
    /// <para>Used for 3D reconstruction from partial scans.</para>
    /// </remarks>
    PointCloudCompletion,

    /// <summary>
    /// Volumetric classification (classifying 3D voxel grids).
    /// </summary>
    /// <remarks>
    /// <para>Used for 3D medical imaging, CT/MRI analysis.</para>
    /// <para>Typical models: 3D CNN, 3D U-Net, VoxelCNN.</para>
    /// </remarks>
    VolumetricClassification,

    /// <summary>
    /// Volumetric segmentation (per-voxel classification).
    /// </summary>
    /// <remarks>
    /// <para>Used for organ segmentation, tumor detection in medical imaging.</para>
    /// <para>Typical models: 3D U-Net, V-Net.</para>
    /// </remarks>
    VolumetricSegmentation,

    /// <summary>
    /// Mesh classification (classifying 3D mesh objects).
    /// </summary>
    /// <remarks>
    /// <para>Used for 3D shape recognition with mesh inputs.</para>
    /// <para>Typical models: MeshCNN, SpiralNet++, DiffusionNet.</para>
    /// </remarks>
    MeshClassification,

    /// <summary>
    /// Mesh segmentation (per-face or per-vertex classification).
    /// </summary>
    /// <remarks>
    /// <para>Used for mesh part segmentation, texture segmentation.</para>
    /// <para>Typical models: MeshCNN, DiffusionNet.</para>
    /// </remarks>
    MeshSegmentation,

    /// <summary>
    /// Neural radiance field reconstruction (novel view synthesis from images).
    /// </summary>
    /// <remarks>
    /// <para>Used for 3D scene reconstruction and photorealistic rendering.</para>
    /// <para>Typical models: NeRF, Instant-NGP, Gaussian Splatting.</para>
    /// </remarks>
    RadianceFieldReconstruction,

    /// <summary>
    /// 3D object detection (detecting and localizing objects in 3D space).
    /// </summary>
    /// <remarks>
    /// <para>Used for autonomous driving, robotics, AR/VR applications.</para>
    /// </remarks>
    ThreeDObjectDetection,

    /// <summary>
    /// Depth estimation (predicting depth from 2D images).
    /// </summary>
    /// <remarks>
    /// <para>Used for monocular/stereo depth prediction, 3D reconstruction.</para>
    /// </remarks>
    DepthEstimation
}

