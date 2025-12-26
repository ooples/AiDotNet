namespace AiDotNet.Augmentation.Interfaces;

/// <summary>
/// Specifies the type of machine learning task for augmentation recommendations.
/// </summary>
public enum AugmentationTaskType
{
    /// <summary>
    /// Image classification task.
    /// </summary>
    ImageClassification,

    /// <summary>
    /// Object detection task.
    /// </summary>
    ObjectDetection,

    /// <summary>
    /// Instance segmentation task.
    /// </summary>
    InstanceSegmentation,

    /// <summary>
    /// Semantic segmentation task.
    /// </summary>
    SemanticSegmentation,

    /// <summary>
    /// Pose estimation task.
    /// </summary>
    PoseEstimation,

    /// <summary>
    /// Audio classification task.
    /// </summary>
    AudioClassification,

    /// <summary>
    /// Speech recognition task.
    /// </summary>
    SpeechRecognition,

    /// <summary>
    /// Tabular classification task.
    /// </summary>
    TabularClassification,

    /// <summary>
    /// Tabular regression task.
    /// </summary>
    TabularRegression,

    /// <summary>
    /// Time series forecasting task.
    /// </summary>
    TimeSeriesForecasting,

    /// <summary>
    /// Custom/unknown task type.
    /// </summary>
    Custom
}

/// <summary>
/// Represents metadata about a dataset for augmentation recommendations.
/// </summary>
public class DatasetCharacteristics
{
    /// <summary>
    /// Gets or sets the number of samples in the dataset.
    /// </summary>
    public int SampleCount { get; set; }

    /// <summary>
    /// Gets or sets the number of classes (for classification).
    /// </summary>
    public int NumClasses { get; set; }

    /// <summary>
    /// Gets or sets whether the dataset is imbalanced.
    /// </summary>
    public bool IsImbalanced { get; set; }

    /// <summary>
    /// Gets or sets the class imbalance ratio (max/min class count).
    /// </summary>
    public double ImbalanceRatio { get; set; }

    /// <summary>
    /// Gets or sets the average image dimensions (for image data).
    /// </summary>
    public (int width, int height)? ImageDimensions { get; set; }

    /// <summary>
    /// Gets or sets whether images have varying sizes.
    /// </summary>
    public bool HasVariableSizes { get; set; }

    /// <summary>
    /// Gets or sets the number of features (for tabular data).
    /// </summary>
    public int NumFeatures { get; set; }

    /// <summary>
    /// Gets or sets the percentage of missing values.
    /// </summary>
    public double MissingValuePercentage { get; set; }

    /// <summary>
    /// Gets or sets whether the data contains spatial targets.
    /// </summary>
    public bool HasSpatialTargets { get; set; }

    /// <summary>
    /// Gets or sets whether the data contains bounding boxes.
    /// </summary>
    public bool HasBoundingBoxes { get; set; }

    /// <summary>
    /// Gets or sets whether the data contains keypoints.
    /// </summary>
    public bool HasKeypoints { get; set; }

    /// <summary>
    /// Gets or sets whether the data contains segmentation masks.
    /// </summary>
    public bool HasMasks { get; set; }

    /// <summary>
    /// Gets or sets custom characteristics as key-value pairs.
    /// </summary>
    public IDictionary<string, object>? CustomProperties { get; set; }
}

/// <summary>
/// Represents a recommendation for an augmentation with its configuration.
/// </summary>
public class AugmentationRecommendation
{
    /// <summary>
    /// Gets or sets the augmentation type name.
    /// </summary>
    public string AugmentationType { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the recommended probability (0.0 to 1.0).
    /// </summary>
    public double RecommendedProbability { get; set; }

    /// <summary>
    /// Gets or sets the recommended parameters.
    /// </summary>
    public IDictionary<string, object> RecommendedParameters { get; set; } = new Dictionary<string, object>();

    /// <summary>
    /// Gets or sets the confidence score of this recommendation (0.0 to 1.0).
    /// </summary>
    public double ConfidenceScore { get; set; }

    /// <summary>
    /// Gets or sets the priority order (lower = higher priority).
    /// </summary>
    public int Priority { get; set; }

    /// <summary>
    /// Gets or sets the reason for this recommendation.
    /// </summary>
    public string? Reason { get; set; }

    /// <summary>
    /// Gets or sets whether this augmentation is critical for the task.
    /// </summary>
    public bool IsCritical { get; set; }

    /// <summary>
    /// Gets or sets augmentations this is incompatible with.
    /// </summary>
    public IList<string>? IncompatibleWith { get; set; }
}

/// <summary>
/// Interface for recommending augmentations based on task and data characteristics.
/// </summary>
/// <remarks>
/// <para>
/// This interface enables integration with agent systems and AutoML pipelines
/// by providing intelligent augmentation recommendations based on:
/// - The type of ML task being performed
/// - Characteristics of the dataset
/// - Best practices from research and industry
/// </para>
/// <para><b>For Beginners:</b> Different tasks need different augmentations.
/// Object detection needs augmentations that preserve bounding boxes,
/// while pose estimation needs ones that correctly transform keypoints.
/// This recommender helps choose the right augmentations automatically.
/// </para>
/// </remarks>
public interface IAugmentationRecommender
{
    /// <summary>
    /// Gets recommendations for augmentations based on task and data.
    /// </summary>
    /// <param name="taskType">The ML task type.</param>
    /// <param name="characteristics">The dataset characteristics.</param>
    /// <returns>A list of recommended augmentations ordered by priority.</returns>
    IList<AugmentationRecommendation> GetRecommendations(
        AugmentationTaskType taskType,
        DatasetCharacteristics characteristics);

    /// <summary>
    /// Gets a pre-configured augmentation policy for a task.
    /// </summary>
    /// <param name="taskType">The ML task type.</param>
    /// <param name="strength">The augmentation strength (0.0 = none, 1.0 = maximum).</param>
    /// <returns>A ready-to-use augmentation policy.</returns>
    IAugmentationPolicy GetDefaultPolicy(AugmentationTaskType taskType, double strength = 0.5);

    /// <summary>
    /// Validates whether augmentations are compatible with the task.
    /// </summary>
    /// <param name="augmentations">The augmentations to validate.</param>
    /// <param name="taskType">The ML task type.</param>
    /// <returns>Validation result with any incompatibility warnings.</returns>
    AugmentationValidationResult ValidateAugmentations(
        IEnumerable<string> augmentations,
        AugmentationTaskType taskType);
}

/// <summary>
/// Result of validating augmentations for a task.
/// </summary>
public class AugmentationValidationResult
{
    /// <summary>
    /// Gets or sets whether the augmentations are valid.
    /// </summary>
    public bool IsValid { get; set; }

    /// <summary>
    /// Gets or sets any warnings about the augmentations.
    /// </summary>
    public IList<string> Warnings { get; set; } = new List<string>();

    /// <summary>
    /// Gets or sets any errors that make the configuration invalid.
    /// </summary>
    public IList<string> Errors { get; set; } = new List<string>();

    /// <summary>
    /// Gets or sets suggested fixes for any issues.
    /// </summary>
    public IList<string> SuggestedFixes { get; set; } = new List<string>();
}

/// <summary>
/// Interface for composable augmentation policies.
/// </summary>
/// <remarks>
/// Policies define collections of augmentations and their application strategies.
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <typeparam name="TData">The data type being augmented.</typeparam>
public interface IAugmentationPolicy<T, TData>
{
    /// <summary>
    /// Gets the name of this policy.
    /// </summary>
    string Name { get; }

    /// <summary>
    /// Gets the augmentations in this policy.
    /// </summary>
    IList<IAugmentation<T, TData>> Augmentations { get; }

    /// <summary>
    /// Applies the policy to input data.
    /// </summary>
    /// <param name="data">The input data.</param>
    /// <param name="context">The augmentation context.</param>
    /// <returns>The augmented data.</returns>
    TData Apply(TData data, AugmentationContext<T>? context = null);

    /// <summary>
    /// Gets the parameters of this policy for serialization.
    /// </summary>
    /// <returns>A dictionary of policy configuration.</returns>
    IDictionary<string, object> GetConfiguration();
}

/// <summary>
/// Non-generic base interface for augmentation policies.
/// </summary>
public interface IAugmentationPolicy
{
    /// <summary>
    /// Gets the name of this policy.
    /// </summary>
    string Name { get; }

    /// <summary>
    /// Gets the number of augmentations in this policy.
    /// </summary>
    int AugmentationCount { get; }

    /// <summary>
    /// Gets the augmentation names in this policy.
    /// </summary>
    IList<string> AugmentationNames { get; }
}
