namespace AiDotNet.Enums;

/// <summary>
/// Defines the different types of tasks that a neural network can be designed to perform.
/// </summary>
public enum NeuralNetworkTaskType
{
    /// <summary>
    /// Binary classification task (two classes)
    /// </summary>
    BinaryClassification,
    
    /// <summary>
    /// Multi-class classification task (more than two classes)
    /// </summary>
    MultiClassClassification,
    
    /// <summary>
    /// Multi-label classification task (multiple labels can be assigned)
    /// </summary>
    MultiLabelClassification,
    
    /// <summary>
    /// Regression task (predicting continuous values)
    /// </summary>
    Regression,
    
    /// <summary>
    /// Sequence-to-sequence task (e.g., machine translation)
    /// </summary>
    SequenceToSequence,

    /// <summary>
    /// Sequence-to-sequence classification
    /// </summary>
    SequenceClassification,
    
    /// <summary>
    /// Time series forecasting task
    /// </summary>
    TimeSeriesForecasting,
    
    /// <summary>
    /// Image classification task
    /// </summary>
    ImageClassification,
    
    /// <summary>
    /// Object detection task
    /// </summary>
    ObjectDetection,
    
    /// <summary>
    /// Image segmentation task
    /// </summary>
    ImageSegmentation,
    
    /// <summary>
    /// Natural language processing task
    /// </summary>
    NaturalLanguageProcessing,
    
    /// <summary>
    /// Text generation task
    /// </summary>
    TextGeneration,
    
    /// <summary>
    /// Reinforcement learning task
    /// </summary>
    ReinforcementLearning,
    
    /// <summary>
    /// Anomaly detection task
    /// </summary>
    AnomalyDetection,
    
    /// <summary>
    /// Recommendation system task
    /// </summary>
    Recommendation,
    
    /// <summary>
    /// Clustering task
    /// </summary>
    Clustering,
    
    /// <summary>
    /// Dimensionality reduction task
    /// </summary>
    DimensionalityReduction,
    
    /// <summary>
    /// Generative task (e.g., GANs, VAEs)
    /// </summary>
    Generative,
    
    /// <summary>
    /// Speech recognition task
    /// </summary>
    SpeechRecognition,
    
    /// <summary>
    /// Audio processing task
    /// </summary>
    AudioProcessing,

    /// <summary>
    /// Language translation
    /// </summary>
    Translation,
    
    /// <summary>
    /// Custom task type
    /// </summary>
    Custom
}