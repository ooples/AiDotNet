using AiDotNet.Enums;
using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configuration options for the Upscale4KAgent agentic multi-model pipeline.
/// </summary>
/// <remarks>
/// <para>
/// Upscale4KAgent (2025) orchestrates multiple specialized SR models in an agentic pipeline:
/// - Quality assessment agent: evaluates each frame to determine optimal processing strategy
/// - Multi-model routing: dynamically selects and chains SR models based on content analysis
/// - Iterative refinement: applies progressive upscaling stages with quality checkpoints
/// - Resolution-adaptive: handles arbitrary input resolution up to 4K output
/// </para>
/// <para>
/// <b>For Beginners:</b> Instead of using one model for everything, Upscale4KAgent acts
/// like a "manager" that looks at each frame and decides which combination of upscaling
/// models will produce the best result. It can chain multiple models together and check
/// quality at each step, similar to how a human editor would approach video upscaling.
/// </para>
/// </remarks>
public class Upscale4KAgentOptions : NeuralNetworkOptions
{
    /// <summary>
    /// Initializes a new instance with default values.
    /// </summary>
    public Upscale4KAgentOptions()
    {
    }

    /// <summary>
    /// Initializes a new instance by copying from another instance.
    /// </summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public Upscale4KAgentOptions(Upscale4KAgentOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        Variant = other.Variant;
        NumFeatures = other.NumFeatures;
        NumStages = other.NumStages;
        NumResBlocks = other.NumResBlocks;
        ScaleFactor = other.ScaleFactor;
        MaxAgentSteps = other.MaxAgentSteps;
        QualityThreshold = other.QualityThreshold;
        ModelPath = other.ModelPath;
        OnnxOptions = other.OnnxOptions;
        LearningRate = other.LearningRate;
        DropoutRate = other.DropoutRate;
    }

    #region Architecture

    /// <summary>Gets or sets the model variant.</summary>
    public VideoModelVariant Variant { get; set; } = VideoModelVariant.Base;

    /// <summary>Gets or sets the number of feature channels in the backbone.</summary>
    public int NumFeatures { get; set; } = 64;

    /// <summary>Gets or sets the number of pipeline stages for iterative refinement.</summary>
    /// <remarks>Each stage applies a specialized SR model selected by the agent controller.</remarks>
    public int NumStages { get; set; } = 3;

    /// <summary>Gets or sets the number of residual blocks per stage backbone.</summary>
    public int NumResBlocks { get; set; } = 16;

    /// <summary>Gets or sets the spatial upscaling factor.</summary>
    public int ScaleFactor { get; set; } = 4;

    /// <summary>Gets or sets the maximum number of agent decision steps per frame.</summary>
    /// <remarks>Limits the routing agent's iterations to prevent infinite loops.</remarks>
    public int MaxAgentSteps { get; set; } = 5;

    /// <summary>Gets or sets the quality threshold (0.0-1.0) for the agent to accept a result.</summary>
    /// <remarks>The agent continues refining until this SSIM-like threshold is met or MaxAgentSteps is reached.</remarks>
    public double QualityThreshold { get; set; } = 0.85;

    #endregion

    #region Model Loading

    /// <summary>Gets or sets the path to the ONNX model file.</summary>
    public string? ModelPath { get; set; }

    /// <summary>Gets or sets the ONNX runtime options.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    #endregion

    #region Training

    /// <summary>Gets or sets the learning rate.</summary>
    public double LearningRate { get; set; } = 1e-4;

    /// <summary>Gets or sets the dropout rate.</summary>
    public double DropoutRate { get; set; } = 0.0;

    #endregion
}
