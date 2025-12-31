using AiDotNet.Interfaces;

namespace AiDotNet.Video.Interfaces;

/// <summary>
/// Base interface for all video AI models in AiDotNet.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// This interface extends IFullModel to provide the core contract for video AI models,
/// inheriting standard methods for training, inference, model persistence, and gradient computation.
/// </para>
/// <para>
/// <b>For Beginners:</b> A video AI model processes video data (sequences of frames)
/// to perform tasks like enhancement, classification, or feature extraction.
///
/// Key concepts:
/// - Video tensors have shape [batch, numFrames, channels, height, width]
/// - Models can run in Native mode (pure C#) or ONNX mode (optimized runtime)
/// - All models support both training and inference
/// - Models inherit full serialization, checkpointing, and gradient computation from IFullModel
///
/// Example usage:
/// <code>
/// // Create a model
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(InputType.ThreeDimensional, 128, 128, 3);
/// var model = new RealESRGAN&lt;double&gt;(architecture, scaleFactor: 4);
///
/// // Train on data
/// model.Train(lowResInput, highResTarget);
///
/// // Save for later
/// model.Save("model.bin");
///
/// // Make predictions
/// var result = model.Predict(inputVideo);
/// </code>
/// </para>
/// </remarks>
public interface IVideoModel<T> : IFullModel<T, Tensor<T>, Tensor<T>>
{
    #region Video-Specific Properties

    /// <summary>
    /// Gets the expected input tensor shape for this model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The standard video tensor shape is [batch, numFrames, channels, height, width].
    /// For example, [1, 16, 3, 224, 224] means 1 video with 16 frames, RGB (3 channels), 224x224 resolution.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This tells you what shape your input video data needs to be.
    /// If your video doesn't match this shape, you'll need to resize or pad it.
    /// </para>
    /// </remarks>
    int[] ExpectedInputShape { get; }

    #endregion

    #region Video-Specific Methods

    /// <summary>
    /// Trains the model on video data asynchronously with progress reporting.
    /// </summary>
    /// <param name="videos">Input video tensors.</param>
    /// <param name="labels">Target labels or output tensors.</param>
    /// <param name="epochs">Number of training epochs.</param>
    /// <param name="progress">Optional progress reporter.</param>
    /// <param name="cancellationToken">Optional token to cancel training.</param>
    /// <returns>A task representing the asynchronous training operation.</returns>
    Task TrainAsync(
        Tensor<T> videos,
        Tensor<T> labels,
        int epochs = 100,
        IProgress<TrainingProgress>? progress = null,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Processes multiple videos in a batch.
    /// </summary>
    /// <param name="videos">Batch of video tensors.</param>
    /// <returns>Batch of output tensors.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Processing videos in batches is more efficient than one at a time,
    /// especially on GPUs. Use this when you have multiple videos to process.
    /// </para>
    /// </remarks>
    IEnumerable<Tensor<T>> PredictBatch(IEnumerable<Tensor<T>> videos);

    /// <summary>
    /// Validates that an input tensor has the correct shape for this model.
    /// </summary>
    /// <param name="video">The video tensor to validate.</param>
    /// <exception cref="ArgumentException">Thrown if the tensor shape is invalid.</exception>
    void ValidateInputShape(Tensor<T> video);

    /// <summary>
    /// Gets a summary of the model architecture.
    /// </summary>
    /// <returns>A string describing the model structure.</returns>
    string GetModelSummary();

    #endregion
}

/// <summary>
/// Reports training progress for video AI models.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Training can take a long time. This class helps you track
/// progress and estimate how much longer training will take.
/// </para>
/// </remarks>
public class TrainingProgress
{
    /// <summary>
    /// Gets or sets the current epoch number (1-based).
    /// </summary>
    public int CurrentEpoch { get; set; }

    /// <summary>
    /// Gets or sets the total number of epochs.
    /// </summary>
    public int TotalEpochs { get; set; }

    /// <summary>
    /// Gets or sets the current batch number within the epoch.
    /// </summary>
    public int CurrentBatch { get; set; }

    /// <summary>
    /// Gets or sets the total number of batches per epoch.
    /// </summary>
    public int TotalBatches { get; set; }

    /// <summary>
    /// Gets or sets the current training loss value.
    /// </summary>
    public double Loss { get; set; }

    /// <summary>
    /// Gets or sets any additional metrics being tracked.
    /// </summary>
    public Dictionary<string, double>? Metrics { get; set; }

    /// <summary>
    /// Gets the overall progress as a percentage (0-100).
    /// </summary>
    public double ProgressPercentage => TotalEpochs > 0
        ? (CurrentEpoch - 1 + (TotalBatches > 0 ? (double)CurrentBatch / TotalBatches : 0)) / TotalEpochs * 100
        : 0;
}
