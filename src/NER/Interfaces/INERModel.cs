using AiDotNet.Interfaces;

namespace AiDotNet.NER.Interfaces;

/// <summary>
/// Base interface for all Named Entity Recognition (NER) AI models in AiDotNet.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// This interface extends IFullModel to provide the core contract for NER models,
/// inheriting standard methods for training, inference, model persistence, and gradient computation.
/// </para>
/// <para>
/// <b>For Beginners:</b> A NER model processes text to identify and classify named entities
/// such as people, organizations, locations, dates, and more.
///
/// Key concepts:
/// - Input tensors represent token embeddings with shape [batch, sequenceLength, embeddingDim]
/// - Output tensors represent label scores with shape [batch, sequenceLength, numLabels]
/// - Models can run in Native mode (pure C#) or ONNX mode (optimized runtime)
/// - All models support both training and inference
/// - Models inherit full serialization, checkpointing, and gradient computation from IFullModel
///
/// Example usage:
/// <code>
/// // Create a model
/// var architecture = new NeuralNetworkArchitecture&lt;double&gt;(inputFeatures: 100, outputSize: 9);
/// var model = new BiLSTMCRF&lt;double&gt;(architecture);
///
/// // Train on data
/// model.Train(tokenEmbeddings, labelSequence);
///
/// // Make predictions
/// var result = model.Predict(inputTokens);
/// </code>
/// </para>
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("NERModel")]
public interface INERModel<T> : IFullModel<T, Tensor<T>, Tensor<T>>
{
    #region NER-Specific Properties

    /// <summary>
    /// Gets the number of entity label classes this model predicts.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Common NER label sets include:
    /// - CoNLL-2003: 9 labels (O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC, B-MISC, I-MISC)
    /// - OntoNotes 5.0: 37 labels (18 entity types in BIO scheme)
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This tells you how many different entity categories the model can recognize.
    /// More labels means the model can distinguish more types of entities.
    /// </para>
    /// </remarks>
    int NumLabels { get; }

    /// <summary>
    /// Gets the embedding dimension for input token representations.
    /// </summary>
    int EmbeddingDimension { get; }

    /// <summary>
    /// Gets the expected input tensor shape for this model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// NER models accept input in the following format:
    /// - 2D tensor: [sequenceLength, embeddingDim] for a single sequence
    /// - 3D tensor: [batch, sequenceLength, embeddingDim] for batched input
    /// </para>
    /// </remarks>
    int[] ExpectedInputShape { get; }

    #endregion

    #region NER-Specific Methods

    /// <summary>
    /// Performs NER prediction on input token embeddings and returns the best label sequence.
    /// </summary>
    /// <param name="tokenEmbeddings">Token embeddings [batch, sequenceLength, embeddingDim] or [sequenceLength, embeddingDim].</param>
    /// <returns>Predicted label indices [batch, sequenceLength] or [sequenceLength].</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the main method to identify entities in text.
    /// Pass in token embeddings (numerical representations of words) and get back
    /// label predictions for each token.
    /// </para>
    /// </remarks>
    Tensor<T> PredictLabels(Tensor<T> tokenEmbeddings);

    /// <summary>
    /// Trains the model on NER data asynchronously with progress reporting.
    /// </summary>
    /// <param name="tokenEmbeddings">Input token embeddings.</param>
    /// <param name="labels">Target label sequences.</param>
    /// <param name="epochs">Number of training epochs.</param>
    /// <param name="progress">Optional progress reporter.</param>
    /// <param name="cancellationToken">Optional token to cancel training.</param>
    /// <returns>A task representing the asynchronous training operation.</returns>
    Task TrainAsync(
        Tensor<T> tokenEmbeddings,
        Tensor<T> labels,
        int epochs = 100,
        IProgress<NERTrainingProgress>? progress = null,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Processes multiple sequences in a batch.
    /// </summary>
    /// <param name="sequences">Batch of token embedding tensors.</param>
    /// <returns>Batch of label prediction tensors.</returns>
    IEnumerable<Tensor<T>> PredictBatch(IEnumerable<Tensor<T>> sequences);

    /// <summary>
    /// Validates that an input tensor has the correct shape for this model.
    /// </summary>
    /// <param name="input">The input tensor to validate.</param>
    /// <exception cref="ArgumentException">Thrown if the tensor shape is invalid.</exception>
    void ValidateInputShape(Tensor<T> input);

    /// <summary>
    /// Gets a summary of the model architecture.
    /// </summary>
    /// <returns>A string describing the model structure.</returns>
    string GetModelSummary();

    #endregion
}

/// <summary>
/// Reports training progress for NER models.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Training can take a long time. This class helps you track
/// progress and estimate how much longer training will take.
/// </para>
/// </remarks>
public class NERTrainingProgress
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
    /// Gets or sets the entity-level F1 score on validation data.
    /// </summary>
    public double F1Score { get; set; }

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
