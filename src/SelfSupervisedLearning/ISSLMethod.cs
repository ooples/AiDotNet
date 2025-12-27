using AiDotNet.Enums;
using AiDotNet.Interfaces;

namespace AiDotNet.SelfSupervisedLearning;

/// <summary>
/// Defines the contract for self-supervised learning methods.
/// </summary>
/// <typeparam name="T">The numeric type used for computations (typically float or double).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Self-supervised learning methods learn useful representations from
/// unlabeled data. They create "pretext tasks" that provide supervision signals without human labels.</para>
///
/// <para>Each SSL method implements this interface and provides:</para>
/// <list type="bullet">
/// <item>A training step that processes batches and returns loss</item>
/// <item>Access to the learned encoder for downstream tasks</item>
/// <item>Encoding functionality to transform inputs into representations</item>
/// </list>
///
/// <para><b>Example usage:</b></para>
/// <code>
/// // Create an SSL method
/// var simclr = new SimCLR&lt;float&gt;(encoder, config);
///
/// // Train for one step
/// var result = simclr.TrainStep(batch, augmentationContext);
/// Console.WriteLine($"Loss: {result.Loss}");
///
/// // Get learned representations
/// var embeddings = simclr.Encode(newData);
/// </code>
/// </remarks>
public interface ISSLMethod<T>
{
    /// <summary>
    /// Gets the name of this SSL method.
    /// </summary>
    /// <remarks>
    /// <para>Examples: "SimCLR", "MoCo v2", "BYOL", "DINO", "MAE"</para>
    /// </remarks>
    string Name { get; }

    /// <summary>
    /// Gets the category of this SSL method.
    /// </summary>
    /// <remarks>
    /// <para>Categories include Contrastive, NonContrastive, Generative, and SelfDistillation.</para>
    /// </remarks>
    SSLMethodCategory Category { get; }

    /// <summary>
    /// Indicates whether this method requires a memory bank for negative samples.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Memory banks store embeddings from previous batches to use as
    /// negative samples in contrastive learning. MoCo uses this, SimCLR does not.</para>
    /// </remarks>
    bool RequiresMemoryBank { get; }

    /// <summary>
    /// Indicates whether this method uses a momentum-updated encoder.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> A momentum encoder is a slowly-updated copy of the main encoder.
    /// Methods like MoCo, BYOL, and DINO use this to provide stable targets.</para>
    /// </remarks>
    bool UsesMomentumEncoder { get; }

    /// <summary>
    /// Gets the underlying encoder neural network.
    /// </summary>
    /// <returns>The encoder network that produces representations.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> The encoder is the neural network that transforms raw inputs
    /// (like images) into learned representations. This is what you keep after pretraining.</para>
    /// </remarks>
    INeuralNetwork<T> GetEncoder();

    /// <summary>
    /// Performs a single training step on a batch of data.
    /// </summary>
    /// <param name="batch">The input batch tensor (e.g., images).</param>
    /// <param name="augmentationContext">Optional context for augmentation (method may handle internally).</param>
    /// <returns>The result of the training step including loss and metrics.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the main training loop step. It:</para>
    /// <list type="number">
    /// <item>Creates augmented views of the input</item>
    /// <item>Passes views through the encoder</item>
    /// <item>Computes the SSL loss</item>
    /// <item>Updates model parameters</item>
    /// </list>
    /// </remarks>
    SSLStepResult<T> TrainStep(Tensor<T> batch, SSLAugmentationContext<T>? augmentationContext = null);

    /// <summary>
    /// Encodes input data into learned representations.
    /// </summary>
    /// <param name="input">The input tensor to encode.</param>
    /// <returns>The encoded representation tensor.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> After pretraining, use this to get representations for downstream tasks.
    /// The output embeddings can be used for classification, clustering, or similarity search.</para>
    /// </remarks>
    Tensor<T> Encode(Tensor<T> input);

    /// <summary>
    /// Resets the SSL method to its initial state.
    /// </summary>
    /// <remarks>
    /// <para>This clears any accumulated state like memory banks, running statistics,
    /// and resets the momentum encoder if present.</para>
    /// </remarks>
    void Reset();

    /// <summary>
    /// Gets the current parameters of the SSL method for serialization.
    /// </summary>
    /// <returns>A vector containing all trainable parameters.</returns>
    Vector<T> GetParameters();

    /// <summary>
    /// Sets the parameters of the SSL method from a serialized vector.
    /// </summary>
    /// <param name="parameters">The parameter vector to load.</param>
    void SetParameters(Vector<T> parameters);

    /// <summary>
    /// Gets the total number of trainable parameters.
    /// </summary>
    int ParameterCount { get; }

    /// <summary>
    /// Called at the start of each training epoch.
    /// </summary>
    /// <param name="epochNumber">The current epoch number (0-indexed).</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method is called before each epoch begins.
    /// Methods use it to update learning rate schedules, momentum schedules, etc.</para>
    /// </remarks>
    void OnEpochStart(int epochNumber);

    /// <summary>
    /// Called at the end of each training epoch.
    /// </summary>
    /// <param name="epochNumber">The current epoch number (0-indexed).</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method is called after each epoch completes.
    /// Methods use it for cleanup, logging, or updating statistics.</para>
    /// </remarks>
    void OnEpochEnd(int epochNumber);
}
