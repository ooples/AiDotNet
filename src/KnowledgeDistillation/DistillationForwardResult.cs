using AiDotNet.LinearAlgebra;

namespace AiDotNet.KnowledgeDistillation;

/// <summary>
/// Encapsulates the result of a forward pass during knowledge distillation training.
/// </summary>
/// <typeparam name="T">The numeric type for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> When training with knowledge distillation, we need more than just
/// the final output of a model. We also need intermediate layer activations to enable advanced
/// distillation techniques (like feature matching or neuron selectivity). This class packages both
/// the final output and optional intermediate activations together.</para>
///
/// <para><b>Components:</b>
/// - <b>FinalOutput</b>: The model's final predictions (e.g., class logits). Shape: [batch_size x num_classes]
/// - <b>IntermediateActivations</b>: Internal layer outputs collected during forward pass (optional)</para>
///
/// <para><b>Example:</b>
/// <code>
/// // Standard forward pass (no intermediate activations)
/// var result = new DistillationForwardResult&lt;double&gt;(finalOutput);
///
/// // Forward pass with intermediate activations collection
/// var activations = new IntermediateActivations&lt;double&gt;();
/// activations.Add("layer1", layer1Output);
/// activations.Add("layer2", layer2Output);
/// var result = new DistillationForwardResult&lt;double&gt;(finalOutput, activations);
/// </code>
/// </para>
///
/// <para><b>Used By:</b>
/// - Teacher models: Provide reference outputs and activations
/// - Student models: Provide outputs to compare against teacher
/// - Distillation strategies: Compute loss from both final outputs and intermediate activations</para>
/// </remarks>
public class DistillationForwardResult<T>
{
    /// <summary>
    /// The final output of the model's forward pass.
    /// </summary>
    /// <remarks>
    /// <para>Shape: [batch_size x output_dimension]</para>
    /// <para>For classification: Each row contains logits for one sample.</para>
    /// <para>For regression: Each row contains predicted values for one sample.</para>
    /// </remarks>
    public Matrix<T> FinalOutput { get; }

    /// <summary>
    /// Optional intermediate layer activations collected during the forward pass.
    /// </summary>
    /// <remarks>
    /// <para>Will be null if intermediate activations were not requested.</para>
    /// <para>Only needed for advanced distillation strategies that match internal representations
    /// (e.g., feature-based distillation, attention transfer, neuron selectivity).</para>
    /// </remarks>
    public IntermediateActivations<T> IntermediateActivations { get; }

    /// <summary>
    /// Initializes a new instance of DistillationForwardResult with final output only.
    /// </summary>
    /// <param name="finalOutput">The final output matrix from the forward pass.</param>
    /// <remarks>
    /// <para>Use this constructor for standard response-based distillation that only needs final outputs.</para>
    /// </remarks>
    public DistillationForwardResult(Matrix<T> finalOutput)
        : this(finalOutput, null)
    {
    }

    /// <summary>
    /// Initializes a new instance of DistillationForwardResult with final output and intermediate activations.
    /// </summary>
    /// <param name="finalOutput">The final output matrix from the forward pass.</param>
    /// <param name="intermediateActivations">Optional intermediate layer activations.</param>
    /// <remarks>
    /// <para>Use this constructor for advanced distillation strategies that need intermediate layer outputs.</para>
    /// </remarks>
    public DistillationForwardResult(Matrix<T> finalOutput, IntermediateActivations<T> intermediateActivations)
    {
        FinalOutput = finalOutput ?? throw new System.ArgumentNullException(nameof(finalOutput));
        IntermediateActivations = intermediateActivations;
    }

    /// <summary>
    /// Checks if intermediate activations are available.
    /// </summary>
    public bool HasIntermediateActivations => IntermediateActivations != null && IntermediateActivations.LayerCount > 0;
}
