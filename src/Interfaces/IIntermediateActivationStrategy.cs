using AiDotNet.KnowledgeDistillation;

namespace AiDotNet.Interfaces;

/// <summary>
/// Defines methods for distillation strategies that utilize intermediate layer activations.
/// </summary>
/// <typeparam name="T">The numeric type for calculations (e.g., double, float).</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Some advanced distillation strategies don't just compare final outputs.
/// They also compare what's happening inside the models at intermediate layers. This interface is for
/// those advanced strategies.</para>
///
/// <para><b>Example Strategies Needing Intermediate Activations:</b>
/// - <b>Feature-Based Distillation (FitNets)</b>: Match intermediate layer features between teacher and student
/// - <b>Attention Transfer</b>: Transfer attention patterns from internal layers
/// - <b>Neuron Selectivity</b>: Match how individual neurons respond across batches
/// - <b>Relational Knowledge Distillation</b>: Transfer relationships between layer activations</para>
///
/// <para><b>Why Separate Interface?</b>
/// Not all strategies need intermediate activations. Simple response-based distillation only needs final outputs.
/// This interface is optional - only implement it if your strategy needs access to internal layer outputs.</para>
///
/// <para><b>Usage Pattern:</b>
/// <code>
/// // Strategy that needs both final outputs AND intermediate activations
/// public class MyAdvancedStrategy&lt;T&gt; : DistillationStrategyBase&lt;T&gt;, IIntermediateActivationStrategy&lt;T&gt;
/// {
///     // Implement standard loss/gradient for final outputs
///     public override T ComputeLoss(Matrix&lt;T&gt; studentBatch, Matrix&lt;T&gt; teacherBatch, Matrix&lt;T&gt; labels) { ... }
///     public override Matrix&lt;T&gt; ComputeGradient(...) { ... }
///
///     // Implement intermediate activation loss
///     public T ComputeIntermediateLoss(
///         IntermediateActivations&lt;T&gt; studentActivations,
///         IntermediateActivations&lt;T&gt; teacherActivations) { ... }
/// }
/// </code>
/// </para>
/// </remarks>
public interface IIntermediateActivationStrategy<T>
{
    /// <summary>
    /// Computes a loss component based on intermediate layer activations for a batch.
    /// </summary>
    /// <param name="studentIntermediateActivations">Student's intermediate activations for a batch.</param>
    /// <param name="teacherIntermediateActivations">Teacher's intermediate activations for a batch.</param>
    /// <returns>The computed intermediate activation loss component.</returns>
    /// <remarks>
    /// <para><b>For Beginners:</b> This method compares what's happening inside the teacher and student models,
    /// not just their final outputs. For example:
    /// - Are neurons in layer 3 responding similarly?
    /// - Do attention patterns match in the middle layers?
    /// - Are feature representations aligned?</para>
    ///
    /// <para><b>How It's Used:</b>
    /// The trainer collects intermediate activations during forward passes, then calls this method
    /// to compute an additional loss component. This loss is combined with the standard output loss.</para>
    ///
    /// <para><b>Example Calculation:</b>
    /// <code>
    /// // In training loop
    /// var teacherResult = teacher.Forward(inputBatch, collectIntermediateActivations: true);
    /// var studentResult = student.Forward(inputBatch, collectIntermediateActivations: true);
    ///
    /// // Standard loss (final outputs)
    /// T outputLoss = strategy.ComputeLoss(studentResult.FinalOutput, teacherResult.FinalOutput, labels);
    ///
    /// // Intermediate loss (internal layers) - only if strategy implements this interface
    /// T intermediateLoss = 0;
    /// if (strategy is IIntermediateActivationStrategy&lt;T&gt; advancedStrategy)
    /// {
    ///     intermediateLoss = advancedStrategy.ComputeIntermediateLoss(
    ///         studentResult.IntermediateActivations,
    ///         teacherResult.IntermediateActivations);
    /// }
    ///
    /// // Total loss
    /// T totalLoss = outputLoss + intermediateLoss;
    /// </code>
    /// </para>
    ///
    /// <para><b>Implementation Tips:</b>
    /// - Match layers by name (e.g., "conv1", "layer3")
    /// - Handle mismatched architectures gracefully (teacher has more layers than student)
    /// - Consider weighting different layers differently (early layers vs. late layers)
    /// - Normalize activations if needed (different layers have different scales)</para>
    /// </remarks>
    T ComputeIntermediateLoss(
        IntermediateActivations<T> studentIntermediateActivations,
        IntermediateActivations<T> teacherIntermediateActivations);
}
