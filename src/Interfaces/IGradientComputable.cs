using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;

namespace AiDotNet.Interfaces;

/// <summary>
/// Base interface for models that can compute gradients explicitly without updating parameters.
/// </summary>
/// <typeparam name="T">The numeric data type (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// This interface enables models to compute gradients without immediately applying parameter updates.
/// This is essential for:
/// - <b>Distributed Training</b>: Compute local gradients, synchronize across workers, then apply averaged gradients
/// - <b>Meta-Learning</b>: Compute gradients on query sets after adaptation (see <see cref="ISecondOrderGradientComputable{T, TInput, TOutput}"/>)
/// - <b>Custom Optimization</b>: Manually control when and how to apply gradients
/// - <b>Gradient Analysis</b>: Inspect gradient values for debugging or monitoring
/// </para>
/// <para><b>For Beginners:</b>
/// Regular training computes gradients and immediately updates the model in one step.
/// This interface separates those two operations:
///
/// 1. <see cref="ComputeGradients"/> - Calculate which direction improves the model (WITHOUT changing it)
/// 2. <see cref="ApplyGradients"/> - Actually update the model using those directions
///
/// This separation is crucial when you need to process gradients before applying them,
/// such as averaging gradients across multiple GPUs in distributed training.
/// </para>
/// <para><b>Distributed Training Use Case:</b>
/// In Data Parallel training (DDP), each GPU:
/// 1. Computes gradients on its local data batch
/// 2. Communicates gradients with other GPUs to compute the average
/// 3. Applies the averaged gradients to update parameters
///
/// Without this interface, step 2 would be impossible because gradients would already
/// be applied in step 1.
/// </para>
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("GradientComputable")]
public interface IGradientComputable<T, TInput, TOutput>
{
    /// <summary>
    /// Computes gradients of the loss function with respect to model parameters for the given data,
    /// WITHOUT updating the model parameters.
    /// </summary>
    /// <param name="input">The input data.</param>
    /// <param name="target">The target/expected output.</param>
    /// <param name="lossFunction">The loss function to use for gradient computation. If null, uses the model's default loss function.</param>
    /// <returns>A vector containing gradients with respect to all model parameters.</returns>
    /// <remarks>
    /// <para>
    /// This method performs a forward pass, computes the loss, and back-propagates to compute gradients,
    /// but does NOT update the model's parameters. The parameters remain unchanged after this call.
    /// </para>
    /// <para><b>Distributed Training:</b>
    /// In DDP/ZeRO-2, each worker calls this to compute local gradients on its data batch.
    /// These gradients are then synchronized (averaged) across workers before applying updates.
    /// This ensures all workers compute the same parameter updates despite having different data.
    /// </para>
    /// <para><b>For Meta-Learning:</b>
    /// After adapting a model on a support set, you can use this method to compute gradients
    /// on the query set. These gradients become the meta-gradients for updating the meta-parameters.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// Think of this as "dry run" training:
    /// - The model sees what direction it should move (the gradients)
    /// - But it doesn't actually move (parameters stay the same)
    /// - You get to decide what to do with this information (average with others, inspect, modify, etc.)
    /// </para>
    /// </remarks>
    /// <exception cref="System.InvalidOperationException">If lossFunction is null and the model has no default loss function.</exception>
    Vector<T> ComputeGradients(TInput input, TOutput target, ILossFunction<T>? lossFunction = null);

    /// <summary>
    /// Applies pre-computed gradients to update the model parameters.
    /// </summary>
    /// <param name="gradients">The gradient vector to apply.</param>
    /// <param name="learningRate">The learning rate for the update.</param>
    /// <remarks>
    /// <para>
    /// Updates parameters using: θ = θ - learningRate * gradients
    /// </para>
    /// <para><b>For Beginners:</b>
    /// After computing gradients (seeing which direction to move),
    /// this method actually moves the model in that direction.
    /// The learning rate controls how big of a step to take.
    /// </para>
    /// <para><b>Distributed Training:</b>
    /// In DDP/ZeRO-2, this applies the synchronized (averaged) gradients after
    /// communication across workers. Each worker applies the same averaged gradients
    /// to keep parameters consistent.
    /// </para>
    /// </remarks>
    void ApplyGradients(Vector<T> gradients, T learningRate);
}
