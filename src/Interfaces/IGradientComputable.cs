namespace AiDotNet.Interfaces;

/// <summary>
/// Interface for models that can compute gradients explicitly for MAML and other meta-learning algorithms.
/// </summary>
/// <typeparam name="T">The numeric data type (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// This interface extends models with the ability to compute gradients without updating parameters,
/// which is essential for meta-learning algorithms like MAML that need to:
/// 1. Compute gradients on query sets after adaptation
/// 2. Backpropagate through the adaptation process (second-order)
/// 3. Manually control gradient computation and application
/// </para>
/// <para><b>For Beginners:</b>
/// Regular training computes gradients and immediately updates the model.
/// This interface allows you to compute gradients WITHOUT updating the model,
/// giving you full control over when and how to apply them.
///
/// This is crucial for meta-learning where you need to:
/// - See what the gradients are without applying them
/// - Use gradients from one dataset to update different parameters
/// - Compute gradients through multiple steps of optimization
/// </para>
/// </remarks>
public interface IGradientComputable<T, TInput, TOutput>
{
    /// <summary>
    /// Computes gradients of the loss function with respect to model parameters for the given data,
    /// WITHOUT updating the model parameters.
    /// </summary>
    /// <param name="input">The input data.</param>
    /// <param name="target">The target/expected output.</param>
    /// <param name="lossFunction">The loss function to use for gradient computation.</param>
    /// <returns>A vector containing gradients with respect to all model parameters.</returns>
    /// <remarks>
    /// <para>
    /// This method performs a forward pass, computes the loss, and back-propagates to compute gradients,
    /// but does NOT update the model's parameters. The parameters remain unchanged after this call.
    /// </para>
    /// <para><b>For MAML:</b>
    /// After adapting a model on a support set, you can use this method to compute gradients
    /// on the query set. These gradients become the meta-gradients for updating the meta-parameters.
    /// </para>
    /// <para><b>For Beginners:</b>
    /// Think of this as "dry run" training:
    /// - The model sees what direction it should move (the gradients)
    /// - But it doesn't actually move (parameters stay the same)
    /// - You get to decide what to do with this information
    /// </para>
    /// </remarks>
    Vector<T> ComputeGradients(TInput input, TOutput target, ILossFunction<T> lossFunction);

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
    /// </remarks>
    void ApplyGradients(Vector<T> gradients, T learningRate);

    /// <summary>
    /// Computes second-order gradients (Hessian-vector product) for full MAML.
    /// </summary>
    /// <param name="adaptationSteps">The sequence of adaptation steps (support set training).</param>
    /// <param name="queryInput">The query set input.</param>
    /// <param name="queryTarget">The query set target.</param>
    /// <param name="lossFunction">The loss function.</param>
    /// <param name="innerLearningRate">The inner loop learning rate.</param>
    /// <returns>The meta-gradient computed through the adaptation process.</returns>
    /// <remarks>
    /// <para>
    /// This computes the true MAML gradient by backpropagating through the inner loop adaptation.
    /// This requires computing second-order derivatives (gradients of gradients).
    /// </para>
    /// <para><b>Algorithm:</b>
    /// 1. Record the adaptation trajectory (parameter updates during inner loop)
    /// 2. Compute query loss gradient at adapted parameters
    /// 3. Backpropagate through each adaptation step
    /// 4. Return gradient w.r.t. original (pre-adaptation) parameters
    /// </para>
    /// <para><b>For Beginners:</b>
    /// Full MAML asks: "If I change my starting point slightly, how does that affect
    /// my performance after adaptation?"
    ///
    /// This requires tracking not just how to improve on this task, but how changing
    /// the starting point would have changed the whole adaptation process.
    /// It's more accurate but computationally expensive.
    /// </para>
    /// </remarks>
    Vector<T> ComputeSecondOrderGradients(
        List<(TInput input, TOutput target)> adaptationSteps,
        TInput queryInput,
        TOutput queryTarget,
        ILossFunction<T> lossFunction,
        T innerLearningRate);
}
