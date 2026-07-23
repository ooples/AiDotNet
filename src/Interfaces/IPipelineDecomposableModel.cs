namespace AiDotNet.Interfaces;

/// <summary>
/// Interface for models that support decomposing the backward pass into separate
/// activation gradient and weight gradient computations. This enables Zero Bubble
/// pipeline schedules (ZB-H1, ZB-H2, ZB-V) to overlap weight gradient computation
/// with other pipeline stages.
/// </summary>
/// <remarks>
/// <para>
/// Standard backward passes compute both dL/dInput (activation gradients) and dL/dWeights
/// (weight gradients) together. This interface allows splitting them:
/// </para>
/// <list type="bullet">
/// <item><description>
/// <b>BackwardInput (B)</b>: Computes dL/dInput - needed by the upstream stage (critical path).
/// </description></item>
/// <item><description>
/// <b>BackwardWeight (W)</b>: Computes dL/dWeights - can be deferred to fill pipeline bubbles.
/// </description></item>
/// </list>
/// <para><b>For Beginners:</b> Most models compute all gradients at once. This interface lets
/// advanced pipeline schedules split that work into two parts: one that's urgent (the upstream
/// stage is waiting for it) and one that can wait (filling idle time in the pipeline).
///
/// If your model doesn't implement this interface, pipeline schedules will automatically
/// fall back to computing both gradient types together (which still works, just can't
/// fill bubbles as effectively).</para>
/// <para><b>Reference:</b> Qi et al., "Zero Bubble Pipeline Parallelism", ICLR 2024 Spotlight.
/// https://arxiv.org/abs/2401.10241</para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output/target data type.</typeparam>
public interface IPipelineDecomposableModel<T, TInput, TOutput>
{
    /// <summary>
    /// Computes only the activation gradients (dL/dInput) for the backward pass.
    /// This is on the critical path: the upstream pipeline stage needs these gradients
    /// to continue its own backward pass.
    /// </summary>
    /// <param name="input">The input data that was used in the forward pass.</param>
    /// <param name="target">The expected output for loss computation.</param>
    /// <returns>
    /// A tuple containing:
    /// - activationGradients: The gradient of the loss with respect to the input (dL/dInput),
    ///   used to send gradients upstream in the pipeline.
    /// - cachedState: An opaque state object that can be passed to <see cref="ComputeWeightGradients"/>
    ///   to avoid redundant computation. May be null if no caching is needed.
    /// </returns>
    (Vector<T> activationGradients, object? cachedState) ComputeActivationGradients(
        TInput input, TOutput target);

    /// <summary>
    /// Computes only the weight gradients (dL/dWeights) for the backward pass.
    /// This is NOT on the critical path and can be deferred to fill pipeline bubbles.
    /// </summary>
    /// <param name="input">The input data that was used in the forward pass.</param>
    /// <param name="target">The expected output for loss computation.</param>
    /// <param name="cachedState">
    /// Optional cached state from <see cref="ComputeActivationGradients"/> to avoid
    /// redundant forward pass computation. If null, the forward pass will be recomputed.
    /// </param>
    /// <returns>The gradient of the loss with respect to the model's weights (dL/dWeights).</returns>
    Vector<T> ComputeWeightGradients(TInput input, TOutput target, object? cachedState);
}
