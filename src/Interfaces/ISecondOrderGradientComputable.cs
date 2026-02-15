using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;

namespace AiDotNet.Interfaces;

/// <summary>
/// Extended gradient computation interface for MAML meta-learning algorithms.
/// </summary>
/// <typeparam name="T">The numeric data type (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para>
/// This interface extends <see cref="IGradientComputable{T, TInput, TOutput}"/> with second-order
/// gradient computation capability required for full MAML (Model-Agnostic Meta-Learning).
/// </para>
/// <para><b>For Beginners:</b>
/// While basic gradient computation tells you how to improve on a single task, second-order
/// gradients tell you how changing your starting point would affect learning on that task.
///
/// Think of it like this:
/// - First-order: "If I start here, which direction improves this task?"
/// - Second-order: "If I started slightly differently, how would my entire learning trajectory change?"
///
/// This is computationally expensive but more accurate for meta-learning.
/// </para>
/// <para><b>MAML Use Case:</b>
/// Full MAML uses second-order gradients to backpropagate through the inner loop adaptation,
/// computing true meta-gradients that account for how the adaptation process itself changes.
/// Reptile and first-order MAML approximate this with only first-order gradients.
/// </para>
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("SecondOrderGradientComputable")]
public interface ISecondOrderGradientComputable<T, TInput, TOutput> : IGradientComputable<T, TInput, TOutput>
{
    /// <summary>
    /// Computes second-order gradients (Hessian-vector product) for full MAML meta-learning.
    /// </summary>
    /// <param name="adaptationSteps">The sequence of adaptation steps (support set training).</param>
    /// <param name="queryInput">The query set input for evaluation after adaptation.</param>
    /// <param name="queryTarget">The query set target for evaluation after adaptation.</param>
    /// <param name="lossFunction">The loss function to use for gradient computation.</param>
    /// <param name="innerLearningRate">The inner loop learning rate for adaptation.</param>
    /// <returns>The meta-gradient computed through the entire adaptation process.</returns>
    /// <remarks>
    /// <para>
    /// This computes the true MAML gradient by backpropagating through the inner loop adaptation.
    /// This requires computing second-order derivatives (gradients of gradients).
    /// </para>
    /// <para><b>Algorithm:</b>
    /// 1. Record the adaptation trajectory (parameter updates during inner loop)
    /// 2. Compute query loss gradient at adapted parameters
    /// 3. Backpropagate through each adaptation step using chain rule
    /// 4. Return gradient w.r.t. original (pre-adaptation) parameters
    /// </para>
    /// <para><b>For Beginners:</b>
    /// Full MAML asks: "If I change my starting point slightly, how does that affect
    /// my performance after adaptation?"
    ///
    /// This requires tracking not just how to improve on this task, but how changing
    /// the starting point would have changed the whole adaptation process.
    ///
    /// Example: If your starting point makes you learn faster, the second-order gradient
    /// captures that benefit. First-order approximations (like Reptile) miss this.
    /// </para>
    /// <para><b>Computational Cost:</b>
    /// Significantly more expensive than first-order MAML:
    /// - Requires storing intermediate computations from adaptation
    /// - Performs additional backward passes through adaptation steps
    /// - Memory scales with number of adaptation steps
    ///
    /// Use only when accuracy is more important than speed.
    /// </para>
    /// </remarks>
    /// <exception cref="System.ArgumentNullException">If any parameter is null.</exception>
    /// <exception cref="System.ArgumentException">If adaptationSteps is empty.</exception>
    Vector<T> ComputeSecondOrderGradients(
        List<(TInput input, TOutput target)> adaptationSteps,
        TInput queryInput,
        TOutput queryTarget,
        ILossFunction<T> lossFunction,
        T innerLearningRate);
}
