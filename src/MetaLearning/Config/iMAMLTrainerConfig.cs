using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.MetaLearning.Config;

/// <summary>
/// Configuration options for the iMAML (implicit MAML) meta-learning algorithm.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// iMAML (implicit MAML) is an extension of MAML that uses implicit differentiation
/// to compute meta-gradients more efficiently. Instead of backpropagating through
/// all adaptation steps, it uses the implicit function theorem to compute gradients
/// directly at the adapted parameters.
/// </para>
/// <para>
/// <b>For Beginners:</b> iMAML is a more efficient version of MAML.
///
/// Regular MAML problem:
/// - To learn from adaptation, you need to remember every step of the adaptation process
/// - This requires a lot of memory and computation
///
/// iMAML solution:
/// - Uses a mathematical trick (implicit differentiation) to skip remembering all steps
/// - Just looks at where you started and where you ended up
/// - Much more memory-efficient, allowing deeper adaptation
///
/// The trade-off: slightly more complex math, but same or better results with less memory.
/// </para>
/// </remarks>
public class iMAMLTrainerConfig<T> : MetaLearnerConfig<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <inheritdoc/>
    public T InnerLearningRate { get; set; } = NumOps.FromDouble(0.01);

    /// <inheritdoc/>
    public T MetaLearningRate { get; set; } = NumOps.FromDouble(0.001);

    /// <inheritdoc/>
    public int InnerSteps { get; set; } = 5;

    /// <inheritdoc/>
    public int MetaBatchSize { get; set; } = 4;

    /// <inheritdoc/>
    public int NumMetaIterations { get; set; } = 1000;

    /// <summary>
    /// Gets or sets the regularization strength for implicit gradients.
    /// </summary>
    /// <value>The lambda regularization parameter, defaulting to 1.0.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This parameter helps stabilize the implicit gradient computation.
    /// - Higher values make the computation more stable but less accurate
    /// - Lower values are more accurate but might be unstable
    /// - 1.0 is a good default that balances stability and accuracy
    ///
    /// Think of it like adding a safety margin to ensure the math stays numerically stable.
    /// </para>
    /// </remarks>
    public double LambdaRegularization { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the number of CG (Conjugate Gradient) iterations for solving implicit equations.
    /// </summary>
    /// <value>The number of CG iterations, defaulting to 20.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> iMAML needs to solve a system of equations to compute gradients.
    /// Conjugate Gradient is an iterative method for solving these equations.
    ///
    /// - More iterations mean more accurate solutions but take longer
    /// - 5-10 iterations is typically sufficient
    /// - If training is unstable, try increasing this
    ///
    /// Think of it like refining an answer - more iterations give you a more precise answer.
    /// </para>
    /// </remarks>
    public int ConjugateGradientIterations { get; set; } = 20;

    /// <summary>
    /// Gets or sets the tolerance for CG convergence.
    /// </summary>
    /// <value>The CG tolerance, defaulting to 1e-10.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This determines when the equation solver decides it's "close enough."
    /// - Smaller values require more precision (more iterations)
    /// - Larger values allow faster but less precise solutions
    /// - 1e-10 is very precise; 1e-6 would be faster but less accurate
    ///
    /// It's like deciding how many decimal places you need in your answer.
    /// </para>
    /// </remarks>
    public double ConjugateGradientTolerance { get; set; } = 1e-10;

    /// <summary>
    /// Gets or sets the maximum gradient norm for gradient clipping.
    /// </summary>
    /// <value>The maximum allowed gradient norm. Set to 0 or negative to disable. Default is 10.0.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Gradient clipping prevents training instability.
    ///
    /// During meta-learning, gradients can sometimes become very large, causing:
    /// - Unstable training (parameters jumping around)
    /// - Numerical overflow (infinity/NaN values)
    /// - Poor convergence
    ///
    /// Gradient clipping limits how large gradients can be:
    /// - If gradients are too large, scale them down proportionally
    /// - This stabilizes training without changing the direction
    /// - Common in production meta-learning systems
    ///
    /// Recommended values:
    /// - 10.0 (default) for most tasks
    /// - 1.0 for very sensitive tasks
    /// - 0 to disable (only if training is already stable)
    /// </para>
    /// </remarks>
    public T MaxGradientNorm { get; set; } = NumOps.FromDouble(10.0);

    /// <summary>
    /// Gets or sets whether to use adaptive meta-optimizer (Adam) for the outer loop.
    /// </summary>
    /// <value>If true, uses Adam for meta-updates. Default is true.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Adaptive optimizers like Adam adjust learning rates automatically.
    ///
    /// Vanilla SGD uses the same learning rate for all parameters:
    /// - Simple but can be slow to converge
    /// - Sensitive to learning rate choice
    ///
    /// Adam adjusts learning rates per parameter:
    /// - Remembers recent gradients (momentum)
    /// - More robust to learning rate choice
    /// - Industry standard for deep learning
    /// - Recommended for meta-learning
    ///
    /// This is the meta-optimizer that updates the meta-parameters, not the inner loop adaptation.
    /// </para>
    /// </remarks>
    public bool UseAdaptiveMetaOptimizer { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to use first-order approximation.
    /// </summary>
    /// <value>If true, uses FOMAML approximation. Default is true for efficiency.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is a speed vs accuracy trade-off.
    ///
    /// <b>First-order iMAML (default):</b>
    /// - Faster training (ignores second-order derivatives)
    /// - Uses less memory
    /// - Still very effective in practice
    /// - Recommended for most use cases
    ///
    /// <b>Full iMAML:</b>
    /// - More computationally expensive
    /// - Computes gradients through gradient computations
    /// - Theoretically more accurate
    /// - Use only if you need maximum performance
    ///
    /// Even with first-order approximation, iMAML often performs as well as or better than MAML
    /// while being much more memory-efficient.
    /// </para>
    /// </remarks>
    public bool UseFirstOrderApproximation { get; set; } = true;

    /// <inheritdoc/>
    public bool IsValid()
    {
        return InnerLearningRate != null &&
               MetaLearningRate != null &&
               InnerSteps > 0 &&
               MetaBatchSize > 0 &&
               NumMetaIterations > 0 &&
               LambdaRegularization >= 0 &&
               ConjugateGradientIterations > 0 &&
               ConjugateGradientTolerance > 0 &&
               ConjugateGradientTolerance < 1.0 &&
               !double.IsNaN(ConjugateGradientTolerance);
    }
}