namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the iMAML (implicit MAML) algorithm.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix<T>, Tensor<T>).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector<T>, Tensor<T>).</typeparam>
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
public class iMAMLAlgorithmOptions<T, TInput, TOutput> : MetaLearningAlgorithmOptions<T, TInput, TOutput>
{
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
    /// <value>The number of CG iterations, defaulting to 5.</value>
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
    public int ConjugateGradientIterations { get; set; } = 5;

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
}
