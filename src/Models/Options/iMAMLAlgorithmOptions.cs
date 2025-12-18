namespace AiDotNet.Models.Options;

/// <summary>
/// Methods for computing Hessian-vector products in iMAML.
/// </summary>
public enum HessianVectorProductMethod
{
    /// <summary>
    /// Finite differences approximation using two gradient evaluations.
    /// </summary>
    FiniteDifferences,

    /// <summary>
    /// Automatic differentiation using Pearlmutter's algorithm.
    /// </summary>
    AutomaticDifferentiation,

    /// <summary>
    /// Both methods available with configuration option.
    /// </summary>
    Both
}

/// <summary>
/// Preconditioning methods for the Conjugate Gradient solver.
/// </summary>
public enum CGPreconditioningMethod
{
    /// <summary>
    /// No preconditioning - vanilla CG.
    /// </summary>
    None,

    /// <summary>
    /// Jacobi preconditioning using diagonal of the matrix.
    /// </summary>
    Jacobi,

    /// <summary>
    /// Limited-memory BFGS preconditioning.
    /// </summary>
    LBFGS
}

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
    /// <value>The number of CG iterations, defaulting to 10.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> iMAML needs to solve a system of equations to compute gradients.
    /// Conjugate Gradient is an iterative method for solving these equations.
    ///
    /// - More iterations mean more accurate solutions but take longer
    /// - 10 iterations is typically sufficient for most problems
    /// - If training is unstable, try increasing this
    ///
    /// Think of it like refining an answer - more iterations give you a more precise answer.
    /// </para>
    /// </remarks>
    public int ConjugateGradientIterations { get; set; } = 10;

    /// <summary>
    /// Gets or sets the tolerance for CG convergence.
    /// </summary>
    /// <value>The CG tolerance, defaulting to 1e-8.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This determines when the equation solver decides it's "close enough."
    /// - Smaller values require more precision (more iterations)
    /// - Larger values allow faster but less precise solutions
    /// - 1e-8 is a good balance of precision and speed
    ///
    /// It's like deciding how many decimal places you need in your answer.
    /// </para>
    /// </remarks>
    public double ConjugateGradientTolerance { get; set; } = 1e-8;

    /// <summary>
    /// Gets or sets the method for computing Hessian-vector products.
    /// </summary>
    /// <value>The Hvp computation method, defaulting to FiniteDifferences.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> To compute implicit gradients, iMAML needs to multiply vectors
    /// by the Hessian matrix (second derivatives). This parameter controls how that's done.
    /// </para>
    /// </remarks>
    public HessianVectorProductMethod HessianVectorProductMethod { get; set; } = HessianVectorProductMethod.FiniteDifferences;

    /// <summary>
    /// Gets or sets the preconditioning method for the CG solver.
    /// </summary>
    /// <value>The preconditioning method, defaulting to Jacobi.</value>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Preconditioning helps the CG solver converge faster by
    /// transforming the problem to be better-conditioned.
    /// </para>
    /// </remarks>
    public CGPreconditioningMethod CGPreconditioningMethod { get; set; } = CGPreconditioningMethod.Jacobi;

    /// <summary>
    /// Gets or sets the finite differences epsilon for Hvp computation.
    /// </summary>
    /// <value>The epsilon value, defaulting to 1e-5.</value>
    /// <remarks>
    /// <para>
    /// Small epsilon = more accurate but numerically unstable
    /// Large epsilon = more stable but less accurate
    /// </para>
    /// </remarks>
    public double FiniteDifferencesEpsilon { get; set; } = 1e-5;

    /// <summary>
    /// Gets or sets whether to use adaptive learning rates in inner loop.
    /// </summary>
    /// <value>True to use adaptive learning rates, defaulting to true.</value>
    /// <remarks>
    /// <para>
    /// When enabled, uses Adam-style adaptive learning rates for inner loop adaptation.
    /// This improves stability and convergence of the adaptation process.
    /// </para>
    /// </remarks>
    public bool UseAdaptiveInnerLearningRate { get; set; } = true;

    /// <summary>
    /// Gets or sets the minimum inner learning rate for adaptive schemes.
    /// </summary>
    /// <value>The minimum learning rate, defaulting to 1e-6.</value>
    /// <remarks>
    /// <para>
    /// Prevents learning rate from becoming too small during adaptive updates.
    /// </para>
    /// </remarks>
    public double MinInnerLearningRate { get; set; } = 1e-6;

    /// <summary>
    /// Gets or sets the maximum inner learning rate for adaptive schemes.
    /// </summary>
    /// <value>The maximum learning rate, defaulting to 0.1.</value>
    /// <remarks>
    /// <para>
    /// Prevents learning rate from becoming too large during adaptive updates.
    /// </para>
    /// </remarks>
    public double MaxInnerLearningRate { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets whether to enable line search in inner loop adaptation.
    /// </summary>
    /// <value>True to enable line search, defaulting to false.</value>
    /// <remarks>
    /// <para>
    /// Line search finds the optimal step size along the gradient direction.
    /// Adds computational overhead but can improve convergence.
    /// </para>
    /// </remarks>
    public bool EnableLineSearch { get; set; } = false;

    /// <summary>
    /// Gets or sets the line search step reduction factor.
    /// </summary>
    /// <value>The reduction factor, defaulting to 0.5.</value>
    /// <remarks>
    /// <para>
    /// When line search fails to reduce loss, the step size is multiplied by this factor.
    /// </para>
    /// </remarks>
    public double LineSearchReduction { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets the line search minimum step size.
    /// </summary>
    /// <value>The minimum step size, defaulting to 1e-10.</value>
    /// <remarks>
    /// <para>
    /// Line search stops if step size becomes smaller than this value.
    /// </para>
    /// </remarks>
    public double LineSearchMinStep { get; set; } = 1e-10;

    /// <summary>
    /// Gets or sets the line search maximum iterations.
    /// </summary>
    /// <value>The maximum iterations, defaulting to 20.</value>
    /// <remarks>
    /// <para>
    /// Maximum number of attempts to find a suitable step size.
    /// </para>
    /// </remarks>
    public int LineSearchMaxIterations { get; set; } = 20;
}
