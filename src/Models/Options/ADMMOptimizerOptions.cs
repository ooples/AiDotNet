namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the Alternating Direction Method of Multipliers (ADMM) optimization algorithm,
/// which is particularly effective for problems with complex regularization requirements.
/// </summary>
/// <remarks>
/// <para>
/// ADMM is an advanced optimization algorithm that breaks complex problems into smaller, more manageable subproblems.
/// It's especially useful for large-scale distributed optimization and problems with L1/L2 regularization.
/// </para>
/// <para><b>For Beginners:</b> ADMM is like solving a complex puzzle by breaking it into smaller pieces.
/// Instead of trying to solve everything at once, it tackles different parts of the problem separately and then
/// combines the solutions. This approach is particularly good when you want your AI model to be both accurate and simple
/// (avoiding unnecessary complexity). Think of it as a team of specialists working together rather than one person
/// trying to do everything.</para>
/// </remarks>
public class ADMMOptimizerOptions<T, TInput, TOutput> : GradientBasedOptimizerOptions<T, TInput, TOutput>
{
    /// <summary>
    /// Gets or sets the penalty parameter that controls the balance between the original objective and the constraint satisfaction.
    /// </summary>
    /// <value>The penalty parameter, defaulting to 1.0.</value>
    /// <remarks>
    /// <para>
    /// Rho affects how strongly the algorithm enforces constraints during optimization.
    /// Higher values prioritize constraint satisfaction over optimizing the original objective.
    /// </para>
    /// <para><b>For Beginners:</b> Rho is like a referee that decides how strictly to enforce the rules.
    /// A higher value (above 1.0) means the algorithm will focus more on following the constraints (rules)
    /// even if it means a slightly less optimal solution. A lower value allows more flexibility but might
    /// bend the rules too much. The default value of 1.0 provides a balanced approach for most problems.</para>
    /// </remarks>
    public double Rho { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the convergence tolerance that determines when the algorithm should stop.
    /// </summary>
    /// <value>The absolute tolerance, defaulting to 0.0001.</value>
    /// <remarks>
    /// <para>
    /// The algorithm stops when the error falls below this threshold, indicating that the solution is sufficiently accurate.
    /// </para>
    /// <para><b>For Beginners:</b> This is like deciding when a measurement is "close enough."
    /// If you're measuring ingredients for a recipe, you might be satisfied when you're within 0.0001 ounces of the target.
    /// Similarly, this value tells the algorithm when its solution is close enough to stop trying to improve further.
    /// A smaller value means higher precision but might take longer to achieve.</para>
    /// </remarks>
    public double AbsoluteTolerance { get; set; } = 1e-4;

    /// <summary>
    /// Gets or sets whether the algorithm should automatically adjust the Rho parameter during optimization.
    /// </summary>
    /// <value>True to use adaptive Rho (default), false to keep Rho constant.</value>
    /// <remarks>
    /// <para>
    /// When enabled, the algorithm will dynamically adjust the Rho parameter based on the relative magnitudes of primal and dual residuals.
    /// </para>
    /// <para><b>For Beginners:</b> This is like having an automatic transmission in a car versus a manual one.
    /// When set to true (the default), the algorithm will automatically adjust how strictly it enforces constraints
    /// as it progresses, similar to how an automatic transmission changes gears based on driving conditions.
    /// This usually leads to better performance without requiring you to fine-tune the Rho value yourself.</para>
    /// </remarks>
    public bool UseAdaptiveRho { get; set; } = true;

    /// <summary>
    /// Gets or sets the factor used to determine when to adjust the adaptive Rho value.
    /// </summary>
    /// <value>The adaptive Rho factor, defaulting to 10.0.</value>
    /// <remarks>
    /// <para>
    /// If the ratio of primal to dual residuals exceeds this factor, Rho will be adjusted to balance them.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how sensitive the "automatic transmission" is to changing conditions.
    /// A higher value (like the default 10.0) means the algorithm will only adjust Rho when there's a significant imbalance
    /// in how well different parts of the problem are being solved. It's like setting the threshold for when your car decides
    /// to shift gears - too sensitive and it shifts constantly, too insensitive and it stays in the wrong gear too long.</para>
    /// </remarks>
    public double AdaptiveRhoFactor { get; set; } = 10.0;

    /// <summary>
    /// Gets or sets the factor by which to increase Rho when primal residuals are much larger than dual residuals.
    /// </summary>
    /// <value>The increase factor, defaulting to 2.0.</value>
    /// <remarks>
    /// <para>
    /// When primal residuals are significantly larger than dual residuals, Rho is multiplied by this factor.
    /// </para>
    /// <para><b>For Beginners:</b> This determines how aggressively the algorithm increases the enforcement of constraints
    /// when needed. The default value of 2.0 means it will double the strictness when it detects that constraints aren't
    /// being satisfied well enough. Think of it like doubling your effort when you notice you're falling behind on a project.</para>
    /// </remarks>
    public double AdaptiveRhoIncrease { get; set; } = 2.0;

    /// <summary>
    /// Gets or sets the factor by which to decrease Rho when dual residuals are much larger than primal residuals.
    /// </summary>
    /// <value>The decrease factor, defaulting to 2.0.</value>
    /// <remarks>
    /// <para>
    /// When dual residuals are significantly larger than primal residuals, Rho is divided by this factor.
    /// </para>
    /// <para><b>For Beginners:</b> This determines how quickly the algorithm relaxes constraint enforcement when it's being
    /// too strict. The default value of 2.0 means it will reduce the strictness by half when it detects that it's
    /// focusing too much on constraints at the expense of the main objective. It's like easing up when you realize
    /// you're being too perfectionist about one aspect of a project.</para>
    /// </remarks>
    public double AdaptiveRhoDecrease { get; set; } = 2.0;

    /// <summary>
    /// Gets or sets the type of regularization to apply to the optimization problem.
    /// </summary>
    /// <value>The regularization type, defaulting to L1 regularization.</value>
    /// <remarks>
    /// <para>
    /// Regularization helps prevent overfitting by adding a penalty for model complexity.
    /// L1 promotes sparsity (more zero coefficients), L2 promotes smaller coefficients overall,
    /// and ElasticNet combines both approaches.
    /// </para>
    /// <para><b>For Beginners:</b> Regularization is like adding a budget constraint to your model.
    /// L1 (the default) is like having a limited number of features you can use - it forces the model to pick only
    /// the most important ones and set the rest to zero. L2 is like having a limited total "strength" to distribute
    /// among features - it makes all features smaller but keeps most of them. ElasticNet is a mix of both approaches.
    /// This helps prevent your model from becoming too complex and "memorizing" the training data instead of learning general patterns.</para>
    /// </remarks>
    public RegularizationType RegularizationType { get; set; } = RegularizationType.L1;

    /// <summary>
    /// Gets or sets the strength of the regularization penalty.
    /// </summary>
    /// <value>The regularization strength, defaulting to 0.1.</value>
    /// <remarks>
    /// <para>
    /// Higher values result in stronger regularization, which means simpler models with potentially lower accuracy on training data
    /// but better generalization to new data.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how strict your "budget constraint" is.
    /// A higher value forces the model to be simpler, which helps prevent overfitting but might miss some patterns in the data.
    /// A lower value allows more complexity, which can capture more patterns but risks memorizing noise in the training data.
    /// The default of 0.1 is a moderate value that works well for many problems, but you might need to adjust it
    /// based on how complex your data is and how much training data you have.</para>
    /// </remarks>
    public double RegularizationStrength { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the mixing parameter for ElasticNet regularization, balancing L1 and L2 penalties.
    /// </summary>
    /// <value>The mixing parameter, defaulting to 0.5.</value>
    /// <remarks>
    /// <para>
    /// Only used when RegularizationType is set to ElasticNet. A value of 1.0 corresponds to pure L1 regularization,
    /// while 0.0 corresponds to pure L2 regularization.
    /// </para>
    /// <para><b>For Beginners:</b> If you're using ElasticNet regularization (which combines L1 and L2),
    /// this controls the balance between them. The default value of 0.5 gives equal weight to both approaches.
    /// Values closer to 1.0 favor the L1 approach (selecting fewer features), while values closer to 0.0
    /// favor the L2 approach (making all features smaller). Think of it like adjusting the blend in a coffee mix
    /// between two different beans to get the flavor profile you want.</para>
    /// </remarks>
    public double ElasticNetMixing { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets the type of matrix decomposition to use when solving linear systems within the ADMM algorithm.
    /// </summary>
    /// <value>The decomposition type, defaulting to LU decomposition.</value>
    /// <remarks>
    /// <para>
    /// Different decomposition methods have different trade-offs in terms of numerical stability, memory usage, and computational efficiency.
    /// LU decomposition is a good general-purpose choice.
    /// </para>
    /// <para><b>For Beginners:</b> This is like choosing which mathematical tool to use for solving equations within the algorithm.
    /// The default (Lu) works well for most problems, similar to how a standard screwdriver works for most screws.
    /// Other options might be more efficient for specific types of problems, just like specialized tools can be better
    /// for specific tasks. Unless you have a specific reason to change this, the default is usually the best choice.</para>
    /// </remarks>
    public MatrixDecompositionType DecompositionType { get; set; } = MatrixDecompositionType.Lu;
}
