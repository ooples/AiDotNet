namespace AiDotNet.InferenceOptimization.Core;

/// <summary>
/// Configuration options for graph optimization.
/// </summary>
public class OptimizationOptions
{
    /// <summary>
    /// The optimization level to apply.
    /// </summary>
    public OptimizationLevel Level { get; set; } = OptimizationLevel.Standard;

    /// <summary>
    /// Target layout for tensor operations (NCHW or NHWC).
    /// </summary>
    public string TargetLayout { get; set; } = "NCHW";

    /// <summary>
    /// Maximum number of optimization iterations.
    /// </summary>
    public int MaxIterations { get; set; } = 10;

    /// <summary>
    /// Enable operator fusion.
    /// </summary>
    public bool EnableOperatorFusion { get; set; } = true;

    /// <summary>
    /// Enable constant folding.
    /// </summary>
    public bool EnableConstantFolding { get; set; } = true;

    /// <summary>
    /// Enable dead code elimination.
    /// </summary>
    public bool EnableDeadCodeElimination { get; set; } = true;

    /// <summary>
    /// Enable common subexpression elimination.
    /// </summary>
    public bool EnableCSE { get; set; } = true;

    /// <summary>
    /// Enable layout optimization.
    /// </summary>
    public bool EnableLayoutOptimization { get; set; } = true;

    /// <summary>
    /// Enable in-place operations.
    /// </summary>
    public bool EnableInPlaceOptimization { get; set; } = true;

    /// <summary>
    /// Enable memory reuse optimization.
    /// </summary>
    public bool EnableMemoryReuse { get; set; } = true;

    /// <summary>
    /// Enable algebraic simplification.
    /// </summary>
    public bool EnableAlgebraicSimplification { get; set; } = true;

    /// <summary>
    /// Enable strength reduction.
    /// </summary>
    public bool EnableStrengthReduction { get; set; } = true;

    /// <summary>
    /// Print optimization statistics.
    /// </summary>
    public bool PrintStatistics { get; set; } = false;

    /// <summary>
    /// Validate graph after each pass.
    /// </summary>
    public bool ValidateAfterEachPass { get; set; } = false;

    /// <summary>
    /// Creates options based on optimization level.
    /// </summary>
    public static OptimizationOptions FromLevel(OptimizationLevel level)
    {
        var options = new OptimizationOptions { Level = level };

        switch (level)
        {
            case OptimizationLevel.None:
                options.DisableAllOptimizations();
                break;

            case OptimizationLevel.Basic:
                options.DisableAllOptimizations();
                options.EnableDeadCodeElimination = true;
                options.EnableConstantFolding = true;
                break;

            case OptimizationLevel.Standard:
                options.EnableOperatorFusion = true;
                options.EnableConstantFolding = true;
                options.EnableDeadCodeElimination = true;
                options.EnableAlgebraicSimplification = true;
                break;

            case OptimizationLevel.Aggressive:
                options.EnableOperatorFusion = true;
                options.EnableConstantFolding = true;
                options.EnableDeadCodeElimination = true;
                options.EnableCSE = true;
                options.EnableAlgebraicSimplification = true;
                options.EnableStrengthReduction = true;
                options.EnableInPlaceOptimization = true;
                options.EnableMemoryReuse = true;
                break;

            case OptimizationLevel.Maximum:
                // Enable everything
                options.EnableOperatorFusion = true;
                options.EnableConstantFolding = true;
                options.EnableDeadCodeElimination = true;
                options.EnableCSE = true;
                options.EnableLayoutOptimization = true;
                options.EnableInPlaceOptimization = true;
                options.EnableMemoryReuse = true;
                options.EnableAlgebraicSimplification = true;
                options.EnableStrengthReduction = true;
                break;
        }

        return options;
    }

    private void DisableAllOptimizations()
    {
        EnableOperatorFusion = false;
        EnableConstantFolding = false;
        EnableDeadCodeElimination = false;
        EnableCSE = false;
        EnableLayoutOptimization = false;
        EnableInPlaceOptimization = false;
        EnableMemoryReuse = false;
        EnableAlgebraicSimplification = false;
        EnableStrengthReduction = false;
    }
}
