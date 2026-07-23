namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the CMA-ES (Covariance Matrix Adaptation Evolution Strategy) optimization algorithm.
/// </summary>
/// <typeparam name="T">The type of data being optimized.</typeparam>
/// <remarks>
/// <para>
/// CMA-ES is a powerful evolutionary algorithm for difficult non-linear, non-convex optimization problems.
/// It adapts its search strategy during optimization by learning dependencies between variables and
/// step sizes, making it effective for complex problems where other methods might fail.
/// </para>
/// <para><b>For Beginners:</b> CMA-ES is like a smart search algorithm that tries to find the best solution 
/// to a complex problem. Imagine you're looking for the lowest point in a mountain range with fog everywhere - 
/// you can only see what's right around you. CMA-ES works by sending out multiple "scouts" (solutions) in different 
/// directions, then learning from their findings to decide where to search next. It's particularly good at handling 
/// tricky landscapes where simple "always go downhill" approaches would get stuck. This class lets you configure 
/// how that search process works.</para>
/// </remarks>
public class CMAESOptimizerOptions<T, TInput, TOutput> : GeneticAlgorithmOptimizerOptions<T, TInput, TOutput>
{
    /// <summary>
    /// Gets or sets the number of candidate solutions evaluated in each generation.
    /// </summary>
    /// <value>
    /// The population size, defaulting to a formula based on problem dimensionality: 4 + (int)(3 * Math.Log(100)).
    /// </value>
    /// <remarks>
    /// <para>
    /// The population size affects both the quality of solutions and computational cost.
    /// Larger populations provide more exploration but require more function evaluations.
    /// The default formula (4 + 3*ln(D)) is a common heuristic where D is the problem dimension.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how many different potential solutions the algorithm 
    /// tries in each round (or "generation"). The default value is calculated based on a formula that works well 
    /// for most problems. Think of it like sending out search parties - more parties can cover more ground but 
    /// require more resources. For simple problems, you might reduce this value to speed things up, while for 
    /// complex problems with many variables, you might increase it to find better solutions.</para>
    /// </remarks>
    public new int PopulationSize { get; set; } = 4 + (int)(3 * Math.Log(100)); // Default assuming 100 dimensions

    /// <summary>
    /// Gets or sets the initial step size that controls how far the algorithm explores from the starting point.
    /// </summary>
    /// <value>The initial step size as a decimal, defaulting to 0.5.</value>
    /// <remarks>
    /// <para>
    /// The step size determines the initial spread of the search distribution.
    /// A larger value encourages more exploration early in the search process,
    /// while a smaller value focuses on exploitation near the starting point.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines how big the initial "steps" are when the algorithm 
    /// starts searching. The default (0.5) means it will look at solutions that are moderately different from 
    /// your starting point. If you set this higher (like 2.0), it will explore more widely at first, which is 
    /// good if you think the best solution might be far from your starting point. If you set it lower (like 0.1), 
    /// it will focus more closely around your starting point, which works well if you're already close to the 
    /// optimal solution.</para>
    /// </remarks>
    public double InitialStepSize { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets the maximum number of generations (iterations) the algorithm will run.
    /// </summary>
    /// <value>The maximum number of generations, defaulting to 100.</value>
    /// <remarks>
    /// <para>
    /// This parameter limits the total computational budget of the optimization process.
    /// The algorithm will stop either when it converges or when it reaches this number of generations.
    /// </para>
    /// <para><b>For Beginners:</b> This is simply how many rounds of optimization the algorithm will perform 
    /// before stopping. Each generation involves creating and evaluating multiple solutions, then learning from 
    /// the results to improve the next generation. The default (100) is enough for many problems, but complex 
    /// problems might need more generations to find good solutions. If the algorithm finds an excellent solution 
    /// earlier, it may stop before reaching this limit.</para>
    /// </remarks>
    public new int MaxGenerations { get; set; } = 100;

    /// <summary>
    /// Gets or sets the convergence threshold that determines when the algorithm should stop.
    /// </summary>
    /// <value>The stop tolerance as a decimal, defaulting to 1e-12 (0.000000000001).</value>
    /// <remarks>
    /// <para>
    /// The algorithm stops when the change in the best solution falls below this threshold,
    /// indicating that further optimization is unlikely to yield significant improvements.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines how precise the algorithm needs to be before it 
    /// decides it's found a good enough solution and stops. The default value (0.000000000001) is very small, 
    /// meaning the algorithm will keep going until it's making virtually no progress. Think of it like deciding 
    /// when to stop polishing a surface - at some point, more polishing doesn't make a noticeable difference. 
    /// You might increase this value (making it less strict) if you want faster results and don't need the 
    /// absolute best solution.</para>
    /// </remarks>
    public double StopTolerance { get; set; } = 1e-12;
}
