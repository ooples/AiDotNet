global using AiDotNet.Kernels;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the Bayesian optimization algorithm.
/// </summary>
/// <typeparam name="T">The data type used by the kernel function.</typeparam>
/// <remarks>
/// <para>
/// Bayesian optimization is a sequential design strategy for global optimization of black-box functions
/// that doesn't require derivatives. It's particularly useful for optimizing expensive-to-evaluate functions.
/// </para>
/// <para><b>For Beginners:</b> Bayesian optimization is like a smart search strategy. Imagine you're trying to find 
/// the highest point in a hilly landscape while blindfolded - you can only ask for the height at specific locations. 
/// Instead of checking every possible spot (which would take too long), Bayesian optimization uses what it learns from 
/// previous measurements to make educated guesses about where the highest point might be. It balances between exploring 
/// new areas and focusing on promising regions, making it efficient for finding optimal solutions when each evaluation 
/// is time-consuming or expensive.</para>
/// </remarks>
public class BayesianOptimizerOptions<T, TInput, TOutput> : OptimizationAlgorithmOptions<T, TInput, TOutput>
{
    /// <summary>
    /// Gets or sets the number of initial random samples to evaluate before starting the optimization process.
    /// </summary>
    /// <value>The number of initial samples, defaulting to 5.</value>
    /// <remarks>
    /// <para>
    /// These initial samples help build the first Gaussian Process model before the algorithm
    /// starts making informed decisions about where to sample next.
    /// </para>
    /// <para><b>For Beginners:</b> This is like taking a few random measurements in our hilly landscape example 
    /// before starting to make educated guesses. The default value of 5 means the algorithm will first check 5 random 
    /// locations to get a basic understanding of what the landscape looks like. Too few initial samples might give an 
    /// incomplete picture, while too many might waste evaluations on potentially uninteresting areas.</para>
    /// </remarks>
    public int InitialSamples { get; set; } = 5;

    /// <summary>
    /// Gets or sets the number of samples used when optimizing the acquisition function.
    /// </summary>
    /// <value>The number of acquisition optimization samples, defaulting to 1000.</value>
    /// <remarks>
    /// <para>
    /// The acquisition function helps decide where to sample next. This parameter controls how thoroughly
    /// the algorithm searches for the best next point to evaluate.
    /// </para>
    /// <para><b>For Beginners:</b> After taking some measurements, the algorithm needs to decide where to look next. 
    /// This parameter (default: 1000) controls how carefully it considers its options. Think of it like considering 1000 
    /// possible locations and picking the most promising one. A higher number means more thorough consideration but takes 
    /// longer to compute. For most problems, the default value provides a good balance between computation time and 
    /// optimization quality.</para>
    /// </remarks>
    public int AcquisitionOptimizationSamples { get; set; } = 1000;

    /// <summary>
    /// Gets or sets the lower bound for the search space.
    /// </summary>
    /// <value>The lower bound value, defaulting to -10.</value>
    /// <remarks>
    /// <para>
    /// This value defines the minimum value for each dimension in the search space.
    /// </para>
    /// <para><b>For Beginners:</b> This sets the minimum value the algorithm will consider for each variable 
    /// (default: -10). In our landscape analogy, this would be like setting the western and southern boundaries of 
    /// the area you're willing to explore. You should set this based on what makes sense for your specific problem - 
    /// for example, if you're optimizing a learning rate that can't be negative, you might set this to 0 instead.</para>
    /// </remarks>
    public double LowerBound { get; set; } = -10;

    /// <summary>
    /// Gets or sets the upper bound for the search space.
    /// </summary>
    /// <value>The upper bound value, defaulting to 10.</value>
    /// <remarks>
    /// <para>
    /// This value defines the maximum value for each dimension in the search space.
    /// </para>
    /// <para><b>For Beginners:</b> This sets the maximum value the algorithm will consider for each variable 
    /// (default: 10). Continuing our landscape analogy, this would be like setting the eastern and northern boundaries 
    /// of your search area. Setting appropriate bounds helps the algorithm focus on realistic values and can significantly 
    /// improve optimization efficiency.</para>
    /// </remarks>
    public double UpperBound { get; set; } = 10;

    /// <summary>
    /// Gets or sets the exploration factor (kappa) used in the acquisition function.
    /// </summary>
    /// <value>The exploration factor, defaulting to 2.0.</value>
    /// <remarks>
    /// <para>
    /// This parameter balances exploration (searching new areas) versus exploitation (refining known good areas).
    /// Higher values favor exploration, while lower values favor exploitation.
    /// </para>
    /// <para><b>For Beginners:</b> This controls the balance between trying new areas (exploration) and focusing on 
    /// promising areas already discovered (exploitation). The default value of 2.0 provides a balanced approach. Think of 
    /// it like deciding whether to visit new restaurants or return to ones you already know are good. A higher value means 
    /// more adventurous exploration, while a lower value means sticking closer to what's already known to work well.</para>
    /// </remarks>
    public double ExplorationFactor { get; set; } = 2.0;

    /// <summary>
    /// Gets or sets the type of acquisition function to use.
    /// </summary>
    /// <value>The acquisition function type, defaulting to UpperConfidenceBound.</value>
    /// <remarks>
    /// <para>
    /// The acquisition function determines the strategy for selecting the next point to evaluate.
    /// Common options include Upper Confidence Bound (UCB), Expected Improvement (EI), and Probability of Improvement (PI).
    /// </para>
    /// <para><b>For Beginners:</b> This determines the strategy the algorithm uses to decide where to look next. 
    /// The default (UpperConfidenceBound) balances exploring uncertain areas and exploiting promising ones. Think of it like:
    /// <list type="bullet">
    ///   <item>Upper Confidence Bound: "I'll check places that might be good or that I'm uncertain about"</item>
    ///   <item>Expected Improvement: "I'll check places likely to be better than the best I've found so far"</item>
    ///   <item>Probability of Improvement: "I'll check places most likely to be at least slightly better than my current best"</item>
    /// </list>
    /// Different strategies work better for different problems, but the default is a good starting point.</para>
    /// </remarks>
    public AcquisitionFunctionType AcquisitionFunction { get; set; } = AcquisitionFunctionType.UpperConfidenceBound;

    /// <summary>
    /// Gets or sets the kernel function used by the Gaussian Process model.
    /// </summary>
    /// <value>The kernel function, defaulting to Gaussian kernel (also known as Radial Basis Function kernel).</value>
    /// <remarks>
    /// <para>
    /// The kernel function determines how the algorithm measures similarity between points in the search space,
    /// which affects how it generalizes from observed data points to unobserved points.
    /// </para>
    /// <para><b>For Beginners:</b> The kernel function helps the algorithm understand how similar different points are 
    /// to each other. The default Gaussian kernel (also called Radial Basis Function kernel) assumes that points close to each other will 
    /// have similar values, with the similarity decreasing smoothly as distance increases. This is like assuming that in 
    /// our hilly landscape, nearby locations tend to have similar heights. The Gaussian kernel works well for many problems, 
    /// especially when the underlying function is smooth.</para>
    /// </remarks>
    
    public IKernel<T>? Kernel { get; set; } = new GaussianKernel<T>();

    /// <summary>
    /// Gets or sets whether the objective should be maximized (true) or minimized (false).
    /// </summary>
    public bool IsMaximization { get; set; } = true;
}
