namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Powell's method, a derivative-free optimization algorithm used for finding
/// the minimum of a function without requiring gradient information.
/// </summary>
/// <remarks>
/// <para>
/// Powell's method is a powerful optimization technique that does not require gradient information, making it
/// suitable for optimizing functions where derivatives are unavailable, unreliable, or expensive to compute.
/// The algorithm works by performing a series of one-dimensional minimizations along different directions,
/// sequentially updating these directions based on the progress made. Unlike gradient-based methods like
/// gradient descent, Powell's method can navigate complex objective function landscapes even when the gradient
/// is not available. It is particularly effective for smooth, continuous functions with moderate dimensionality.
/// The method's efficiency and reliability can be significantly affected by the step size parameters and adaptation
/// strategy specified in this options class.
/// </para>
/// <para><b>For Beginners:</b> Powell's method is a way to find the lowest point (or minimum) of a function without needing to calculate its slope.
/// 
/// Imagine you're trying to find the lowest point in a hilly landscape while blindfolded:
/// - Gradient-based methods are like feeling which way is downhill and stepping that way
/// - But what if you can't tell which way is downhill? That's where Powell's method helps
/// 
/// What Powell's method does:
/// - It tries stepping in one direction, then another, then another
/// - It keeps track of which directions led to the most improvement
/// - It combines these successful directions to create new search directions
/// - It continues this process until it can't make further progress
/// 
/// Think of it like exploring a dark room:
/// - First, you walk north for a while until you hit something
/// - Then east, then south, then west
/// - Based on what you found, you decide which directions to try next
/// - You keep exploring until you're confident you've found what you're looking for
/// 
/// This class lets you configure how Powell's method explores the function landscape - how big its steps are,
/// how small they can get, and how it adapts as it explores.
/// </para>
/// </remarks>
public class PowellOptimizerOptions<T, TInput, TOutput> : OptimizationAlgorithmOptions<T, TInput, TOutput>
{
    /// <summary>
    /// Gets or sets the initial step size used by the algorithm when exploring the function space.
    /// </summary>
    /// <value>The initial step size, defaulting to 0.1.</value>
    /// <remarks>
    /// <para>
    /// This parameter determines the initial magnitude of steps taken by the algorithm during line searches.
    /// It represents a balance between exploration speed and precision. A larger initial step size allows
    /// the algorithm to explore the function space more quickly but may miss narrow valleys or detailed features.
    /// A smaller initial step size provides more detailed exploration but slows down the overall optimization process.
    /// The optimal value depends on the scale and characteristics of the objective function being optimized.
    /// Note that if adaptive step sizing is enabled, this value serves as the starting point for step size adjustment.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how big the steps are when the algorithm first starts searching.
    /// 
    /// The default value of 0.1 means:
    /// - The algorithm initially takes moderately-sized steps when exploring
    /// - It's a balance between speed and carefulness
    /// 
    /// Think of it like searching for something in the dark:
    /// - A small step size (0.01) is like taking tiny, cautious steps - safer but very slow
    /// - A large step size (1.0) is like taking big strides - faster but might miss important details
    /// - The default (0.1) is like normal walking - reasonably careful but not overly slow
    /// 
    /// You might want a larger initial step size (like 0.5 or 1.0):
    /// - When you're optimizing functions with broad, smooth landscapes
    /// - When you know the minimum is far from the starting point
    /// - When quick approximate solutions are more valuable than precise ones
    /// 
    /// You might want a smaller initial step size (like 0.01):
    /// - When your function has many narrow valleys or sharp features
    /// - When high precision is required from the beginning
    /// - When the minimum is expected to be close to the starting point
    /// </para>
    /// </remarks>
    public double InitialStepSize { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the minimum step size allowed during optimization, serving as a stopping criterion.
    /// </summary>
    /// <value>The minimum step size, defaulting to 1e-6 (0.000001).</value>
    /// <remarks>
    /// <para>
    /// This parameter establishes a lower bound on the step size used by the algorithm. When the step size
    /// falls below this threshold, the algorithm considers that it has converged to a solution with sufficient
    /// precision. It effectively controls the precision of the final solution - a smaller minimum step size 
    /// allows for more precise results but may require more iterations to reach convergence. This parameter
    /// is particularly important for preventing the algorithm from spending excessive computational resources
    /// trying to achieve marginal improvements beyond the required precision.
    /// </para>
    /// <para><b>For Beginners:</b> This setting defines how small the steps can get before the algorithm decides it's done.
    /// 
    /// The default value of 0.000001 (1e-6) means:
    /// - The algorithm will stop when its steps become extremely tiny
    /// - At this point, further improvements would be negligible
    /// 
    /// Think of it like finding a penny on the ground:
    /// - At first, you might walk with normal steps looking for it
    /// - As you get closer, you slow down and take smaller steps
    /// - Eventually, you're making such tiny movements that continuing to search isn't worth the effort
    /// - MinStepSize is like saying "when I'm moving less than a millimeter at a time, I'm done searching"
    /// 
    /// You might want a smaller minimum step size (like 1e-8 or 1e-10):
    /// - For applications requiring extremely high precision
    /// - When even tiny improvements in the solution are valuable
    /// - In scientific or engineering applications with strict accuracy requirements
    /// 
    /// You might want a larger minimum step size (like 1e-4 or 1e-3):
    /// - When approximate solutions are sufficient
    /// - To finish optimization more quickly
    /// - When the function being optimized doesn't warrant extreme precision
    /// </para>
    /// </remarks>
    public double MinStepSize { get; set; } = 1e-6;

    /// <summary>
    /// Gets or sets the maximum step size allowed during the optimization process.
    /// </summary>
    /// <value>The maximum step size, defaulting to 1.0.</value>
    /// <remarks>
    /// <para>
    /// This parameter establishes an upper bound on the step size used by the algorithm, preventing it from
    /// taking excessively large steps that might skip over important regions of the function space. Limiting
    /// the maximum step size is particularly important for functions with complex landscapes or multiple local
    /// minima, where large steps could lead to missing the global minimum. The value should be chosen based on
    /// the expected scale of the optimization domain and the characteristics of the objective function.
    /// When adaptive step sizing is enabled, this constraint ensures that the adaptation process doesn't
    /// produce counterproductively large steps.
    /// </para>
    /// <para><b>For Beginners:</b> This setting limits how large the algorithm's steps can become during optimization.
    /// 
    /// The default value of 1.0 means:
    /// - The algorithm will never take steps larger than 1.0
    /// - This prevents it from making huge jumps that might miss important features
    /// 
    /// Think of it like searching a room for a small object:
    /// - If you take giant leaps across the room, you might completely step over what you're looking for
    /// - The maximum step size puts a limit on how far you can jump in a single move
    /// - With adaptive step sizing, the algorithm might try to increase step size when making good progress
    /// - This parameter ensures those steps don't become too large
    /// 
    /// You might want a larger maximum step size (like 5.0 or 10.0):
    /// - When optimizing over very large parameter spaces
    /// - When the function is smooth and doesn't have small, important features
    /// - When you want to explore more quickly, even at the risk of reduced precision
    /// 
    /// You might want a smaller maximum step size (like 0.5 or 0.1):
    /// - When your function has many closely-spaced local minima
    /// - When you know the function has important small-scale features
    /// - When you want to ensure a thorough exploration of the space
    /// </para>
    /// </remarks>
    public double MaxStepSize { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets a value indicating whether the algorithm should adaptively adjust step sizes based on optimization progress.
    /// </summary>
    /// <value>A boolean indicating whether to use adaptive step sizing, defaulting to true.</value>
    /// <remarks>
    /// <para>
    /// This parameter controls whether the algorithm dynamically adjusts its step size based on progress.
    /// When enabled, the algorithm increases step size when making good progress (to speed up optimization)
    /// and decreases it when progress slows (to refine the search). Adaptive step sizing can significantly
    /// improve convergence rates and overall efficiency by balancing exploration (large steps) and exploitation
    /// (small steps) as needed throughout the optimization process. However, it adds some computational overhead
    /// and may introduce additional complexity in analyzing the algorithm's behavior.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines whether the algorithm automatically adjusts its step size as it searches.
    /// 
    /// The default value of true means:
    /// - The algorithm will dynamically change its step size during optimization
    /// - It takes larger steps when making good progress
    /// - It takes smaller steps when progress becomes difficult
    /// 
    /// Think of it like adjusting your pace when hiking:
    /// - On flat, clear terrain, you walk faster (larger steps)
    /// - On rocky, steep terrain, you slow down and step carefully (smaller steps)
    /// - This automatic adjustment helps you cover ground efficiently while still being careful when needed
    /// 
    /// You might want to disable adaptive step size (set to false):
    /// - When you need completely predictable algorithm behavior
    /// - When you're comparing different optimization runs and want consistent settings
    /// - When you've already fine-tuned the fixed step size for your specific problem
    /// 
    /// The adaptive step size is generally recommended as it:
    /// - Reduces the need to manually tune step size parameters
    /// - Often converges faster than fixed step sizes
    /// - Automatically adjusts to different regions of the function landscape
    /// </para>
    /// </remarks>
    public bool UseAdaptiveStepSize { get; set; } = true;

    /// <summary>
    /// Gets or sets the rate at which the step size adapts when adaptive step sizing is enabled.
    /// </summary>
    /// <value>The adaptation rate, defaulting to 0.1.</value>
    /// <remarks>
    /// <para>
    /// This parameter controls how quickly the step size changes when adaptive step sizing is enabled.
    /// A higher value causes more aggressive adaptation, with the step size changing substantially based
    /// on recent progress. A lower value results in more conservative adaptation, with the step size changing
    /// more gradually. The optimal value depends on the characteristics of the objective function. Too high
    /// an adaptation rate may cause instability in the optimization process, while too low a rate may negate
    /// the benefits of adaptive step sizing. This parameter is only relevant when UseAdaptiveStepSize is true.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how quickly the algorithm adjusts its step size when adaptation is enabled.
    /// 
    /// The default value of 0.1 means:
    /// - The step size adapts at a moderate rate
    /// - It finds a balance between responsive adaptation and stability
    /// 
    /// Think of it like learning to adjust your pace:
    /// - A high adaptation rate (0.5) is like dramatically changing your pace with every obstacle
    /// - A low adaptation rate (0.01) is like very gradually adjusting your pace over time
    /// - The default (0.1) gives noticeable but not dramatic adjustments
    /// 
    /// You might want a higher adaptation rate (like 0.3 or 0.5):
    /// - When your function has widely varying characteristics in different regions
    /// - When you want the algorithm to respond quickly to changes in progress
    /// - When convergence seems too slow with the default setting
    /// 
    /// You might want a lower adaptation rate (like 0.05 or 0.01):
    /// - When you notice oscillating behavior in the optimization process
    /// - When more stable, predictable step size changes are preferred
    /// - When the function landscape is relatively uniform
    /// 
    /// Note: This setting only has an effect when UseAdaptiveStepSize is set to true.
    /// </para>
    /// </remarks>
    public double AdaptationRate { get; set; } = 0.1;
}
