namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the Nelder-Mead optimization algorithm, a derivative-free method
/// for finding the minimum of an objective function in a multidimensional space.
/// </summary>
/// <remarks>
/// <para>
/// The Nelder-Mead method (also known as the downhill simplex method or amoeba method) is a numerical
/// optimization technique that does not require derivative information. It works by constructing a simplex
/// (a generalization of a triangle to higher dimensions) and systematically replacing the worst vertex
/// of this simplex with a new point through a series of reflection, expansion, contraction, and shrinking
/// operations. This approach is particularly useful for problems where the objective function is non-differentiable,
/// is not known in closed form, or when gradient calculations are computationally expensive or unstable.
/// </para>
/// <para><b>For Beginners:</b> The Nelder-Mead algorithm is a clever way to find the minimum value of a function
/// without needing to calculate derivatives (which can be complicated or impossible for some problems).
/// 
/// Imagine you're trying to find the lowest point in a hilly landscape:
/// - Instead of following the steepest downhill path (as gradient-based methods do)
/// - Nelder-Mead places a "flexible shape" on the landscape (like a triangle in 2D space)
/// - It then moves and reshapes this triangle, always trying to "slide" toward lower ground
/// 
/// The algorithm works through a series of simple operations:
/// - Reflection: Try moving away from the highest point
/// - Expansion: If that works well, try moving even further
/// - Contraction: If reflection doesn't work, try moving a shorter distance
/// - Shrinking: If all else fails, make the triangle smaller and try again
/// 
/// This class allows you to configure how aggressively or cautiously these operations are performed,
/// which affects how quickly and reliably the algorithm can find the optimal solution.
/// </para>
/// </remarks>
public class NelderMeadOptimizerOptions<T, TInput, TOutput> : OptimizationAlgorithmOptions<T, TInput, TOutput>
{
    /// <summary>
    /// Gets or sets the initial reflection coefficient, which controls how far to reflect the simplex away from the worst point.
    /// </summary>
    /// <value>The initial reflection coefficient, defaulting to 1.0.</value>
    /// <remarks>
    /// <para>
    /// The alpha parameter controls the distance to which the simplex is reflected away from the worst point.
    /// Specifically, when the algorithm identifies the worst vertex, it computes the centroid of all other vertices
    /// and then reflects the worst vertex through this centroid by a distance proportional to alpha. The standard value
    /// of 1.0 creates a reflection point that is the same distance from the centroid as the worst point, but in the
    /// opposite direction. Larger values create more aggressive reflections, while smaller values create more
    /// conservative reflections.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how far the algorithm tries to move when
    /// it's "reflecting" away from a bad point.
    /// 
    /// In our landscape analogy:
    /// - When the algorithm finds that one corner of its triangle is on high ground
    /// - It wants to move that corner to potentially lower ground
    /// - Alpha determines how far it jumps in the opposite direction
    /// 
    /// The default value of 1.0 means:
    /// - It will try moving exactly as far away from the bad point as the bad point was from the center
    /// - This creates a balanced exploration of the space
    /// 
    /// You might want to increase this value if:
    /// - Your function has large flat regions that need to be crossed quickly
    /// - You want more aggressive exploration early in the optimization
    /// 
    /// You might want to decrease this value if:
    /// - Your function has many sharp peaks and valleys
    /// - The algorithm seems to be overshooting good solutions
    /// 
    /// Think of it like deciding how big of a step to take when moving away from a known bad area -
    /// larger steps cover more ground but might miss details.
    /// </para>
    /// </remarks>
    public double InitialAlpha { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the initial contraction coefficient, which controls how far to contract the simplex toward the centroid.
    /// </summary>
    /// <value>The initial contraction coefficient, defaulting to 0.5.</value>
    /// <remarks>
    /// <para>
    /// The beta parameter controls the contraction operation, which is performed when the reflection yields a point
    /// that is not better than the second worst point. In this case, the algorithm tries to contract the simplex by moving
    /// either the reflected point or the worst point (depending on which is better) toward the centroid of the remaining
    /// points. The contraction coefficient determines what fraction of the distance to the centroid is used. Smaller values
    /// create more aggressive contractions, moving closer to the centroid.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how cautiously the algorithm approaches
    /// a potentially promising area by "contracting" toward it.
    /// 
    /// Continuing our landscape analogy:
    /// - If the algorithm reflects away from a bad point but doesn't find better ground
    /// - It tries contracting toward the center of the good points instead
    /// - Beta determines how far along this path it moves
    /// 
    /// The default value of 0.5 means:
    /// - It will try a point halfway between the worst point and the center of the other points
    /// - This creates a moderate approach that doesn't commit too strongly
    /// 
    /// You might want to increase this value (closer to 1.0) if:
    /// - The algorithm seems to be contracting too aggressively
    /// - You want more conservative exploration of the space
    /// 
    /// You might want to decrease this value (closer to 0.0) if:
    /// - You want to approach promising areas more aggressively
    /// - The function has narrow valleys that need precise localization
    /// 
    /// Think of it like carefully approaching what might be a good spot - a smaller beta means
    /// moving more directly toward the center of what looks promising.
    /// </para>
    /// </remarks>
    public double InitialBeta { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets the initial expansion coefficient, which controls how far to expand the simplex in a promising direction.
    /// </summary>
    /// <value>The initial expansion coefficient, defaulting to 2.0.</value>
    /// <remarks>
    /// <para>
    /// The gamma parameter controls the expansion operation, which is performed when the reflection yields a new best point.
    /// In this case, the algorithm tries to expand further in this promising direction by moving even farther from the centroid
    /// than the reflection point. The expansion coefficient determines how much farther to go, as a multiple of the reflection
    /// distance. Values greater than 1.0 result in points that are farther from the centroid than the reflection point, with
    /// larger values creating more aggressive expansions.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how far the algorithm will try to move when
    /// it discovers a promising direction.
    /// 
    /// In our landscape analogy:
    /// - If reflecting away from a bad point discovers significantly lower ground
    /// - The algorithm gets excited and wants to try moving even further in that direction
    /// - Gamma determines how much further it tries to go
    /// 
    /// The default value of 2.0 means:
    /// - If reflection finds a good point, try going twice as far in that direction
    /// - This allows for rapid progress when a good direction is found
    /// 
    /// You might want to increase this value if:
    /// - Your function has long, steep slopes that can be descended quickly
    /// - You want more aggressive exploration when promising directions are found
    /// 
    /// You might want to decrease this value if:
    /// - Your function has narrow valleys where the minimum could be easily overshot
    /// - The optimization seems to bounce around too much
    /// 
    /// Think of it like deciding how enthusiastically to follow a promising path - larger values
    /// mean taking bigger leaps of faith when things look good.
    /// </para>
    /// </remarks>
    public double InitialGamma { get; set; } = 2.0;

    /// <summary>
    /// Gets or sets the initial shrink coefficient, which controls how much to shrink the entire simplex when other operations fail.
    /// </summary>
    /// <value>The initial shrink coefficient, defaulting to 0.5.</value>
    /// <remarks>
    /// <para>
    /// The delta parameter controls the shrinking operation, which is performed when reflection, expansion, and contraction
    /// all fail to improve the worst vertex. In this case, the algorithm shrinks the entire simplex toward the best vertex,
    /// moving all vertices except the best one closer to it. The shrink coefficient determines what fraction of the distance
    /// to the best vertex is preserved. Smaller values create more aggressive shrinking, making the simplex smaller and
    /// concentrating the search in a smaller region.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how much the algorithm shrinks its search area
    /// when it can't find a better solution through other means.
    /// 
    /// In our landscape analogy:
    /// - If none of the other strategies (reflection, expansion, contraction) work
    /// - The algorithm needs to make its triangle smaller to focus on more detailed exploration
    /// - Delta determines how much smaller the triangle becomes
    /// 
    /// The default value of 0.5 means:
    /// - All points (except the best one) move halfway toward the best point
    /// - This creates a smaller triangle focused around the most promising area
    /// 
    /// You might want to increase this value (closer to 1.0) if:
    /// - The algorithm seems to be shrinking too aggressively and getting stuck
    /// - You want to maintain a broader search even when progress is difficult
    /// 
    /// You might want to decrease this value if:
    /// - You want faster convergence when the algorithm is near the solution
    /// - Your function has very complex local structure that requires detailed exploration
    /// 
    /// Think of it like zooming in on a promising area - a smaller delta means zooming in more
    /// aggressively when the algorithm needs to focus.
    /// </para>
    /// </remarks>
    public double InitialDelta { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets the minimum allowed value for the reflection coefficient when using adaptive parameters.
    /// </summary>
    /// <value>The minimum reflection coefficient, defaulting to 0.1.</value>
    /// <remarks>
    /// <para>
    /// When using adaptive parameters (UseAdaptiveParameters = true), this value sets a lower bound on how small
    /// the reflection coefficient (alpha) can become. This prevents the reflection operations from becoming too
    /// conservative, which could slow down progress. The appropriate minimum value depends on the characteristics
    /// of the objective function and the desired balance between exploration and exploitation.
    /// </para>
    /// <para><b>For Beginners:</b> This setting establishes the minimum value for Alpha when
    /// the algorithm is automatically adjusting its parameters.
    /// 
    /// If you enable adaptive parameters (UseAdaptiveParameters = true):
    /// - The algorithm will adjust Alpha based on its success or failure
    /// - This setting prevents Alpha from becoming too small
    /// - It ensures the algorithm maintains some minimum level of exploration
    /// 
    /// The default value of 0.1 means:
    /// - Even if reflections repeatedly fail, Alpha won't go below 0.1
    /// - This prevents the algorithm from becoming too timid in its exploration
    /// 
    /// You would typically only change this if you're using adaptive parameters and:
    /// - You want to allow for even more conservative reflections (lower value)
    /// - You want to ensure more aggressive minimum exploration (higher value)
    /// 
    /// This setting has no effect if UseAdaptiveParameters is false.
    /// </para>
    /// </remarks>
    public double MinAlpha { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the maximum allowed value for the reflection coefficient when using adaptive parameters.
    /// </summary>
    /// <value>The maximum reflection coefficient, defaulting to 2.0.</value>
    /// <remarks>
    /// <para>
    /// When using adaptive parameters (UseAdaptiveParameters = true), this value sets an upper bound on how large
    /// the reflection coefficient (alpha) can become. This prevents the reflection operations from becoming too
    /// aggressive, which could cause the algorithm to overlook detailed structure in the objective function. The
    /// appropriate maximum value depends on the characteristics of the objective function and the desired balance
    /// between exploration and exploitation.
    /// </para>
    /// <para><b>For Beginners:</b> This setting establishes the maximum value for Alpha when
    /// the algorithm is automatically adjusting its parameters.
    /// 
    /// If you enable adaptive parameters:
    /// - The algorithm will increase Alpha when reflections are successful
    /// - This setting prevents Alpha from becoming too large
    /// - It ensures the algorithm doesn't become too aggressive in its exploration
    /// 
    /// The default value of 2.0 means:
    /// - Even if reflections are consistently successful, Alpha won't go above 2.0
    /// - This prevents the algorithm from taking excessively large steps
    /// 
    /// You would typically only change this if you're using adaptive parameters and:
    /// - You want to allow for even more aggressive reflections (higher value)
    /// - You want to limit how far the algorithm can reflect (lower value)
    /// 
    /// This setting has no effect if UseAdaptiveParameters is false.
    /// </para>
    /// </remarks>
    public double MaxAlpha { get; set; } = 2.0;

    /// <summary>
    /// Gets or sets the minimum allowed value for the contraction coefficient when using adaptive parameters.
    /// </summary>
    /// <value>The minimum contraction coefficient, defaulting to 0.1.</value>
    /// <remarks>
    /// <para>
    /// When using adaptive parameters (UseAdaptiveParameters = true), this value sets a lower bound on how small
    /// the contraction coefficient (beta) can become. This prevents the contraction operations from becoming too
    /// aggressive, which could cause the simplex to collapse prematurely. The appropriate minimum value depends on
    /// the characteristics of the objective function and the desired exploration behavior.
    /// </para>
    /// <para><b>For Beginners:</b> This setting establishes the minimum value for Beta when
    /// the algorithm is automatically adjusting its parameters.
    /// 
    /// If you enable adaptive parameters:
    /// - The algorithm will adjust Beta based on how well contractions work
    /// - This setting prevents Beta from becoming too small
    /// - It ensures the algorithm doesn't contract too aggressively
    /// 
    /// The default value of 0.1 means:
    /// - Beta won't go below 0.1, meaning contracted points will always be at least 10%
    ///   of the way from the centroid to the worst point
    /// - This prevents the simplex from collapsing too quickly
    /// 
    /// You would typically only change this if you're using adaptive parameters and:
    /// - You want to allow for even more aggressive contractions (lower value)
    /// - You want to ensure more conservative minimum contractions (higher value)
    /// 
    /// This setting has no effect if UseAdaptiveParameters is false.
    /// </para>
    /// </remarks>
    public double MinBeta { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the maximum allowed value for the contraction coefficient when using adaptive parameters.
    /// </summary>
    /// <value>The maximum contraction coefficient, defaulting to 1.0.</value>
    /// <remarks>
    /// <para>
    /// When using adaptive parameters (UseAdaptiveParameters = true), this value sets an upper bound on how large
    /// the contraction coefficient (beta) can become. This prevents the contraction operations from becoming too
    /// conservative, which could slow down convergence. A value of 1.0 means that in the most conservative case,
    /// the contracted point would be at the centroid itself. Values greater than 1.0 are generally not used for
    /// contraction as they would move beyond the centroid.
    /// </para>
    /// <para><b>For Beginners:</b> This setting establishes the maximum value for Beta when
    /// the algorithm is automatically adjusting its parameters.
    /// 
    /// If you enable adaptive parameters:
    /// - The algorithm will increase Beta when contractions need to be more conservative
    /// - This setting prevents Beta from becoming too large
    /// 
    /// The default value of 1.0 means:
    /// - Beta won't go above 1.0, meaning contracted points won't go beyond the centroid
    /// - This represents the most conservative possible contraction
    /// 
    /// You would typically not set this above 1.0, as that would mean moving beyond the centroid,
    /// which defeats the purpose of contraction. You might set it lower if:
    /// - You want to ensure contractions always make meaningful progress toward the centroid
    /// - You don't want the algorithm to be too conservative even when adapting
    /// 
    /// This setting has no effect if UseAdaptiveParameters is false.
    /// </para>
    /// </remarks>
    public double MaxBeta { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the minimum allowed value for the expansion coefficient when using adaptive parameters.
    /// </summary>
    /// <value>The minimum expansion coefficient, defaulting to 1.0.</value>
    /// <remarks>
    /// <para>
    /// When using adaptive parameters (UseAdaptiveParameters = true), this value sets a lower bound on how small
    /// the expansion coefficient (gamma) can become. A value of 1.0 means that in the most conservative case,
    /// expansion would be equivalent to reflection (no additional movement). Values less than 1.0 are generally
    /// not used for expansion as they would be more conservative than the reflection.
    /// </para>
    /// <para><b>For Beginners:</b> This setting establishes the minimum value for Gamma when
    /// the algorithm is automatically adjusting its parameters.
    /// 
    /// If you enable adaptive parameters:
    /// - The algorithm will adjust Gamma based on how successful expansions are
    /// - This setting prevents Gamma from becoming too small
    /// 
    /// The default value of 1.0 means:
    /// - Gamma won't go below 1.0, which would make it equivalent to just reflection
    /// - This ensures expansion always tries to go at least as far as reflection
    /// 
    /// You would typically not set this below 1.0, as that would make expansion more conservative
    /// than reflection, which defeats its purpose. You might set it higher if:
    /// - You want to ensure expansions always go significantly beyond reflection
    /// - You want more aggressive minimum exploration in promising directions
    /// 
    /// This setting has no effect if UseAdaptiveParameters is false.
    /// </para>
    /// </remarks>
    public double MinGamma { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the maximum allowed value for the expansion coefficient when using adaptive parameters.
    /// </summary>
    /// <value>The maximum expansion coefficient, defaulting to 3.0.</value>
    /// <remarks>
    /// <para>
    /// When using adaptive parameters (UseAdaptiveParameters = true), this value sets an upper bound on how large
    /// the expansion coefficient (gamma) can become. This prevents the expansion operations from becoming too
    /// aggressive, which could cause the algorithm to overlook detailed structure in the objective function by
    /// taking excessively large steps. The appropriate maximum value depends on the characteristics of the
    /// objective function and the desired balance between exploration and exploitation.
    /// </para>
    /// <para><b>For Beginners:</b> This setting establishes the maximum value for Gamma when
    /// the algorithm is automatically adjusting its parameters.
    /// 
    /// If you enable adaptive parameters:
    /// - The algorithm will increase Gamma when expansions are successful
    /// - This setting prevents Gamma from becoming too large
    /// - It ensures the algorithm doesn't take excessively large steps
    /// 
    /// The default value of 3.0 means:
    /// - Gamma won't go above 3.0, meaning the algorithm won't try going more than 3 times
    ///   the reflection distance
    /// - This prevents the algorithm from making wild leaps that might miss important details
    /// 
    /// You might want to change this if you're using adaptive parameters and:
    /// - You want to allow for even more aggressive expansions (higher value)
    /// - You want to limit how far the algorithm can expand (lower value)
    /// 
    /// This setting has no effect if UseAdaptiveParameters is false.
    /// </para>
    /// </remarks>
    public double MaxGamma { get; set; } = 3.0;

    /// <summary>
    /// Gets or sets the minimum allowed value for the shrink coefficient when using adaptive parameters.
    /// </summary>
    /// <value>The minimum shrink coefficient, defaulting to 0.1.</value>
    /// <remarks>
    /// <para>
    /// When using adaptive parameters (UseAdaptiveParameters = true), this value sets a lower bound on how small
    /// the shrink coefficient (delta) can become. This prevents the shrinking operations from becoming too
    /// aggressive, which could cause the simplex to collapse too quickly and potentially miss the optimal solution.
    /// The appropriate minimum value depends on the characteristics of the objective function and the desired
    /// exploration behavior.
    /// </para>
    /// <para><b>For Beginners:</b> This setting establishes the minimum value for Delta when
    /// the algorithm is automatically adjusting its parameters.
    /// 
    /// If you enable adaptive parameters:
    /// - The algorithm will adjust Delta based on how effective shrinking is
    /// - This setting prevents Delta from becoming too small
    /// - It ensures the algorithm doesn't shrink too aggressively
    /// 
    /// The default value of 0.1 means:
    /// - Delta won't go below 0.1, meaning points will never move more than 90% of the way
    ///   toward the best point during shrinking
    /// - This prevents the simplex from collapsing too rapidly
    /// 
    /// You would typically only change this if you're using adaptive parameters and:
    /// - You want to allow for even more aggressive shrinking (lower value)
    /// - You want to ensure more conservative minimum shrinking (higher value)
    /// 
    /// This setting has no effect if UseAdaptiveParameters is false.
    /// </para>
    /// </remarks>
    public double MinDelta { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the maximum allowed value for the shrink coefficient when using adaptive parameters.
    /// </summary>
    /// <value>The maximum shrink coefficient, defaulting to 1.0.</value>
    /// <remarks>
    /// <para>
    /// When using adaptive parameters (UseAdaptiveParameters = true), this value sets an upper bound on how large
    /// the shrink coefficient (delta) can become. This prevents the shrinking operations from becoming too
    /// conservative, which could slow down convergence. A value of 1.0 would mean no shrinking at all (points
    /// maintain their original distances), so practical values should be less than 1.0.
    /// </para>
    /// <para><b>For Beginners:</b> This setting establishes the maximum value for Delta when
    /// the algorithm is automatically adjusting its parameters.
    /// 
    /// If you enable adaptive parameters:
    /// - The algorithm will increase Delta when less aggressive shrinking is needed
    /// - This setting prevents Delta from becoming too large
    /// 
    /// The default value of 1.0 means:
    /// - Delta won't go above 1.0, which would mean no shrinking at all
    /// - This represents the most conservative possible shrinking
    /// 
    /// You would typically not set this to 1.0 or above, as that would prevent effective shrinking.
    /// You might set it lower if:
    /// - You want to ensure the algorithm always makes meaningful progress during shrinking
    /// - You don't want the algorithm to be too conservative even when adapting
    /// 
    /// This setting has no effect if UseAdaptiveParameters is false.
    /// </para>
    /// </remarks>
    public double MaxDelta { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets whether to adaptively adjust the algorithm parameters based on their effectiveness.
    /// </summary>
    /// <value>Flag indicating whether to use adaptive parameters, defaulting to false.</value>
    /// <remarks>
    /// <para>
    /// When set to true, the algorithm will automatically adjust the reflection, expansion, contraction, and shrink
    /// coefficients based on how successful these operations are during optimization. Coefficients for operations
    /// that consistently yield improvements will be increased (within their specified bounds), while coefficients
    /// for operations that frequently fail will be decreased. This can improve performance on complex objective
    /// functions by tailoring the algorithm's behavior to the specific characteristics of the problem.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines whether the algorithm automatically
    /// adjusts its own parameters during optimization.
    /// 
    /// Think of it like learning from experience:
    /// - When enabled, the algorithm will increase parameters for operations that work well
    /// - And decrease parameters for operations that don't work well
    /// - This helps it adapt to the specific "terrain" it's exploring
    /// 
    /// The default value of false means:
    /// - The algorithm will use fixed values for Alpha, Beta, Gamma, and Delta throughout the optimization
    /// - This provides more predictable behavior but might be less efficient
    /// 
    /// You might want to set this to true if:
    /// - Your objective function has varied characteristics in different regions
    /// - You're not sure what parameter values would work best
    /// - You want the algorithm to self-tune based on its experience
    /// 
    /// You might want to keep it false if:
    /// - You've already tuned the parameters for your specific problem
    /// - You want consistent, predictable behavior
    /// - You're comparing different algorithm configurations systematically
    /// 
    /// When enabled, the other "Min" and "Max" parameters determine the bounds for adaptation,
    /// and AdaptationRate controls how quickly parameters change.
    /// </para>
    /// </remarks>
    public bool UseAdaptiveParameters { get; set; } = false;

    /// <summary>
    /// Gets or sets the rate at which parameters are adjusted when using adaptive parameters.
    /// </summary>
    /// <value>The adaptation rate, defaulting to 0.1.</value>
    /// <remarks>
    /// <para>
    /// When using adaptive parameters (UseAdaptiveParameters = true), this value controls how quickly the
    /// algorithm adjusts the reflection, expansion, contraction, and shrink coefficients based on their success
    /// or failure. Higher values cause more rapid adaptation, which can help the algorithm quickly find effective
    /// parameters but may also lead to more volatile behavior. Lower values provide more stable, gradual adaptation
    /// that is less likely to overreact to temporary patterns in the optimization landscape.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how quickly the algorithm adjusts its
    /// parameters when adaptive mode is enabled.
    /// 
    /// Think of it like a "learning rate" for the algorithm itself:
    /// - A higher value means it quickly changes its behavior based on recent experiences
    /// - A lower value means it changes more gradually, considering longer history
    /// 
    /// The default value of 0.1 means:
    /// - Parameters will change by roughly 10% of the distance toward their new target values
    ///   after each relevant operation
    /// - This provides moderate adaptation that doesn't overreact to individual successes or failures
    /// 
    /// You might want to increase this value if:
    /// - You want the algorithm to adapt more quickly to different regions of the function
    /// - You're starting with parameters that might be far from optimal
    /// 
    /// You might want to decrease this value if:
    /// - The algorithm seems to be changing its behavior too erratically
    /// - You want more stable, gradual adaptation
    /// - Your function has noisy evaluations that might mislead quick adaptation
    /// 
    /// This setting has no effect if UseAdaptiveParameters is false.
    /// </para>
    /// </remarks>
    public double AdaptationRate { get; set; } = 0.1;
}
