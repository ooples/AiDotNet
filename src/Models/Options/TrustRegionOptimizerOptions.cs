namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Trust Region optimization algorithms, which are robust methods for
/// solving nonlinear optimization problems.
/// </summary>
/// <remarks>
/// <para>
/// Trust Region methods are iterative optimization techniques designed to find local minima or maxima of objective 
/// functions, particularly in nonlinear problems. Unlike line search methods, which determine the step size along 
/// a predefined direction, trust region methods concurrently optimize both the direction and the magnitude of the 
/// step within a specified neighborhood (the "trust region") around the current iterate. The central idea is to 
/// construct a simplified model—often a quadratic approximation—that represents the objective function near the 
/// current point. This model serves as a surrogate, guiding the search for the optimum within a bounded region. 
/// The size of the trust region is dynamically adjusted based on how well the model predicts the actual behavior 
/// of the objective function. This class inherits from GradientBasedOptimizerOptions and adds parameters specific 
/// to Trust Region optimization.
/// </para>
/// <para><b>For Beginners:</b> Trust Region methods are like exploring with a map that's only accurate near your current location.
/// 
/// When solving optimization problems:
/// - We often use approximations of the objective function to determine the next step
/// - But these approximations are only reliable within a certain distance
/// 
/// Trust Region methods solve this by:
/// - Creating a simplified model of the function around the current point
/// - Defining a "trust region" where this model is considered reliable
/// - Finding the best step within this region
/// - Adjusting the region size based on how well the model predicts actual function behavior
/// 
/// This approach offers several benefits:
/// - More robust than many other methods, especially for difficult problems
/// - Handles non-convex functions well
/// - Can make progress even when the Hessian is not positive definite
/// - Naturally limits step sizes to prevent erratic behavior
/// 
/// This class lets you configure how the Trust Region algorithm behaves.
/// </para>
/// </remarks>
public class TrustRegionOptimizerOptions<T, TInput, TOutput> : GradientBasedOptimizerOptions<T, TInput, TOutput>
{
    /// <summary>
    /// Gets or sets the initial radius of the trust region.
    /// </summary>
    /// <value>A positive double value, defaulting to 1.0.</value>
    /// <remarks>
    /// <para>
    /// This property specifies the initial size of the trust region, which defines the maximum distance the algorithm 
    /// can move from the current point in a single iteration. The trust region is a neighborhood around the current 
    /// point within which the quadratic model is considered to be a good approximation of the objective function. 
    /// A larger initial radius allows for more aggressive initial steps, potentially accelerating convergence if the 
    /// quadratic model is accurate over a larger region. A smaller initial radius is more conservative, ensuring that 
    /// the algorithm makes smaller, more reliable steps initially. The default value of 1.0 provides a moderate 
    /// initial radius suitable for many applications. The optimal value depends on the scale and characteristics of 
    /// the objective function.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines how far the algorithm can move in the first iteration.
    /// 
    /// The initial trust region radius:
    /// - Sets the starting size of the area where the simplified model is trusted
    /// - Limits how far the algorithm can move in the first step
    /// - Affects the initial balance between exploration and caution
    /// 
    /// The default value of 1.0 means:
    /// - The algorithm starts with a moderate-sized trust region
    /// - This provides a balanced approach for many problems
    /// 
    /// Think of it like this:
    /// - Larger values (e.g., 5.0): More aggressive initial steps, faster progress if the model is good
    /// - Smaller values (e.g., 0.1): More cautious initial steps, more reliable but potentially slower
    /// 
    /// When to adjust this value:
    /// - Increase it when you're confident the function is well-behaved and want faster initial progress
    /// - Decrease it when dealing with highly nonlinear or poorly scaled functions
    /// - Scale it according to the typical magnitude of variables in your problem
    /// 
    /// For example, if your variables typically have values around 100, you might
    /// increase this to 10.0 to allow appropriately scaled steps.
    /// </para>
    /// </remarks>
    public double InitialTrustRegionRadius { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the minimum allowed radius of the trust region.
    /// </summary>
    /// <value>A positive double value, defaulting to 1e-6.</value>
    /// <remarks>
    /// <para>
    /// This property specifies the lower bound for the trust region radius. If the trust region radius becomes 
    /// smaller than this value, it might indicate that the algorithm has converged to a solution or is unable to 
    /// make further progress. A very small minimum radius allows the algorithm to take very small steps when 
    /// necessary, potentially achieving higher precision at the cost of more iterations. The default value of 1e-6 
    /// provides a small minimum radius suitable for many applications, allowing for fine-grained convergence while 
    /// preventing numerical issues associated with extremely small steps. The optimal value depends on the desired 
    /// precision and the numerical characteristics of the problem.
    /// </para>
    /// <para><b>For Beginners:</b> This setting sets a lower limit on how small the trust region can become.
    /// 
    /// The minimum trust region radius:
    /// - Prevents the trust region from becoming too small
    /// - Acts as a stopping criterion when the algorithm can't make further progress
    /// - Affects the precision of the final solution
    /// 
    /// The default value of 1e-6 (0.000001) means:
    /// - The trust region won't shrink below this very small value
    /// - This allows for high precision while avoiding numerical issues
    /// 
    /// Think of it like this:
    /// - Smaller values (e.g., 1e-8): Allow for higher precision but may require more iterations
    /// - Larger values (e.g., 1e-4): May terminate earlier with slightly less precision
    /// 
    /// When to adjust this value:
    /// - Decrease it when you need extremely high precision in the solution
    /// - Increase it when you're willing to accept slightly less precision for faster termination
    /// - Consider the scale of your problem and numerical precision of your system
    /// 
    /// For example, in a scientific computing application requiring high precision,
    /// you might decrease this to 1e-8 or 1e-10.
    /// </para>
    /// </remarks>
    public double MinTrustRegionRadius { get; set; } = 1e-6;

    /// <summary>
    /// Gets or sets the maximum allowed radius of the trust region.
    /// </summary>
    /// <value>A positive double value, defaulting to 10.0.</value>
    /// <remarks>
    /// <para>
    /// This property specifies the upper bound for the trust region radius. It prevents the trust region from 
    /// becoming too large, which could lead to steps that are too aggressive and potentially destabilize the 
    /// optimization process. A larger maximum radius allows for more aggressive steps when the quadratic model 
    /// is accurate over a larger region, potentially accelerating convergence. A smaller maximum radius is more 
    /// conservative, ensuring that the algorithm never takes excessively large steps. The default value of 10.0 
    /// provides a moderate upper bound suitable for many applications. The optimal value depends on the scale and 
    /// characteristics of the objective function and the desired balance between aggressive exploration and 
    /// stability.
    /// </para>
    /// <para><b>For Beginners:</b> This setting sets an upper limit on how large the trust region can become.
    /// 
    /// The maximum trust region radius:
    /// - Prevents the trust region from becoming too large
    /// - Limits how aggressive the algorithm can be, even when successful
    /// - Helps maintain stability in the optimization process
    /// 
    /// The default value of 10.0 means:
    /// - The trust region won't expand beyond this size
    /// - This prevents excessively large steps that might overshoot good solutions
    /// 
    /// Think of it like this:
    /// - Larger values (e.g., 50.0): Allow more aggressive exploration when the model is good
    /// - Smaller values (e.g., 5.0): More conservative, never taking very large steps
    /// 
    /// When to adjust this value:
    /// - Increase it for well-behaved functions where large steps might be beneficial
    /// - Decrease it for highly nonlinear or poorly scaled functions
    /// - Scale it according to the typical magnitude of variables in your problem
    /// 
    /// For example, if your variables typically have values around 1000, you might
    /// increase this to 100.0 to allow appropriately scaled steps.
    /// </para>
    /// </remarks>
    public double MaxTrustRegionRadius { get; set; } = 10.0;

    /// <summary>
    /// Gets or sets the threshold for accepting a step.
    /// </summary>
    /// <value>A double value between 0 and 1, defaulting to 0.1.</value>
    /// <remarks>
    /// <para>
    /// This property specifies the minimum ratio of actual improvement to predicted improvement required for a step 
    /// to be accepted. The ratio is calculated as (actual reduction in objective function) / (predicted reduction 
    /// by quadratic model). If this ratio is greater than the acceptance threshold, the step is accepted and the 
    /// algorithm moves to the new point. A higher threshold is more strict, requiring better agreement between the 
    /// model and the actual function, while a lower threshold is more lenient. The default value of 0.1 means that 
    /// a step is accepted if the actual improvement is at least 10% of what was predicted by the model. This 
    /// provides a moderate criterion suitable for many applications, allowing progress even when the model is not 
    /// perfectly accurate. The optimal value depends on the accuracy of the quadratic model and the desired balance 
    /// between progress and reliability.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines how good a step must be to be accepted.
    /// 
    /// The acceptance threshold:
    /// - Determines when a proposed step is good enough to accept
    /// - Compares the actual improvement to what the model predicted
    /// - Affects how strict the algorithm is about model accuracy
    /// 
    /// The default value of 0.1 means:
    /// - A step is accepted if the actual improvement is at least 10% of what was predicted
    /// - This allows progress even when the model isn't perfectly accurate
    /// 
    /// Think of it like this:
    /// - Higher values (e.g., 0.3): More strict, requires better agreement between model and reality
    /// - Lower values (e.g., 0.01): More lenient, accepts steps with minimal improvement
    /// 
    /// When to adjust this value:
    /// - Increase it when you want to ensure more reliable progress
    /// - Decrease it when you want to accept more steps, even if they're not as good as predicted
    /// - Lower values can help escape flat regions but may accept poor steps
    /// 
    /// For example, in a noisy optimization problem where function evaluations have some randomness,
    /// you might decrease this to 0.05 to be more tolerant of discrepancies.
    /// </para>
    /// </remarks>
    public double AcceptanceThreshold { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the threshold for considering a step very successful.
    /// </summary>
    /// <value>A double value between 0 and 1, defaulting to 0.75.</value>
    /// <remarks>
    /// <para>
    /// This property specifies the ratio of actual improvement to predicted improvement above which a step is 
    /// considered very successful. If this ratio exceeds the threshold, the trust region radius is typically 
    /// increased for the next iteration, allowing for more aggressive steps. A higher threshold is more strict, 
    /// requiring better agreement between the model and the actual function before expanding the trust region, 
    /// while a lower threshold is more lenient. The default value of 0.75 means that a step is considered very 
    /// successful if the actual improvement is at least 75% of what was predicted by the model. This provides a 
    /// relatively strict criterion suitable for many applications, ensuring that the trust region is expanded only 
    /// when the model is quite accurate. The optimal value depends on the accuracy of the quadratic model and the 
    /// desired balance between aggressive exploration and stability.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines when a step is considered excellent, leading to a larger trust region.
    /// 
    /// The very successful threshold:
    /// - Determines when a step is considered excellent
    /// - When exceeded, the trust region is usually expanded
    /// - Affects how quickly the algorithm becomes more aggressive
    /// 
    /// The default value of 0.75 means:
    /// - A step is "very successful" if the actual improvement is at least 75% of what was predicted
    /// - This indicates the model is quite accurate and we can trust it over a larger region
    /// 
    /// Think of it like this:
    /// - Higher values (e.g., 0.9): More strict, requires excellent agreement before expanding
    /// - Lower values (e.g., 0.6): More lenient, expands the trust region more readily
    /// 
    /// When to adjust this value:
    /// - Increase it when you want to be more conservative about expanding the trust region
    /// - Decrease it when you want to be more aggressive in exploring larger regions
    /// - Higher values lead to more stable but potentially slower convergence
    /// 
    /// For example, in a well-behaved optimization problem where you want faster convergence,
    /// you might decrease this to 0.6 to expand the trust region more aggressively.
    /// </para>
    /// </remarks>
    public double VerySuccessfulThreshold { get; set; } = 0.75;

    /// <summary>
    /// Gets or sets the threshold for considering a step unsuccessful.
    /// </summary>
    /// <value>A double value between 0 and 1, defaulting to 0.25.</value>
    /// <remarks>
    /// <para>
    /// This property specifies the ratio of actual improvement to predicted improvement below which a step is 
    /// considered unsuccessful. If this ratio falls below the threshold, the trust region radius is typically 
    /// decreased for the next iteration, leading to more conservative steps. A higher threshold is more strict, 
    /// more readily declaring steps as unsuccessful and shrinking the trust region, while a lower threshold is 
    /// more lenient. The default value of 0.25 means that a step is considered unsuccessful if the actual 
    /// improvement is less than 25% of what was predicted by the model. This provides a moderate criterion 
    /// suitable for many applications, ensuring that the trust region is shrunk when the model is not very 
    /// accurate. The optimal value depends on the accuracy of the quadratic model and the desired balance between 
    /// progress and reliability.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines when a step is considered poor, leading to a smaller trust region.
    /// 
    /// The unsuccessful threshold:
    /// - Determines when a step is considered poor
    /// - When not met, the trust region is usually contracted
    /// - Affects how quickly the algorithm becomes more conservative
    /// 
    /// The default value of 0.25 means:
    /// - A step is "unsuccessful" if the actual improvement is less than 25% of what was predicted
    /// - This indicates the model isn't very accurate and we should be more cautious
    /// 
    /// Think of it like this:
    /// - Higher values (e.g., 0.4): More strict, contracts the trust region more readily
    /// - Lower values (e.g., 0.1): More lenient, keeps the trust region larger even with mediocre steps
    /// 
    /// When to adjust this value:
    /// - Increase it when you want to be more conservative when steps aren't as good as predicted
    /// - Decrease it when you want to maintain larger steps even when the model isn't very accurate
    /// - Higher values lead to more cautious but potentially slower convergence
    /// 
    /// For example, in a highly nonlinear optimization problem where the model might be inaccurate,
    /// you might increase this to 0.4 to be more cautious when the model predictions are off.
    /// </para>
    /// </remarks>
    public double UnsuccessfulThreshold { get; set; } = 0.25;

    /// <summary>
    /// Gets or sets the factor by which to expand the trust region radius after a very successful step.
    /// </summary>
    /// <value>A double value greater than 1, defaulting to 2.0.</value>
    /// <remarks>
    /// <para>
    /// This property specifies the factor by which the trust region radius is increased after a very successful step 
    /// (one where the ratio of actual improvement to predicted improvement exceeds the VerySuccessfulThreshold). A 
    /// larger expansion factor leads to more aggressive growth of the trust region, potentially accelerating convergence 
    /// but with a higher risk of taking steps that are too large. A smaller expansion factor leads to more conservative 
    /// growth, with more gradual increases in step size. The default value of 2.0 means that the trust region radius 
    /// is doubled after a very successful step, providing a moderate rate of expansion suitable for many applications. 
    /// The optimal value depends on the characteristics of the objective function and the desired balance between 
    /// aggressive exploration and stability.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how much the trust region grows after a very successful step.
    /// 
    /// The expansion factor:
    /// - Determines how much the trust region expands after an excellent step
    /// - Affects how quickly the algorithm becomes more aggressive
    /// - Higher values lead to faster expansion but potentially less stability
    /// 
    /// The default value of 2.0 means:
    /// - After a very successful step, the trust region radius is doubled
    /// - This allows for reasonably quick expansion when the model is accurate
    /// 
    /// Think of it like this:
    /// - Higher values (e.g., 3.0): More aggressive expansion, faster potential convergence
    /// - Lower values (e.g., 1.5): More conservative expansion, more stable behavior
    /// 
    /// When to adjust this value:
    /// - Increase it when you want faster expansion of the trust region after successful steps
    /// - Decrease it when you want more gradual, conservative growth
    /// - Higher values can speed convergence but may lead to overshooting
    /// 
    /// For example, in a well-behaved optimization problem where you want faster convergence,
    /// you might increase this to 2.5 or 3.0 to expand the trust region more aggressively.
    /// </para>
    /// </remarks>
    public double ExpansionFactor { get; set; } = 2.0;

    /// <summary>
    /// Gets or sets the factor by which to contract the trust region radius after an unsuccessful step.
    /// </summary>
    /// <value>A double value between 0 and 1, defaulting to 0.5.</value>
    /// <remarks>
    /// <para>
    /// This property specifies the factor by which the trust region radius is decreased after an unsuccessful step 
    /// (one where the ratio of actual improvement to predicted improvement falls below the UnsuccessfulThreshold). 
    /// A smaller contraction factor leads to more aggressive shrinking of the trust region, quickly reducing step 
    /// sizes when the model is inaccurate. A larger contraction factor leads to more conservative shrinking, with 
    /// more gradual decreases in step size. The default value of 0.5 means that the trust region radius is halved 
    /// after an unsuccessful step, providing a moderate rate of contraction suitable for many applications. The 
    /// optimal value depends on the characteristics of the objective function and the desired balance between quick 
    /// adaptation to model inaccuracies and avoiding excessive shrinking of the trust region.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how much the trust region shrinks after an unsuccessful step.
    /// 
    /// The contraction factor:
    /// - Determines how much the trust region shrinks after a poor step
    /// - Affects how quickly the algorithm becomes more conservative
    /// - Lower values lead to faster contraction and more caution
    /// 
    /// The default value of 0.5 means:
    /// - After an unsuccessful step, the trust region radius is halved
    /// - This allows for reasonably quick adjustment when the model is inaccurate
    /// 
    /// Think of it like this:
    /// - Lower values (e.g., 0.25): More aggressive contraction, quickly becoming more cautious
    /// - Higher values (e.g., 0.75): More conservative contraction, maintaining larger steps longer
    /// 
    /// When to adjust this value:
    /// - Decrease it when you want faster contraction of the trust region after poor steps
    /// - Increase it when you want more gradual reduction in step size
    /// - Lower values can help quickly recover from bad steps but may slow convergence
    /// 
    /// For example, in a highly nonlinear optimization problem where large steps often fail,
    /// you might decrease this to 0.3 to more quickly reduce the trust region after poor steps.
    /// </para>
    /// </remarks>
    public double ContractionFactor { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets a value indicating whether to use adaptive trust region radius adjustment.
    /// </summary>
    /// <value>A boolean value, defaulting to true.</value>
    /// <remarks>
    /// <para>
    /// This property specifies whether the algorithm should adaptively adjust the trust region radius based on the 
    /// performance history, rather than using fixed expansion and contraction factors. Adaptive adjustment can lead 
    /// to more efficient optimization by learning appropriate step sizes from the optimization history. When enabled, 
    /// the algorithm might use techniques such as interpolation or extrapolation based on past performance to determine 
    /// the next trust region radius. The default value of true enables adaptive adjustment, which is suitable for many 
    /// applications, especially those with complex or poorly scaled objective functions. Disabling adaptive adjustment 
    /// might be preferred for simpler problems or when more predictable behavior is desired.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines whether the algorithm learns from past steps to adjust the trust region size.
    /// 
    /// Adaptive trust region radius:
    /// - When enabled, the algorithm learns from past performance to adjust the trust region
    /// - Can lead to more efficient optimization by adapting to the specific problem
    /// - Uses more sophisticated strategies than simple expansion/contraction
    /// 
    /// The default value of true means:
    /// - The algorithm will adaptively adjust the trust region based on optimization history
    /// - This is generally more efficient for complex problems
    /// 
    /// Think of it like this:
    /// - Enabled: More intelligent adjustment based on past performance, potentially faster convergence
    /// - Disabled: Simpler, more predictable behavior using fixed expansion/contraction factors
    /// 
    /// When to adjust this value:
    /// - Keep enabled (true) for most problems, especially complex ones
    /// - Disable (false) when you want more predictable, deterministic behavior
    /// - Disable if you're debugging and want simpler algorithm behavior
    /// 
    /// For example, for a complex optimization problem with varying characteristics in different regions,
    /// keeping this enabled helps the algorithm adapt to each region appropriately.
    /// </para>
    /// </remarks>
    public bool UseAdaptiveTrustRegionRadius { get; set; } = true;

    /// <summary>
    /// Gets or sets the rate at which the trust region radius adapts to new information.
    /// </summary>
    /// <value>A double value between 0 and 1, defaulting to 0.1.</value>
    /// <remarks>
    /// <para>
    /// This property specifies the rate at which the trust region radius adapts to new information when adaptive 
    /// adjustment is enabled. It controls the balance between using historical information and recent performance 
    /// to determine the trust region radius. A higher adaptation rate gives more weight to recent performance, 
    /// leading to faster adaptation but potentially more erratic behavior. A lower adaptation rate gives more weight 
    /// to historical information, leading to more stable but potentially slower adaptation. The default value of 0.1 
    /// provides a relatively low adaptation rate suitable for many applications, ensuring stable adaptation while 
    /// still responding to changes in the objective function landscape. The optimal value depends on the variability 
    /// of the objective function and the desired balance between responsiveness and stability.
    /// </para>
    /// <para><b>For Beginners:</b> This setting controls how quickly the algorithm adapts the trust region based on new information.
    /// 
    /// The adaptation rate:
    /// - Determines how much weight is given to recent performance versus history
    /// - Affects how quickly the trust region adapts to changing conditions
    /// - Only relevant when UseAdaptiveTrustRegionRadius is true
    /// 
    /// The default value of 0.1 means:
    /// - The algorithm gives 10% weight to new information and 90% to historical information
    /// - This provides stable adaptation that doesn't overreact to individual steps
    /// 
    /// Think of it like this:
    /// - Higher values (e.g., 0.3): Faster adaptation, more responsive to recent performance
    /// - Lower values (e.g., 0.05): Slower adaptation, more stable behavior
    /// 
    /// When to adjust this value:
    /// - Increase it when you want the algorithm to adapt more quickly to changing conditions
    /// - Decrease it when you want more stable, consistent behavior
    /// - Higher values can help in problems with distinct regions of different characteristics
    /// 
    /// For example, in an optimization problem where the function characteristics change significantly
    /// in different regions, you might increase this to 0.2 for faster adaptation.
    /// </para>
    /// </remarks>
    public double AdaptationRate { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the maximum number of iterations for the Conjugate Gradient method used to solve the trust region subproblem.
    /// </summary>
    /// <value>A positive integer, defaulting to 100.</value>
    /// <remarks>
    /// <para>
    /// This property specifies the maximum number of iterations for the Conjugate Gradient (CG) method, which is 
    /// commonly used to solve the trust region subproblem (finding the best step within the trust region). The CG 
    /// method is an iterative algorithm, and this parameter limits how many iterations it can perform before returning 
    /// the best solution found so far. A higher limit allows for more iterations and potentially more accurate solutions 
    /// to the subproblem, but increases the computational cost per trust region iteration. The default value of 100 
    /// provides a reasonable upper limit for many applications, allowing sufficient iterations for convergence while 
    /// preventing excessive computation. The optimal value depends on the size and complexity of the problem and the 
    /// desired trade-off between subproblem accuracy and computational efficiency.
    /// </para>
    /// <para><b>For Beginners:</b> This setting limits how many iterations the algorithm spends solving each trust region subproblem.
    /// 
    /// The maximum CG iterations:
    /// - Limits the computational effort spent finding the best step within each trust region
    /// - Affects the balance between accuracy of each step and overall speed
    /// - CG (Conjugate Gradient) is the method used to solve the trust region subproblem
    /// 
    /// The default value of 100 means:
    /// - The algorithm will perform at most 100 iterations to find the best step within each trust region
    /// - This is sufficient for many problems while preventing excessive computation
    /// 
    /// Think of it like this:
    /// - Higher values (e.g., 200): More accurate steps within each trust region, but more computation per iteration
    /// - Lower values (e.g., 50): Less computation per iteration, but potentially less optimal steps
    /// 
    /// When to adjust this value:
    /// - Increase it for complex problems where accurate subproblem solutions are important
    /// - Decrease it when computational efficiency is critical or for simpler problems
    /// - Scale it with the dimensionality of your problem
    /// 
    /// For example, for a high-dimensional optimization problem with hundreds of variables,
    /// you might increase this to 200-300 to ensure accurate steps within each trust region.
    /// </para>
    /// </remarks>
    public int MaxCGIterations { get; set; } = 100;

    /// <summary>
    /// Gets or sets the convergence tolerance for the Conjugate Gradient method used to solve the trust region subproblem.
    /// </summary>
    /// <value>A positive double value, defaulting to 1e-6.</value>
    /// <remarks>
    /// <para>
    /// This property specifies the convergence tolerance for the Conjugate Gradient (CG) method, which is commonly 
    /// used to solve the trust region subproblem (finding the best step within the trust region). The CG method is 
    /// considered to have converged when the residual norm is less than this tolerance. A smaller tolerance requires 
    /// more precise convergence, potentially leading to more accurate solutions to the subproblem but requiring more 
    /// iterations. The default value of 1e-6 (0.000001) provides a relatively strict convergence criterion suitable 
    /// for many applications, ensuring accurate solutions to the subproblem while allowing the algorithm to terminate 
    /// in a reasonable number of iterations. The optimal value depends on the desired precision of the subproblem 
    /// solutions and the computational resources available.
    /// </para>
    /// <para><b>For Beginners:</b> This setting determines how precisely the algorithm solves each trust region subproblem.
    /// 
    /// The CG tolerance:
    /// - Defines when the subproblem solver considers itself "done"
    /// - Smaller values require more precise solutions to each subproblem
    /// - Affects both the quality of each step and the computation required
    /// 
    /// The default value of 1e-6 (0.000001) means:
    /// - The subproblem solver stops when the error is very small
    /// - This provides good precision for most applications
    /// 
    /// Think of it like this:
    /// - Smaller values (e.g., 1e-8): More precise subproblem solutions, but may take more iterations
    /// - Larger values (e.g., 1e-4): Faster subproblem solutions, but potentially less optimal steps
    /// 
    /// When to adjust this value:
    /// - Decrease it when you need very precise steps within each trust region
    /// - Increase it when computational efficiency is more important than precision
    /// - For most applications, the default value works well
    /// 
    /// For example, in a scientific computing application requiring high precision,
    /// you might decrease this to 1e-8 for more precise subproblem solutions.
    /// </para>
    /// </remarks>
    public double CGTolerance { get; set; } = 1e-6;
}
