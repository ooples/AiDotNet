namespace AiDotNet.Enums;

/// <summary>
/// Represents different types of acquisition functions used in Bayesian optimization.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Acquisition functions help an AI system decide where to look next when searching 
/// for the best solution to a problem.
/// 
/// Imagine you're trying to find the highest point in a mountain range that's covered in fog:
/// 
/// - You've already explored a few spots and know their heights
/// - Based on these measurements, you can make educated guesses about unexplored areas
/// - But you need to decide: should you explore areas that look promising based on what you know so far,
///   or should you check completely unexplored areas that might contain surprises?
/// 
/// This is the "exploration vs. exploitation" trade-off, and acquisition functions help balance it.
/// 
/// Acquisition functions are particularly useful when:
/// - Testing each possible solution is expensive or time-consuming
/// - You want to find the best solution with as few attempts as possible
/// - The relationship between inputs and outputs is complex
/// 
/// Common applications include hyperparameter tuning in machine learning, experimental design,
/// and optimizing complex systems where each test is costly.
/// </para>
/// </remarks>
public enum AcquisitionFunctionType
{
    /// <summary>
    /// Upper Confidence Bound acquisition function that balances exploration and exploitation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The Upper Confidence Bound (UCB) approach is like an optimistic explorer who 
    /// follows this rule: "I'll check places that either look promising or that I'm very uncertain about."
    /// 
    /// UCB works by:
    /// 
    /// 1. Calculating the predicted value at each possible point (exploitation)
    /// 2. Adding an "uncertainty bonus" that's larger for less-explored areas (exploration)
    /// 3. Selecting the point with the highest combined score
    /// 
    /// The formula is essentially: UCB = predicted_value + exploration_weight × uncertainty
    /// 
    /// Key characteristics:
    /// - Has a tunable parameter that controls the exploration-exploitation balance
    /// - More exploration-focused than some other methods
    /// - Works well when you want to avoid bad outcomes
    /// - Simple to understand and implement
    /// - Provides theoretical guarantees about finding the optimal solution
    /// 
    /// UCB is particularly useful when you want to be cautious and thoroughly explore the space
    /// before committing to a solution.
    /// </para>
    /// </remarks>
    UpperConfidenceBound,

    /// <summary>
    /// Expected Improvement acquisition function that focuses on areas likely to improve upon the current best solution.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Expected Improvement (EI) is like a strategic explorer who asks: "Where am I most 
    /// likely to find something better than the best I've seen so far?"
    /// 
    /// EI works by:
    /// 
    /// 1. Keeping track of the best solution found so far
    /// 2. For each unexplored point, calculating how much better it might be than the current best
    /// 3. Weighting this potential improvement by the probability of achieving it
    /// 4. Selecting the point with the highest expected improvement
    /// 
    /// Key characteristics:
    /// - Naturally balances exploration and exploitation without extra parameters
    /// - Focuses more on exploitation as it finds good solutions
    /// - Tends to explore more efficiently than UCB in many cases
    /// - Works well when you want to find the very best solution
    /// - Popular in practical applications like hyperparameter tuning
    /// 
    /// EI is often the default choice for many Bayesian optimization problems because it efficiently 
    /// finds good solutions with relatively few evaluations.
    /// </para>
    /// </remarks>
    ExpectedImprovement,

    /// <summary>
    /// Probability of Improvement acquisition function that maximizes the probability of finding better solutions.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Probability of Improvement (PI) is like a cautious explorer who asks:
    /// "What's the chance that this location is better than the best I've found so far?"
    ///
    /// PI works by:
    ///
    /// 1. Keeping track of the best solution found so far
    /// 2. For each unexplored point, calculating the probability that it's better than the current best
    /// 3. Selecting the point with the highest probability of improvement
    ///
    /// Key characteristics:
    /// - Simpler than Expected Improvement (focuses on probability, not magnitude)
    /// - Very exploitation-focused once good solutions are found
    /// - Tends to be more conservative than EI or UCB
    /// - Good when you want high confidence of improvement
    /// - May explore less than other methods
    ///
    /// PI is useful when you want to be confident that each new evaluation will be better
    /// than what you've already found, even if the improvement is small.
    /// </para>
    /// </remarks>
    ProbabilityOfImprovement
}
