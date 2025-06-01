namespace AiDotNet.Enums;

/// <summary>
/// Defines strategies for merging results from parallel branches in a pipeline.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> When a pipeline splits into multiple branches (like a river delta), 
/// this enum defines how to combine the results when those branches come back together.
/// </para>
/// </remarks>
public enum BranchMergeStrategy
{
    /// <summary>
    /// Concatenates results from all branches.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Simply puts all results together end-to-end - like combining multiple 
    /// lists into one long list.
    /// </remarks>
    Concatenate,

    /// <summary>
    /// Averages numerical results from all branches.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Calculates the average of all branch results - useful when branches 
    /// produce similar types of numerical predictions.
    /// </remarks>
    Average,

    /// <summary>
    /// Weighted average based on branch importance.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Like average, but gives more weight to more important branches - 
    /// like giving more consideration to expert opinions.
    /// </remarks>
    WeightedAverage,

    /// <summary>
    /// Takes the maximum value from all branches.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Selects the highest value from all branches - useful for finding 
    /// the best score or highest confidence.
    /// </remarks>
    Maximum,

    /// <summary>
    /// Takes the minimum value from all branches.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Selects the lowest value from all branches - useful for conservative 
    /// estimates or finding the worst case.
    /// </remarks>
    Minimum,

    /// <summary>
    /// Sums results from all branches.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Adds up all branch results - useful when branches contribute partial 
    /// results that should be totaled.
    /// </remarks>
    Sum,

    /// <summary>
    /// Multiplies results from all branches.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Multiplies all branch results together - useful for combining 
    /// probabilities or scaling factors.
    /// </remarks>
    Product,

    /// <summary>
    /// Uses voting to select the best result.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Each branch votes for a result, and the most popular choice wins - 
    /// like a democratic decision.
    /// </remarks>
    Voting,

    /// <summary>
    /// Selects result from the first completed branch.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Uses whichever branch finishes first - useful when speed is more 
    /// important than comparing all options.
    /// </remarks>
    FirstCompleted,

    /// <summary>
    /// Selects result from the best performing branch.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Chooses the result from the branch that performed best according 
    /// to some metric - like picking the winner of a competition.
    /// </remarks>
    BestPerforming,

    /// <summary>
    /// Applies logical AND operation to boolean results.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> All branches must return true for the final result to be true - 
    /// like requiring unanimous agreement.
    /// </remarks>
    LogicalAnd,

    /// <summary>
    /// Applies logical OR operation to boolean results.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> If any branch returns true, the final result is true - like needing 
    /// just one person to say yes.
    /// </remarks>
    LogicalOr,

    /// <summary>
    /// Uses a learned model to merge results.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Trains a separate model to learn the best way to combine branch 
    /// results - like hiring a specialist to make the final decision.
    /// </remarks>
    LearnedMerge,

    /// <summary>
    /// Hierarchical merging with multiple levels.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Merges branches in groups first, then merges those groups - like 
    /// organizing a tournament with preliminary rounds.
    /// </remarks>
    Hierarchical,

    /// <summary>
    /// Custom merge strategy defined by user.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Allows you to implement your own custom logic for combining branch results.
    /// </remarks>
    Custom
}