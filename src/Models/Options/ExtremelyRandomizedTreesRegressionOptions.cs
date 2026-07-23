namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for Extremely Randomized Trees regression, an ensemble learning method
/// that builds multiple decision trees with additional randomization for improved prediction accuracy.
/// </summary>
/// <remarks>
/// <para>
/// Extremely Randomized Trees (also known as Extra Trees) is an ensemble learning method that extends
/// Random Forests by introducing additional randomization in the way splits are computed. While Random Forests
/// search for the optimal split among a random subset of features, Extra Trees select random splits for each
/// feature and choose the best among those. This additional randomization helps reduce variance and often
/// improves generalization.
/// </para>
/// <para><b>For Beginners:</b> Extremely Randomized Trees is like having a large committee of decision-makers
/// (trees) who each look at your data in a slightly different, randomized way. Imagine asking 100 people to
/// help you decide whether to buy a house, but each person can only consider a random subset of factors
/// (like price, location, size) and must make quick decisions without overthinking. By averaging all their
/// opinions, you often get better advice than from just one person or from a committee that all thinks the
/// same way. This randomness helps the model avoid focusing too much on specific patterns that might just be
/// coincidences in your training data.</para>
/// </remarks>
public class ExtremelyRandomizedTreesRegressionOptions : DecisionTreeOptions
{
    /// <summary>
    /// Gets or sets the number of decision trees to build in the ensemble.
    /// </summary>
    /// <value>The number of trees, defaulting to 100.</value>
    /// <remarks>
    /// <para>
    /// This parameter controls the size of the ensemble. More trees generally lead to better performance
    /// but increase training time and memory usage. The performance improvement typically plateaus after
    /// a certain number of trees.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how many different decision trees the model will create
    /// and combine for making predictions. With the default value of 100, the model builds 100 different trees
    /// and averages their predictions. Think of it like getting opinions from 100 different advisors instead
    /// of just one. More trees usually give more stable and accurate predictions (like how asking more people
    /// gives you a more reliable consensus), but they also make the model slower to train and use more memory.
    /// For simple problems, you might reduce this to 50 or even 10 trees. For complex problems where accuracy
    /// is critical, you might increase it to 200 or more.</para>
    /// </remarks>
    public int NumberOfTrees { get; set; } = 100;

    /// <summary>
    /// Gets or sets the maximum number of trees that can be trained simultaneously.
    /// </summary>
    /// <value>The maximum degree of parallelism, defaulting to the number of processor cores.</value>
    /// <remarks>
    /// <para>
    /// This parameter controls how many trees can be built in parallel, which can significantly speed up
    /// training on multi-core systems. By default, it uses all available processor cores. Setting this to
    /// a lower value may be beneficial on systems where you want to reserve processing power for other tasks.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how many trees the model can build at the same time,
    /// taking advantage of multiple processor cores in your computer. By default, it uses all available cores
    /// (currently {Environment.ProcessorCount} on your system). This is like having multiple workers building
    /// different parts of a project simultaneously instead of one worker doing everything sequentially.
    /// You might lower this number if you want your computer to remain responsive for other tasks while
    /// training the model. For example, setting it to half your core count would leave processing power
    /// available for other applications.</para>
    /// </remarks>
    public int MaxDegreeOfParallelism { get; set; } = Environment.ProcessorCount;
}
