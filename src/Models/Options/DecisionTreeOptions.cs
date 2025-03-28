namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for decision tree algorithms.
/// </summary>
/// <remarks>
/// <para>
/// Decision trees are machine learning models that make predictions by following a tree-like structure of decisions.
/// These options control how the decision tree is built and how it makes predictions.
/// </para>
/// <para><b>For Beginners:</b> A decision tree works like a flowchart that asks a series of questions about your data
/// to arrive at a prediction. Imagine playing a game of "20 Questions" where each question narrows down the possible answers.
/// These settings control how detailed the questions can get, how many questions to ask, and how to decide which
/// questions are most important.</para>
/// </remarks>
public class DecisionTreeOptions
{
    /// <summary>
    /// Gets or sets the maximum depth (number of levels) of the decision tree.
    /// </summary>
    /// <value>
    /// The maximum depth of the tree, defaulting to 10.
    /// </value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This controls how many questions the tree can ask in sequence before making a prediction.
    /// A higher value allows the tree to learn more complex patterns but might cause "overfitting" (memorizing the training data
    /// instead of learning general patterns). The default value of 10 is a good starting point for most problems.</para>
    /// </remarks>
    public int MaxDepth { get; set; } = 10;

    /// <summary>
    /// Gets or sets the minimum number of samples required to split an internal node.
    /// </summary>
    /// <value>
    /// The minimum number of samples needed to split a node, defaulting to 2.
    /// </value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This determines how many data points must be present before the tree can create a new branch.
    /// Higher values prevent the tree from creating branches based on very few examples, which helps avoid overfitting.
    /// The default value of 2 means at least two data points are needed to create a split.</para>
    /// </remarks>
    public int MinSamplesSplit { get; set; } = 2;

    /// <summary>
    /// Gets or sets the fraction of features to consider when looking for the best split.
    /// </summary>
    /// <value>
    /// The fraction of features to consider (between 0.0 and 1.0), defaulting to 1.0 (all features).
    /// </value>
    /// <remarks>
    /// <para><b>For Beginners:</b> If your data has many characteristics (features), this setting controls what percentage
    /// of those features the tree considers when deciding how to split the data. A value of 1.0 means use all features,
    /// while 0.5 would mean randomly use only half of them for each split. Using fewer features can help create more diverse
    /// trees, which is especially useful when combining multiple trees together.</para>
    /// </remarks>
    public double MaxFeatures { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the random seed for reproducibility.
    /// </summary>
    /// <value>
    /// The random seed value, or null if randomness should not be controlled.
    /// </value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is like setting the starting point for a random number generator.
    /// If you set a specific seed value, the "random" decisions the algorithm makes will be the same each time you run it.
    /// This is useful when you want consistent results or when debugging. If left as null (the default),
    /// the algorithm will make truly random decisions each time it runs.</para>
    /// </remarks>
    public int? Seed { get; set; }

    /// <summary>
    /// Gets or sets the criterion used to evaluate the quality of a split.
    /// </summary>
    /// <value>
    /// The split criterion to use, defaulting to VarianceReduction.
    /// </value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This determines how the tree decides which question (split) is best at each step.
    /// For regression problems (predicting numbers), "VarianceReduction" is typically used - it tries to group similar values together.
    /// Think of it as trying to create groups where the numbers within each group are as close as possible to each other.</para>
    /// <para>Different criteria can be better for different types of problems. VarianceReduction works well for most
    /// regression tasks, which is why it's the default.</para>
    /// </remarks>
    public SplitCriterion SplitCriterion { get; set; } = SplitCriterion.VarianceReduction;
}