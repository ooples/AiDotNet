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
public class DecisionTreeOptions : ModelOptions
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

    /// <summary>
    /// Gets or sets whether to use soft (differentiable) tree mode for JIT compilation support.
    /// </summary>
    /// <value><c>true</c> to enable soft tree mode; <c>false</c> (default) for traditional hard decision trees.</value>
    /// <remarks>
    /// <para>
    /// When enabled, the decision tree uses sigmoid-based soft gating instead of hard if-then splits.
    /// This makes the tree differentiable and enables JIT compilation support.
    /// </para>
    /// <para>
    /// Formula at each split: output = σ((threshold - x[feature]) / temperature) * left + (1 - σ) * right
    /// where σ is the sigmoid function.
    /// </para>
    /// <para><b>For Beginners:</b> Soft tree mode allows the decision tree to be JIT compiled for faster inference.
    ///
    /// Traditional decision trees make hard yes/no decisions at each split:
    /// - "If feature &gt; 5, go LEFT, otherwise go RIGHT"
    /// - This creates sharp boundaries that can't be compiled into a computation graph
    ///
    /// Soft trees use smooth transitions instead:
    /// - Near the boundary, the output blends both left and right paths
    /// - This creates a smooth, differentiable function
    /// - The temperature parameter controls how sharp the transitions are
    ///
    /// Soft trees give similar results to hard trees but can be JIT compiled.
    /// Lower temperature = closer to hard tree behavior.
    /// </para>
    /// </remarks>
    public bool UseSoftTree { get; set; } = false;

    /// <summary>
    /// Gets or sets the temperature parameter for soft decision tree mode.
    /// </summary>
    /// <value>
    /// The temperature for sigmoid gating. Default is 1.0.
    /// Lower values produce sharper decisions (closer to hard tree behavior).
    /// </value>
    /// <remarks>
    /// <para>
    /// Only used when <see cref="UseSoftTree"/> is enabled. Controls the smoothness of
    /// the soft split operations:
    /// </para>
    /// <list type="bullet">
    /// <item><description>Lower temperature (e.g., 0.1) = sharper, more discrete decisions</description></item>
    /// <item><description>Higher temperature (e.g., 10.0) = softer, more blended decisions</description></item>
    /// </list>
    /// <para><b>For Beginners:</b> Temperature controls how "crisp" the decisions are.
    ///
    /// Imagine a dial that goes from "very crisp" to "very smooth":
    /// - Low temperature (0.1): Almost like a regular decision tree, sharp boundaries
    /// - High temperature (10.0): Very smooth transitions, more averaging between branches
    /// - Default (1.0): Balanced behavior
    ///
    /// Start with 1.0 and adjust if needed. Lower values give predictions closer to traditional
    /// decision trees but may have numerical stability issues if too low.
    /// </para>
    /// </remarks>
    public double SoftTreeTemperature { get; set; } = 1.0;
}
