namespace AiDotNet.LinearAlgebra;

/// <summary>
/// Represents a node in a conditional inference tree, which is a type of decision tree
/// that uses statistical tests to make decisions at each node.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// A conditional inference tree is a statistical approach to decision tree learning that
/// uses significance tests to select variables at each split.
/// </para>
/// <para>
/// <b>For Beginners:</b> Think of this as a special type of decision tree that makes decisions
/// based on statistical evidence rather than just information gain. The p-value stored in each
/// node represents how confident we are that the split at this node is meaningful and not just
/// due to random chance. Lower p-values (closer to zero) indicate stronger evidence for the split.
/// </para>
/// </remarks>
public class ConditionalInferenceTreeNode<T> : DecisionTreeNode<T>
{
    /// <summary>
    /// Gets or sets the p-value associated with the statistical test at this node.
    /// </summary>
    /// <remarks>
    /// <para>
    /// In statistical hypothesis testing, the p-value represents the probability of observing
    /// results at least as extreme as the current results, assuming the null hypothesis is true.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> The p-value is a number between 0 and 1 that helps determine if
    /// the split at this node is statistically significant. A smaller p-value (typically below 0.05)
    /// suggests that the split is meaningful and not just due to random chance in the data.
    /// </para>
    /// </remarks>
    public T PValue { get; set; }

    /// <summary>
    /// Initializes a new instance of the <see cref="ConditionalInferenceTreeNode{T}"/> class
    /// with a default p-value of zero.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This creates a new node for the decision tree with an initial
    /// p-value of zero, which would indicate a highly significant split. As the tree is built,
    /// this value may be updated based on statistical tests.
    /// </remarks>
    public ConditionalInferenceTreeNode()
    {
        PValue = MathHelper.GetNumericOperations<T>().Zero;
    }
}
