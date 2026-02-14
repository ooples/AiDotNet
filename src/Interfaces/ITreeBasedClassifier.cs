namespace AiDotNet.Interfaces;

/// <summary>
/// Interface for tree-based classification algorithms.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Tree-based classifiers make decisions by learning a series of hierarchical rules from data.
/// They are highly interpretable and can capture non-linear relationships between features.
/// </para>
/// <para>
/// <b>For Beginners:</b>
/// Decision trees work like a flowchart - they ask a series of yes/no questions about features
/// to reach a decision. For example, to classify if an animal is a cat:
/// "Has fur?" (yes) -> "Has whiskers?" (yes) -> "Meows?" (yes) -> "It's a cat!"
///
/// Key properties:
/// - MaxDepth: How deep the tree can go (more depth = more complex decisions)
/// - Feature importance: Which features were most useful for classification
/// </para>
/// </remarks>
[AiDotNet.Configuration.YamlConfigurable("TreeBasedClassifier")]
public interface ITreeBasedClassifier<T> : IProbabilisticClassifier<T>
{
    /// <summary>
    /// Gets the maximum depth of the tree.
    /// </summary>
    /// <value>
    /// The maximum depth reached during training, or the configured maximum depth.
    /// </value>
    int MaxDepth { get; }

    /// <summary>
    /// Gets the feature importance scores computed during training.
    /// </summary>
    /// <value>
    /// A vector of importance scores, one for each feature. Higher values indicate
    /// more important features. Returns null if the model has not been trained.
    /// </value>
    /// <remarks>
    /// <para>
    /// Feature importance is typically computed based on how much each feature
    /// contributes to reducing impurity (e.g., Gini impurity or entropy) in the tree.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b>
    /// This tells you which features the tree found most useful for making decisions.
    /// A high importance score means that feature appears often near the top of the tree
    /// and is crucial for classification.
    /// </para>
    /// </remarks>
    Vector<T>? FeatureImportances { get; }

    /// <summary>
    /// Gets the number of leaf nodes in the tree.
    /// </summary>
    /// <value>
    /// The count of terminal nodes (leaves) in the trained tree.
    /// Returns 0 if the model has not been trained.
    /// </value>
    int LeafCount { get; }

    /// <summary>
    /// Gets the number of internal (decision) nodes in the tree.
    /// </summary>
    /// <value>
    /// The count of non-terminal nodes that make decisions.
    /// Returns 0 if the model has not been trained.
    /// </value>
    int NodeCount { get; }
}
