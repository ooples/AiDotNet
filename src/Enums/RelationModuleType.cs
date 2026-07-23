namespace AiDotNet.Enums;

/// <summary>
/// Types of relation module architectures for Relation Networks.
/// </summary>
/// <remarks>
/// <para>
/// The relation module learns to compare feature embeddings and output a similarity score.
/// Different architectures provide different ways of computing this learned similarity.
/// </para>
/// <para><b>For Beginners:</b> This determines HOW the network compares two examples.
/// Instead of using a fixed formula (like Euclidean distance), Relation Networks learn
/// a neural network to measure "how related" two examples are. This enum controls the
/// architecture of that comparison network.
/// </para>
/// </remarks>
public enum RelationModuleType
{
    /// <summary>
    /// Concatenates features and passes through MLP.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The simplest approach: concatenate the two feature vectors and pass through
    /// a multi-layer perceptron (MLP) that outputs a relation score.
    /// </para>
    /// <para><b>For Beginners:</b> This is like putting two descriptions side by side
    /// and asking a neural network "how similar are these?" The network learns to
    /// look at both descriptions together and output a similarity score.
    /// </para>
    /// </remarks>
    Concatenate,

    /// <summary>
    /// Stacks features and applies 2D convolution.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Stacks the two feature maps spatially and applies 2D convolution layers
    /// to learn local patterns that indicate similarity.
    /// </para>
    /// <para><b>For Beginners:</b> This is useful for image data where spatial
    /// patterns matter. It's like overlaying two images and looking for patterns
    /// in how they match or differ.
    /// </para>
    /// </remarks>
    Convolution,

    /// <summary>
    /// Uses attention mechanism to relate features.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Uses attention to compute weighted relationships between different parts
    /// of the feature embeddings.
    /// </para>
    /// <para><b>For Beginners:</b> This allows the network to focus on the most
    /// important parts of each example when comparing them, rather than treating
    /// all features equally.
    /// </para>
    /// </remarks>
    Attention,

    /// <summary>
    /// Uses transformer-style self-attention.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Applies transformer-style multi-head self-attention for computing relations,
    /// allowing complex feature interactions.
    /// </para>
    /// <para><b>For Beginners:</b> This uses the same powerful attention mechanism
    /// found in models like GPT and BERT, allowing very sophisticated comparisons
    /// between features at multiple levels.
    /// </para>
    /// </remarks>
    Transformer
}
