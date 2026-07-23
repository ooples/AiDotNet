namespace AiDotNet.ContinualLearning.Interfaces;

/// <summary>
/// Memory sampling strategies for experience replay.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> When storing examples from previous tasks, the sampling
/// strategy determines how examples are selected and maintained in memory.</para>
/// </remarks>
public enum MemorySamplingStrategy
{
    /// <summary>
    /// Reservoir sampling - uniform random selection.
    /// Each item has equal probability of being selected.
    /// </summary>
    Reservoir,

    /// <summary>
    /// Random uniform sampling from the dataset.
    /// Simple but effective for homogeneous data.
    /// </summary>
    Random,

    /// <summary>
    /// Ring buffer - FIFO replacement.
    /// </summary>
    RingBuffer,

    /// <summary>
    /// Class-balanced sampling ensures equal representation of each class.
    /// Best for imbalanced datasets to prevent bias toward majority classes.
    /// </summary>
    ClassBalanced,

    /// <summary>
    /// Herding-based selection picks examples closest to class means.
    /// From iCaRL paper - provides exemplars that well represent the class distribution.
    /// </summary>
    Herding,

    /// <summary>
    /// K-Center coreset selection maximizes coverage of the feature space.
    /// Picks examples that minimize maximum distance to any unselected point.
    /// </summary>
    KCenter,

    /// <summary>
    /// Boundary-focused sampling selects examples near decision boundaries.
    /// Prioritizes hard-to-classify examples for better discrimination.
    /// </summary>
    Boundary,

    /// <summary>
    /// Gradient-based sample selection.
    /// </summary>
    GradientBased
}
