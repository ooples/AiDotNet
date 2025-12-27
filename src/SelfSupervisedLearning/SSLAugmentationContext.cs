namespace AiDotNet.SelfSupervisedLearning;

/// <summary>
/// Context for SSL augmentation operations.
/// </summary>
/// <typeparam name="T">The numeric type used for computations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> This provides information and state for creating augmented views.
/// SSL methods typically create multiple views of each input for contrastive/self-supervised learning.</para>
/// </remarks>
public class SSLAugmentationContext<T>
{
    /// <summary>
    /// Gets or sets whether this is the first view (used in multi-view methods).
    /// </summary>
    public bool IsFirstView { get; set; } = true;

    /// <summary>
    /// Gets or sets the view index for methods that use more than 2 views.
    /// </summary>
    public int ViewIndex { get; set; }

    /// <summary>
    /// Gets or sets the total number of views being generated.
    /// </summary>
    public int TotalViews { get; set; } = 2;

    /// <summary>
    /// Gets or sets optional pre-computed augmentations.
    /// </summary>
    public Tensor<T>? PrecomputedView { get; set; }

    /// <summary>
    /// Gets or sets the random seed for reproducible augmentations.
    /// </summary>
    public int? Seed { get; set; }

    /// <summary>
    /// Gets or sets the augmentation strength multiplier.
    /// </summary>
    public double StrengthMultiplier { get; set; } = 1.0;
}
