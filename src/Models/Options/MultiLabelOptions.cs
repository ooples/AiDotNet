namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for ML-kNN (Multi-Label k-Nearest Neighbors) classifier.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class MLkNNOptions<T> : ClassifierOptions<T>
{
    /// <summary>
    /// Gets or sets the number of nearest neighbors to use.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> More neighbors give smoother predictions but may lose local patterns.</para>
    /// <para>Default: 10</para>
    /// </remarks>
    public int KNeighbors { get; set; } = 10;

    /// <summary>
    /// Gets or sets the smoothing parameter for probability estimation.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Prevents zero probabilities when a label combination hasn't been seen.</para>
    /// <para>Default: 1.0 (Laplace smoothing)</para>
    /// </remarks>
    public double Smoothing { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets the random seed for reproducible results.
    /// </summary>
    public int? RandomSeed { get; set; }
}

/// <summary>
/// Configuration options for RAkEL (Random k-Labelsets) classifier.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class RAkELOptions<T> : ClassifierOptions<T>
{
    /// <summary>
    /// Gets or sets the size of each labelset.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Larger labelsets capture more label correlations but
    /// increase the number of possible label combinations exponentially.</para>
    /// <para>Default: 3</para>
    /// </remarks>
    public int LabelsetSize { get; set; } = 3;

    /// <summary>
    /// Gets or sets the number of labelset classifiers in the ensemble.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> More models generally improve accuracy but increase computation.</para>
    /// <para>Default: 2 * NumLabels (computed at runtime)</para>
    /// </remarks>
    public int? NumModels { get; set; }

    /// <summary>
    /// Gets or sets the random seed for reproducible results.
    /// </summary>
    public int? RandomSeed { get; set; }
}
