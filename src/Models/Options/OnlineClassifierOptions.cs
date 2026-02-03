namespace AiDotNet.Models.Options;

/// <summary>
/// Base configuration options for online (incremental) classifiers.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options control how online classifiers learn
/// from streaming data and adapt to changes over time.</para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class OnlineClassifierOptions<T> : ClassifierOptions<T>
{
    /// <summary>
    /// Gets or sets the initial number of classes.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Set this if you know the number of classes in advance.
    /// If set to 0, the model will discover classes as it sees them.</para>
    /// <para>Default: 0 (discover from data)</para>
    /// </remarks>
    public int InitialNumClasses { get; set; } = 0;

    /// <summary>
    /// Gets or sets the random seed for reproducible results.
    /// </summary>
    public int? RandomSeed { get; set; }
}

/// <summary>
/// Configuration options for Hoeffding Tree classifier.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class HoeffdingTreeOptions<T> : OnlineClassifierOptions<T>
{
    /// <summary>
    /// Gets or sets the confidence level for the Hoeffding bound.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Lower values make the tree more conservative about splitting.
    /// The Hoeffding bound guarantees that with probability (1 - delta), the chosen split
    /// is the true best split.</para>
    /// <para>Default: 1e-7</para>
    /// </remarks>
    public double Delta { get; set; } = 1e-7;

    /// <summary>
    /// Gets or sets the minimum number of samples before considering a split.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Prevents the tree from splitting on too few samples.
    /// Higher values create more stable but potentially less accurate trees.</para>
    /// <para>Default: 200</para>
    /// </remarks>
    public int GracePeriod { get; set; } = 200;

    /// <summary>
    /// Gets or sets the tie threshold for split decisions.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> If two splits have similar quality (within this threshold),
    /// the tree may split early to avoid waiting for more samples.</para>
    /// <para>Default: 0.05</para>
    /// </remarks>
    public double TieThreshold { get; set; } = 0.05;

    /// <summary>
    /// Gets or sets the maximum depth of the tree.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Limits tree growth to prevent overfitting.
    /// Set to 0 for unlimited depth.</para>
    /// <para>Default: 20</para>
    /// </remarks>
    public int MaxDepth { get; set; } = 20;

    /// <summary>
    /// Gets or sets the number of bins for numeric attribute discretization.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Hoeffding trees discretize continuous features into bins.
    /// More bins allow finer splits but use more memory.</para>
    /// <para>Default: 10</para>
    /// </remarks>
    public int NumBins { get; set; } = 10;
}

/// <summary>
/// Configuration options for Online Naive Bayes classifier.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class OnlineNaiveBayesOptions<T> : OnlineClassifierOptions<T>
{
    /// <summary>
    /// Gets or sets the Laplace smoothing parameter.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Smoothing prevents zero probabilities for unseen feature values.
    /// Higher values create more uniform probability distributions.</para>
    /// <para>Default: 1.0 (Laplace smoothing)</para>
    /// </remarks>
    public double Alpha { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets whether to use Gaussian assumption for continuous features.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> If true, assumes features follow Gaussian (normal) distributions.
    /// If false, uses histograms to estimate distributions.</para>
    /// <para>Default: true</para>
    /// </remarks>
    public bool UseGaussian { get; set; } = true;
}
