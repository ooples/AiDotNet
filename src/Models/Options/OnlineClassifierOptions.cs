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

/// <summary>
/// Configuration options for Adaptive Random Forest classifier.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Adaptive Random Forest (ARF) is an ensemble method that combines
/// multiple Hoeffding trees with drift detection to handle evolving data streams.</para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class AdaptiveRandomForestOptions<T> : OnlineClassifierOptions<T>
{
    /// <summary>
    /// Gets or sets the number of trees in the ensemble.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> More trees generally improve accuracy but increase computation.
    /// 10-100 trees is typical for most applications.</para>
    /// <para>Default: 10</para>
    /// </remarks>
    public int NumTrees { get; set; } = 10;

    /// <summary>
    /// Gets or sets the number of features to consider for each tree.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Each tree uses a random subset of features for diversity.
    /// Set to 0 to use sqrt(total_features), which is a common default.</para>
    /// <para>Default: 0 (auto)</para>
    /// </remarks>
    public int NumFeaturesPerTree { get; set; } = 0;

    /// <summary>
    /// Gets or sets the lambda parameter for Poisson resampling.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Controls sample weight distribution. Higher values give
    /// more weight diversity but may increase variance.</para>
    /// <para>Default: 6.0</para>
    /// </remarks>
    public double LambdaPoisson { get; set; } = 6.0;

    /// <summary>
    /// Gets or sets the warning threshold for drift detection (DDM).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> When error rate exceeds this many standard deviations above
    /// the minimum, a warning is triggered and background tree training begins.</para>
    /// <para>Default: 2.0</para>
    /// </remarks>
    public double WarningThreshold { get; set; } = 2.0;

    /// <summary>
    /// Gets or sets the drift threshold for tree replacement (DDM).
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> When error rate exceeds this many standard deviations above
    /// the minimum, drift is confirmed and the tree is replaced.</para>
    /// <para>Default: 3.0</para>
    /// </remarks>
    public double DriftThreshold { get; set; } = 3.0;

    /// <summary>
    /// Gets or sets the Hoeffding bound confidence parameter for individual trees.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Controls how confident the trees need to be before splitting.
    /// Lower values require more samples before splitting.</para>
    /// <para>Default: 1e-7</para>
    /// </remarks>
    public double HoeffdingDelta { get; set; } = 1e-7;

    /// <summary>
    /// Gets or sets the minimum samples before attempting splits.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Trees won't consider splitting until they've seen this many samples.
    /// Higher values create more stable splits.</para>
    /// <para>Default: 200</para>
    /// </remarks>
    public int GracePeriod { get; set; } = 200;

    /// <summary>
    /// Gets or sets the tie threshold for tree split decisions.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> If two splits are this close in quality, the tree may split
    /// early instead of waiting for more evidence.</para>
    /// <para>Default: 0.05</para>
    /// </remarks>
    public double TieThreshold { get; set; } = 0.05;

    /// <summary>
    /// Gets or sets the maximum depth for individual trees.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Limits how deep each tree can grow. Set to 0 for unlimited.</para>
    /// <para>Default: 20</para>
    /// </remarks>
    public int MaxTreeDepth { get; set; } = 20;

    /// <summary>
    /// Gets or sets the number of bins for numeric attribute discretization.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Each tree discretizes continuous features into this many bins.</para>
    /// <para>Default: 10</para>
    /// </remarks>
    public int NumBins { get; set; } = 10;
}
