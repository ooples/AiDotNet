namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for the Time Series Forest classifier.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Time Series Forest is an ensemble method that builds decision trees
/// on randomly selected intervals of the time series. Each tree looks at a different portion
/// of the sequence, and the ensemble combines their predictions for robust classification.</para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class TimeSeriesForestOptions<T> : TimeSeriesClassifierOptions<T>
{
    /// <summary>
    /// Gets or sets the number of trees in the forest.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> More trees generally improve accuracy but increase training time.
    /// The ensemble averages predictions from all trees.</para>
    /// <para>Default: 200</para>
    /// </remarks>
    public int NumTrees { get; set; } = 200;

    /// <summary>
    /// Gets or sets the minimum interval length as a fraction of sequence length.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Intervals shorter than this won't be considered.
    /// Too short intervals may capture noise rather than patterns.</para>
    /// <para>Default: 0.05 (5% of sequence length)</para>
    /// </remarks>
    public double MinIntervalFraction { get; set; } = 0.05;

    /// <summary>
    /// Gets or sets the maximum interval length as a fraction of sequence length.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Intervals longer than this won't be considered.
    /// This limits how much of the sequence each tree can see.</para>
    /// <para>Default: 0.5 (50% of sequence length)</para>
    /// </remarks>
    public double MaxIntervalFraction { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets the maximum depth of each tree.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Deeper trees can learn more complex patterns but may overfit.
    /// Set to 0 for unlimited depth.</para>
    /// <para>Default: 0 (unlimited)</para>
    /// </remarks>
    public int MaxDepth { get; set; } = 0;

    /// <summary>
    /// Gets or sets the minimum number of samples required to split a node.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Higher values prevent the tree from learning overly specific patterns.</para>
    /// <para>Default: 2</para>
    /// </remarks>
    public int MinSamplesSplit { get; set; } = 2;

    /// <summary>
    /// Gets or sets the random seed for reproducible results.
    /// </summary>
    public int? RandomSeed { get; set; }
}
