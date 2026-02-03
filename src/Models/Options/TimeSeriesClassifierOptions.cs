namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for time series classifiers.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These options configure how a time series classifier processes
/// sequence data. The most important parameters are the expected sequence length and number
/// of channels (variables) in your time series data.</para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class TimeSeriesClassifierOptions<T> : ClassifierOptions<T>
{
    /// <summary>
    /// Gets or sets the expected sequence length for input time series.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the number of time steps in each input sequence.
    /// Set to 0 to determine automatically from training data. If set, all input sequences
    /// must have this exact length (unless the classifier supports variable lengths).</para>
    /// <para>Default: 0 (determined from data)</para>
    /// </remarks>
    public int SequenceLength { get; set; } = 0;

    /// <summary>
    /// Gets or sets the number of channels (variables) in the time series.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> For univariate time series (single measurement), set to 1.
    /// For multivariate (e.g., accelerometer x, y, z), set to the number of variables.
    /// Set to 0 to determine automatically from training data.</para>
    /// <para>Default: 0 (determined from data)</para>
    /// </remarks>
    public int NumChannels { get; set; } = 0;

    /// <summary>
    /// Gets or sets whether to normalize sequences before processing.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Normalization scales each sequence to have zero mean and
    /// unit variance. This helps when different sequences have different scales but the
    /// pattern shapes are what matter for classification.</para>
    /// <para>Default: true</para>
    /// </remarks>
    public bool NormalizeSequences { get; set; } = true;

    /// <summary>
    /// Gets or sets whether to use subsequence extraction for data augmentation.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> When true, random subsequences are extracted from the
    /// original sequences during training, effectively augmenting the training data.
    /// This can improve generalization when you have limited training samples.</para>
    /// <para>Default: false</para>
    /// </remarks>
    public bool UseSubsequences { get; set; } = false;

    /// <summary>
    /// Gets or sets the length of subsequences if <see cref="UseSubsequences"/> is true.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> When extracting subsequences, this determines how long
    /// they should be. Must be less than or equal to the original sequence length.</para>
    /// <para>Default: 0 (uses 75% of original sequence length)</para>
    /// </remarks>
    public int SubsequenceLength { get; set; } = 0;

    /// <summary>
    /// Validates the options.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This checks that all option values are valid and make sense
    /// together. Called before training to catch configuration errors early.</para>
    /// </remarks>
    public virtual void Validate()
    {
        if (SequenceLength < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(SequenceLength),
                "Sequence length cannot be negative.");
        }

        if (NumChannels < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(NumChannels),
                "Number of channels cannot be negative.");
        }

        if (SubsequenceLength < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(SubsequenceLength),
                "Subsequence length cannot be negative.");
        }

        if (SequenceLength > 0 && SubsequenceLength > SequenceLength)
        {
            throw new ArgumentException(
                "Subsequence length cannot be greater than sequence length.",
                nameof(SubsequenceLength));
        }
    }
}
