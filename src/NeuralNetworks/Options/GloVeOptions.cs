using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Options;

/// <summary>
/// Configuration options for the GloVe model.
/// </summary>
public class GloVeOptions : EmbeddingModelOptions
{
    /// <summary>
    /// Initializes a new instance of the <see cref="GloVeOptions"/> class carrying
    /// this model's shipped defaults.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> You do not need to set any of these. They are the values this
    /// model has always used, moved here from its constructor so they can be seen and
    /// changed in one place.
    /// </para>
    /// <para>
    /// Carried over unchanged. Whether each matches the published paper is verified, and
    /// corrected where it does not, in a later phase of issue #2090.
    /// </para>
    /// </remarks>
    public GloVeOptions()
    {
        VocabSize = 10000;
        EmbeddingDimension = 100;
        MaxSequenceLength = 512;
        MaxGradNorm = 1.0;
    }

    /// <summary>
    /// Gets or sets the symmetric context window, in words, used when co-occurrences are counted from token
    /// sequences. A pair d words apart adds 1/d to X_ij (Pennington et al. 2014, Section 4.2, which uses 10).
    /// </summary>
    public int WindowSize { get; set; } = 10;

    /// <summary>
    /// Gets or sets x_max, the co-occurrence count at which the weighting f(X_ij) of Eq. 9 reaches 1 (the paper uses 100).
    /// </summary>
    public double XMax { get; set; } = 100.0;

    /// <summary>
    /// Gets or sets α, the exponent of the weighting f(X_ij) = (X_ij / x_max)^α below x_max (the paper uses 3/4).
    /// </summary>
    public double Alpha { get; set; } = 0.75;


    /// <summary>
    /// Throws if a value this model requires has been left unset or is not positive.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// Thrown when a required dimension is zero or negative.
    /// </exception>
    public void Validate()
    {
        ValidateCore();
        if (WindowSize <= 0) throw new ArgumentException("WindowSize must be positive.", nameof(WindowSize));
        if (!(XMax > 0)) throw new ArgumentException("XMax must be positive.", nameof(XMax));
        if (!(Alpha > 0)) throw new ArgumentException("Alpha must be positive.", nameof(Alpha));
    }
}
