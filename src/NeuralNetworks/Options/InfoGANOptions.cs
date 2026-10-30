using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Options;

/// <summary>
/// Configuration options for the InfoGAN neural network.
/// </summary>
public class InfoGANOptions : GanOptions
{
    /// <summary>
    /// Initializes a new instance of the <see cref="InfoGANOptions"/> class carrying the paper's MNIST
    /// defaults.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> You do not need to set any of these. Chen et al. 2016 (section 7) give the MNIST
    /// generator one categorical code with ten values, which learns the digit, and two continuous codes,
    /// which learn rotation and width; those are the defaults.
    /// </para>
    /// </remarks>
    public InfoGANOptions()
    {
        NumCategoricalCodes = 1;
        CategoricalCodeDimension = 10;
        NumContinuousCodes = 2;
        MutualInfoCoefficient = 1.0;
    }

    /// <summary>
    /// Gets or sets how many categorical codes the generator is given. Default: 1.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Each categorical code is sampled uniformly from <see cref="CategoricalCodeDimension"/> values and fed to
    /// the generator one-hot. The Q network predicts it with a softmax, and its mutual-information term is the
    /// cross-entropy of that prediction (Chen et al. 2016, section 5).
    /// </para>
    /// <para><b>For Beginners:</b> A categorical code is a dial with a fixed number of positions. On MNIST the
    /// one ten-position dial ends up choosing the digit.</para>
    /// </remarks>
    public int NumCategoricalCodes { get; set; }

    /// <summary>
    /// Gets or sets how many values each categorical code takes. Default: 10.
    /// </summary>
    public int CategoricalCodeDimension { get; set; }

    /// <summary>
    /// Gets or sets how many continuous codes the generator is given. Default: 2.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Each continuous code is sampled from Uniform(-1, 1). The Q network predicts it as the mean of a
    /// fixed-variance Gaussian, so its mutual-information term is the squared error of that prediction.
    /// </para>
    /// <para><b>For Beginners:</b> A continuous code is a slider. On MNIST the two sliders end up controlling
    /// rotation and stroke width.</para>
    /// </remarks>
    public int NumContinuousCodes { get; set; }

    /// <summary>
    /// Gets the width of the whole latent code: each categorical code one-hot, then the continuous codes.
    /// </summary>
    public int LatentCodeSize => NumCategoricalCodes * CategoricalCodeDimension + NumContinuousCodes;

    /// <summary>
    /// Gets or sets the mutual info coefficient.
    /// </summary>
    public double MutualInfoCoefficient { get; set; }

    /// <summary>
    /// Throws if a value this model requires has been left unset or is not positive.
    /// </summary>
    /// <exception cref="ArgumentException">
    /// Thrown when a code count is negative, a categorical code has fewer than two values, or there is no code
    /// at all.
    /// </exception>
    public void Validate()
    {
        // These values come from the model's constructor arguments and architectures, not these options.
        ValidateCore(requireLatentSize: false, requireChannels: false);
        if (NumCategoricalCodes < 0 || NumContinuousCodes < 0)
        {
            throw new ArgumentException(
                $"Code counts cannot be negative; got NumCategoricalCodes={NumCategoricalCodes}, NumContinuousCodes={NumContinuousCodes}.");
        }

        if (NumCategoricalCodes > 0 && CategoricalCodeDimension < 2)
        {
            throw new ArgumentException(
                $"A categorical code needs at least two values; got CategoricalCodeDimension={CategoricalCodeDimension}.");
        }

        if (LatentCodeSize <= 0)
        {
            throw new ArgumentException("InfoGAN needs at least one latent code, categorical or continuous.");
        }
    }
}
