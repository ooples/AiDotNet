using AiDotNet.Models.Options;

namespace AiDotNet.Video.Options;

/// <summary>
/// Configures FastDVDnet video-denoising training and architecture settings.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These settings control how FastDVDnet learns to remove noise
/// by combining information from a five-frame temporal window.</para>
/// <para>
/// FastDVDnet uses two compact convolutional denoising stages without optical-flow estimation;
/// the standard model uses 32 feature channels.
/// </para>
/// <para><b>Reference:</b> Tassano, Delon, and Veit, “FastDVDnet: Towards Real-Time Deep
/// Video Denoising Without Flow Estimation,” 2020.</para>
/// </remarks>
public class FastDVDNetOptions : NeuralNetworkOptions
{
    /// <summary>Initializes FastDVDnet options with the released training defaults.</summary>
    public FastDVDNetOptions()
    {
    }

    /// <summary>Initializes an independent copy of another FastDVDnet configuration.</summary>
    /// <param name="other">The options to copy.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="other"/> is null.</exception>
    public FastDVDNetOptions(FastDVDNetOptions other)
    {
        if (other is null)
            throw new ArgumentNullException(nameof(other));

        Seed = other.Seed;
        EncoderLayerCount = other.EncoderLayerCount;
        LearningRate = other.LearningRate;
    }

    /// <summary>
    /// Initial Adam learning rate. The released FastDVDnet training recipe uses 1e-3.
    /// </summary>
    /// <value>A positive learning rate; the default is <c>0.001</c>.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This controls the size of each Adam update. Smaller values
    /// make gentler updates, while larger values learn faster but can become unstable.</para>
    /// <para>The default comes from the released FastDVDnet training recipe.</para>
    /// </remarks>
    public double LearningRate { get; set; } = 1e-3;
}
