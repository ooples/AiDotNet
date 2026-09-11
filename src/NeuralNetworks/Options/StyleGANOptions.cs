using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Options;

/// <summary>
/// Configuration options for the StyleGAN neural network.
/// </summary>
public class StyleGANOptions : NeuralNetworkOptions
{
    /// <summary>
    /// Weight gamma of the R1 regulariser, (gamma / 2) * E[ ||grad D(x)||^2 ] over real images.
    /// </summary>
    /// <value>10, the value Karras et al. (2019) use with the non-saturating loss. Zero disables R1.</value>
    public double R1Gamma { get; set; } = 10.0;

    /// <summary>
    /// Factor applied to the learning rate for the mapping network only.
    /// </summary>
    /// <value>
    /// 0.01: Karras et al. (2019) found the mapping network unstable at the generator's rate and
    /// reduced its learning rate by two orders of magnitude.
    /// </value>
    public double MappingLearningRateMultiplier { get; set; } = 0.01;
}
