using AiDotNet.Models.Options;

namespace AiDotNet.Document.Options;

/// <summary>
/// Configuration options for the CRAFT document model.
/// </summary>
public class CRAFTOptions : DocumentNeuralNetworkOptions
{

    /// <summary>
    /// Gets or sets upscale channels. Default: <c>256</c>.
    /// </summary>
    public int UpscaleChannels { get; set; } = 256;

    /// <summary>
    /// Throws when a value on this instance cannot produce a working model.
    /// </summary>
    public void Validate()
    {
        Require(UpscaleChannels, nameof(UpscaleChannels));
    }
}
