using AiDotNet.Models.Options;

namespace AiDotNet.Document.Options;

/// <summary>
/// Configuration options for the CRAFT document model.
/// </summary>
public class CRAFTOptions : DocumentNeuralNetworkOptions
{
    /// <summary>
    /// Initializes a new instance with CRAFT's published defaults.
    /// </summary>
    /// <remarks>
    /// <c>ImageSize</c> and <c>BackboneChannels</c> are declared by
    /// <see cref="DocumentNeuralNetworkOptions"/> but are not universal, so the base leaves them
    /// unset. CRAFT's values lived in its constructor defaults instead, which meant the options
    /// object reported 0 for both while the model ran at 768/512. Baker et al. 2019 uses a
    /// 768x768 render and a VGG16-BN backbone whose final stage is 512 channels.
    /// </remarks>
    public CRAFTOptions()
    {
        ImageSize = 768;
        BackboneChannels = 512;
    }

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
        Require(ImageSize, nameof(ImageSize));
        Require(BackboneChannels, nameof(BackboneChannels));
    }
}
