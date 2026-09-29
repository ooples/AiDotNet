using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Options;

/// <summary>
/// Configuration options for the HopeNetwork.
/// </summary>
public class HopeNetworkOptions : ModelHyperparameterOptions
{
    /// <summary>
    /// Returns a shallow clone of these options. <see cref="HopeNetwork{T}.Clone"/>
    /// uses this so a cloned network gets its own options instance and
    /// caller-side mutation of one doesn't bleed into the other. If a
    /// future field of reference type is added to this class, override
    /// this to perform the appropriate deep copy on that field.
    /// </summary>
    public virtual HopeNetworkOptions MemberwiseCloneOptions()
        => (HopeNetworkOptions)this.MemberwiseClone();

    /// <summary>
    /// Gets or sets hidden dim. Default: <c>256</c>.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> How wide each hidden layer is.</para>
    /// </remarks>
    public int HiddenDim { get; set; } = 256;

    /// <summary>
    /// Gets or sets in context learning levels. Default: <c>5</c>.
    /// </summary>
    public int InContextLearningLevels { get; set; } = 5;

    /// <summary>
    /// Gets or sets num cmslevels. Default: <c>4</c>.
    /// </summary>
    public int NumCMSLevels { get; set; } = 4;

    /// <summary>
    /// Gets or sets num recurrent layers. Default: <c>3</c>.
    /// </summary>
    public int NumRecurrentLayers { get; set; } = 3;

    /// <summary>
    /// Throws when a value on this instance cannot produce a working model.
    /// </summary>
    public void Validate()
    {
        Require(HiddenDim, nameof(HiddenDim));
        Require(InContextLearningLevels, nameof(InContextLearningLevels));
        Require(NumCMSLevels, nameof(NumCMSLevels));
        Require(NumRecurrentLayers, nameof(NumRecurrentLayers));
    }
}
