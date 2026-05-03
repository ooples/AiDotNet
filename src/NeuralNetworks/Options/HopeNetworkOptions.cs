using AiDotNet.Models.Options;

namespace AiDotNet.NeuralNetworks.Options;

/// <summary>
/// Configuration options for the HopeNetwork.
/// </summary>
public class HopeNetworkOptions : NeuralNetworkOptions
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
}
