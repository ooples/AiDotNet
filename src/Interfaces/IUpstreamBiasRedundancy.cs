namespace AiDotNet.Interfaces;

/// <summary>
/// Optional capability for a layer that can make an immediately preceding additive bias redundant.
/// </summary>
/// <remarks>
/// This is intentionally separate from <see cref="ILayer{T}"/>. Bias elimination is an optional
/// optimization capability, and adding it to the core public layer interface would break every
/// external type that implements that interface directly.
/// </remarks>
public interface IUpstreamBiasRedundancy
{
    /// <summary>Whether an immediately preceding layer's additive bias may be omitted.</summary>
    bool MakesUpstreamBiasRedundant { get; }
}
