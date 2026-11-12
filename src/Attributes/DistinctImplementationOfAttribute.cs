namespace AiDotNet.Attributes;

/// <summary>
/// Declares that a model and another model citing the same paper under the same name are different networks.
/// </summary>
/// <remarks>
/// <para>
/// A paper sometimes defines a family rather than one network: Lumina-T2X generates images and videos with
/// differently configured transformers, and WaveNet is both an audio vocoder and a forecasting network. Two models
/// that implement different parts of such a paper are not duplicates.
/// </para>
/// <para>
/// The analyzer checks the claim against the code: if the two models share a network that no model of another paper
/// uses, they are one network and must say so with <see cref="TaskVariantOfAttribute"/> instead (ADNDEF005).
/// </para>
/// </remarks>
[AttributeUsage(AttributeTargets.Class, AllowMultiple = true, Inherited = false)]
public sealed class DistinctImplementationOfAttribute : Attribute
{
    /// <summary>The other model, as an open generic type (for example <c>typeof(WaveNet&lt;&gt;)</c>).</summary>
    public Type Other { get; }

    /// <summary>What the paper defines that makes the two networks different.</summary>
    public string Reason { get; }

    /// <summary>Creates the declaration.</summary>
    public DistinctImplementationOfAttribute(Type other, string reason)
    {
        if (other is null) throw new ArgumentNullException(nameof(other));
        if (reason is null) throw new ArgumentNullException(nameof(reason));
        if (string.IsNullOrWhiteSpace(reason))
            throw new ArgumentException("Why the two networks differ must be stated.", nameof(reason));
        Other = other;
        Reason = reason;
    }
}
