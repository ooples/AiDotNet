using AiDotNet.Models.Options;

namespace AiDotNet.Diffusion.Conditioning;

/// <summary>
/// Transformer dimensions for a text conditioner, overriding the ones its variant selects.
/// </summary>
/// <remarks>
/// <para>
/// Every property is nullable because each conditioner's variant already fixes a paper value for
/// it: <c>null</c> means "use the variant's value", which differs per variant, rather than a
/// single default this class could name. Set a property only to build a non-paper size, such as a
/// small test-scale encoder.
/// </para>
/// <para><b>For Beginners:</b> Leave this out to get the published model size. Set, for example,
/// <c>HiddenSize = 256</c> and <c>NumLayers = 2</c> to build a much smaller text encoder.</para>
/// </remarks>
public class TextConditionerOptions : ModelOptions
{
    /// <summary>Initializes a new instance that keeps every variant value.</summary>
    public TextConditionerOptions() { }

    /// <summary>Initializes a new instance by copying another.</summary>
    /// <param name="other">The options to copy.</param>
    public TextConditionerOptions(TextConditionerOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));
        Seed = other.Seed;
        HiddenSize = other.HiddenSize;
        NumLayers = other.NumLayers;
        NumHeads = other.NumHeads;
        NumKvHeads = other.NumKvHeads;
    }

    /// <summary>Transformer width; the conditioner's embedding dimension follows it. Null keeps the variant's value.</summary>
    public int? HiddenSize { get; set; }

    /// <summary>Transformer depth. Null keeps the variant's value.</summary>
    public int? NumLayers { get; set; }

    /// <summary>Attention heads; must divide <see cref="HiddenSize"/>. Null keeps the variant's value.</summary>
    public int? NumHeads { get; set; }

    /// <summary>
    /// Key/value heads for grouped-query conditioners (ChatGLM3, Qwen2); must divide
    /// <see cref="NumHeads"/>. Ignored by conditioners without grouped-query attention. Null keeps
    /// the variant's value.
    /// </summary>
    public int? NumKvHeads { get; set; }
}
