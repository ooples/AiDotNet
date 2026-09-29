using AiDotNet.Models.Options;

namespace AiDotNet.Diffusion.Conditioning;

/// <summary>
/// Transformer dimensions shared by every text conditioner's options, overriding the ones its
/// variant selects.
/// </summary>
/// <remarks>
/// <para>
/// Every property is nullable, unlike the non-nullable pattern <see cref="ModelHyperparameterOptions"/>
/// describes, because each conditioner's variant already fixes a paper value for it: <c>null</c>
/// means "use the variant's value", which differs per variant and is only known once
/// <c>Variant</c> is read, so no parameterless constructor could assign it. Set a property only
/// to build a non-paper size, such as a small test-scale encoder.
/// </para>
/// <para>
/// Key/value heads are not declared here: only the grouped-query conditioners (ChatGLM3, Qwen2)
/// read them, so they live on those two options classes rather than advertising a knob the
/// other six ignore.
/// </para>
/// <para><b>For Beginners:</b> Leave these unset to get the published model size. Set, for
/// example, <c>HiddenSize = 256</c> and <c>NumLayers = 2</c> to build a much smaller text
/// encoder.</para>
/// </remarks>
public abstract class TextConditionerOptions : ModelHyperparameterOptions
{
    /// <summary>Transformer width; the conditioner's embedding dimension follows it. Null keeps the variant's value.</summary>
    public int? HiddenSize { get; set; }

    /// <summary>Transformer depth. Null keeps the variant's value.</summary>
    public int? NumLayers { get; set; }

    /// <summary>Attention heads; must divide <see cref="HiddenSize"/>. Null keeps the variant's value.</summary>
    public int? NumHeads { get; set; }

    /// <summary>
    /// Throws when a dimension override is set to a value no transformer can be built with.
    /// </summary>
    /// <remarks>
    /// Divisibility of the width by the head count is checked by the conditioner instead, because
    /// either side may be the variant's value, which only the conditioner knows.
    /// </remarks>
    protected void ValidateDimensionOverrides()
    {
        RequireIfSet(HiddenSize, nameof(HiddenSize));
        RequireIfSet(NumLayers, nameof(NumLayers));
        RequireIfSet(NumHeads, nameof(NumHeads));
    }

    /// <summary>Requires a positive value when an override is set; null keeps the variant's value.</summary>
    /// <param name="value">The override, or null.</param>
    /// <param name="propertyName">The property name reported on failure.</param>
    protected void RequireIfSet(int? value, string propertyName)
    {
        if (value.HasValue)
            Require(value.Value, propertyName);
    }
}
