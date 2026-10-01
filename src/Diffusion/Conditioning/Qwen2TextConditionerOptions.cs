using AiDotNet.Enums;
using AiDotNet.Models.Options;

namespace AiDotNet.Diffusion.Conditioning;

/// <summary>
/// Configuration options for the Qwen2TextConditioner text conditioner.
/// </summary>
/// <remarks>
/// <para>
/// Introduced by issue #2090: the variant was a constructor parameter, so the conditioner
/// advertised no configuration surface and nothing could be set through an options object.
/// </para>
/// </remarks>
public class Qwen2TextConditionerOptions : TextConditionerOptions
{
    /// <summary>
    /// Gets or sets which published size of the model to build. Default: <c>Qwen2Variant.OnePointFiveB</c>.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The same architecture is usually published at several sizes. A bigger
    /// one is more accurate and slower; a smaller one fits where the big one will not. Picking a
    /// variant selects the widths and depths the paper reports for it - it changes the network that
    /// gets built, not merely how it is labelled.
    /// </para>
    /// </remarks>
    public Qwen2Variant Variant { get; set; } = Qwen2Variant.OnePointFiveB;

    /// <summary>
    /// Key/value heads for grouped-query attention; must divide <see cref="TextConditionerOptions.NumHeads"/>.
    /// Null keeps the variant's value.
    /// </summary>
    public int? NumKvHeads { get; set; }

    /// <summary>
    /// Throws when a value on this instance cannot produce a working conditioner.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <see cref="Variant"/> is an enum, so every value it can hold is buildable. The dimension
    /// overrides are checked: each one that is set must be positive.
    /// </para>
    /// </remarks>
    public void Validate()
    {
        ValidateDimensionOverrides();
        RequireIfSet(NumKvHeads, nameof(NumKvHeads));
    }
}
