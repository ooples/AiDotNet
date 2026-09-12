using AiDotNet.Enums;
using AiDotNet.Models.Options;

namespace AiDotNet.Diffusion.Conditioning;

/// <summary>
/// Configuration options for the T5TextConditioner text conditioner.
/// </summary>
/// <remarks>
/// <para>
/// Introduced by issue #2090: the variant was a constructor parameter, so the conditioner
/// advertised no configuration surface and nothing could be set through an options object.
/// </para>
/// </remarks>
public class T5TextConditionerOptions : ModelHyperparameterOptions
{
    /// <summary>
    /// Gets or sets which published size of the model to build. Default: <c>T5Variant.Base</c>.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The same architecture is usually published at several sizes. A bigger
    /// one is more accurate and slower; a smaller one fits where the big one will not. Picking a
    /// variant selects the widths and depths the paper reports for it - it changes the network that
    /// gets built, not merely how it is labelled.
    /// </para>
    /// </remarks>
    public T5Variant Variant { get; set; } = T5Variant.Base;

    /// <summary>
    /// Throws when a value on this instance cannot produce a working conditioner.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <see cref="Variant"/> is an enum, so every value it can hold is buildable and there is
    /// nothing to reject. The method exists so this class states that deliberately rather than by
    /// omission.
    /// </para>
    /// </remarks>
    public void Validate()
    {
    }
}
