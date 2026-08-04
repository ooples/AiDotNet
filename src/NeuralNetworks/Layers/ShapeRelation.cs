namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// How a layer's output shape is determined by its input shape, declared per LAYER TYPE rather than
/// per model instance.
/// </summary>
/// <remarks>
/// <para>
/// A declared output shape on its own cannot say whether it is wrong or merely describing a
/// different input than the one that arrived. A relation can: if a layer states that it preserves
/// shape, then a declared output of <c>[8, 32, 32]</c> is a claim that its input was
/// <c>[8, 32, 32]</c>, and an actual input of <c>[8, 8, 8]</c> proves the DECLARATION's provenance
/// wrong rather than the layer. That distinction is the whole point: it moves the finger from the
/// layer, which computed correctly, to whatever wrote the declaration.
/// </para>
/// <para>
/// Relations are per layer type and there are only a few shapes of them — most layers preserve
/// their input, and the rest follow a one-line formula — so unlike per-model declared numbers they
/// do not have to be re-established every time a new model is added.
/// </para>
/// <para>
/// <b>For Beginners:</b> Instead of a layer writing down "my output is 8x32x32", it writes down the
/// RULE it follows — "my output is the same size as my input", or "my output shrinks by the stride".
/// A rule can be checked against reality; a number can only be compared to one.
/// </para>
/// <para>
/// Beyond PyTorch: <c>nn.Module</c> declares no shape and states no rule, so a mis-wired stride
/// chain surfaces only as a runtime error inside some later kernel, with nothing to identify which
/// layer's assumption was violated. <c>torch.fx</c> ShapeProp and the meta device can tell you what
/// a graph DID produce, but neither lets a module state what it is supposed to produce and be held
/// to it.
/// </para>
/// </remarks>
public enum ShapeRelationKind
{
    /// <summary>
    /// The layer states nothing about how output relates to input; no check is performed.
    /// </summary>
    /// <remarks>
    /// The default, so adding relations is opt-in and a layer that has not declared one behaves
    /// exactly as it did before.
    /// </remarks>
    Unknown = 0,

    /// <summary>
    /// Output shape equals input shape, axis for axis — normalization, activation, dropout.
    /// </summary>
    Identity,

    /// <summary>
    /// Leading (channel) axis is set by the layer; the remaining spatial axes are preserved.
    /// </summary>
    /// <remarks>
    /// Covers 1x1 projections and any layer that re-maps channels without touching spatial extent.
    /// </remarks>
    ChannelOnly,

    /// <summary>
    /// Leading axis is set by the layer; spatial axes follow the convolution formula
    /// <c>floor((in + 2*padding - kernel) / stride) + 1</c>.
    /// </summary>
    Convolutional,
}
