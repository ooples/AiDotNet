namespace AiDotNet.Attributes;

/// <summary>
/// Marks a constructor parameter as state that must survive a serialization round-trip, so the
/// layer can be rebuilt exactly rather than guessed at from its shape.
/// </summary>
/// <remarks>
/// <para>
/// Deserialization used to reverse-engineer constructor arguments out of the saved input shape —
/// <c>int maxSeqLen = inputShape[0];</c> and 92 more like it. That only ever worked because layers
/// encoded their capacity in their declared shape. The moment a layer told the truth about an axis
/// being dynamic, the same line handed the constructor a <c>-1</c>.
/// </para>
/// <para>
/// A parameter marked with this attribute is written to the layer's metadata by generated code and
/// read back by a generated factory, so the constructor receives the value it was originally given.
/// The generator resolves the value through a backing field or property matching the parameter name
/// (<c>name</c>, <c>_name</c>, <c>m_name</c> or <c>Name</c>) and <b>fails the build</b> when no such
/// member exists — a layer that cannot round-trip does not compile.
/// </para>
/// <para>
/// Parameters left unmarked are still handled: activation functions are restored from the activation
/// metadata the base layer already writes, and optional parameters fall back to their defaults. A
/// required, unmarked, non-activation parameter is a build error, because nothing could supply it.
/// </para>
/// <para><b>For Beginners:</b> When a model is saved and loaded, each layer has to be built again
/// from scratch. This attribute marks the numbers the constructor needs so they get written down at
/// save time, instead of the loader trying to work them out from the data's shape and getting it
/// wrong.</para>
/// </remarks>
/// <example>
/// <code>
/// public PositionalEncodingLayer(
///     [LayerState] int maxSequenceLength,
///     [LayerState] int embeddingSize)
///     : base([-1, embeddingSize], [-1, embeddingSize])
/// </code>
/// </example>
[AttributeUsage(AttributeTargets.Parameter, AllowMultiple = false, Inherited = false)]
public sealed class LayerStateAttribute : Attribute
{
    /// <summary>
    /// Overrides the metadata key used for this parameter. Defaults to the parameter name.
    /// </summary>
    /// <remarks>
    /// Set this only to match a key an existing hand-written <c>GetMetadata</c> already writes.
    /// Lookup is case-insensitive, so a <c>inputChannels</c> parameter already finds an
    /// <c>"InputChannels"</c> key without needing this.
    /// </remarks>
    public string? Key { get; set; }
}
