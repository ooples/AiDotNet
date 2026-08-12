namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// The kind of values a layer's input tensor is allowed to hold.
/// </summary>
public enum LayerInputDomainKind
{
    /// <summary>The producer intentionally leaves the value domain unchanged or unknown.</summary>
    Unspecified,

    /// <summary>Any real value. The default for almost every layer.</summary>
    Continuous,

    /// <summary>
    /// Integer token indices in a half-open range. An embedding table is the canonical case.
    /// </summary>
    IntegerIndices,

    /// <summary>A mask whose elements represent false/true membership.</summary>
    BooleanMask
}

/// <summary>
/// Describes what a layer will ACCEPT in its input tensor, so callers can generate or
/// validate conforming data without inspecting the layer's type.
/// </summary>
/// <remarks>
/// <para>
/// This contract complements a layer's shape contract. <see cref="EmbeddingLayer{T}"/>
/// is an index lookup by construction, while continuous feature projection is represented
/// explicitly by a projection layer. Callers therefore do not need to inspect values or
/// guess a mode from tensor shape.
/// </para>
/// <para>
/// Data flows from the declared contract: a caller asks the layer what it accepts and
/// produces values that conform. No value inspection changes layer behavior, and output
/// rank remains a function of the declared layer and input shape.
/// </para>
/// <para>
/// Without this, the only way to feed a token model was to hand-write a per-model override
/// that knew the vocabulary size -- which is the hand-maintained surface the parameter
/// automation exists to delete. A layer already knows its own domain; asking it is free.
/// </para>
/// </remarks>
public readonly struct LayerInputDomain
{
    /// <summary>Whether the input holds continuous values or integer indices.</summary>
    public LayerInputDomainKind Kind { get; }

    /// <summary>Smallest legal index. Zero for <see cref="LayerInputDomainKind.Continuous"/>.</summary>
    public int MinInclusive { get; }

    /// <summary>
    /// One past the largest legal index (the vocabulary size). Zero for
    /// <see cref="LayerInputDomainKind.Continuous"/>.
    /// </summary>
    public int MaxExclusive { get; }

    private LayerInputDomain(LayerInputDomainKind kind, int minInclusive, int maxExclusive)
    {
        Kind = kind;
        MinInclusive = minInclusive;
        MaxExclusive = maxExclusive;
    }

    /// <summary>Any real value. The default for a layer that expresses no opinion.</summary>
    public static LayerInputDomain Continuous { get; } =
        new LayerInputDomain(LayerInputDomainKind.Continuous, 0, 0);

    /// <summary>No value-domain opinion; accepted by every consumer.</summary>
    public static LayerInputDomain Unspecified { get; } =
        new LayerInputDomain(LayerInputDomainKind.Unspecified, 0, 0);

    /// <summary>A Boolean mask represented by the numeric tensor type.</summary>
    public static LayerInputDomain BooleanMask { get; } =
        new LayerInputDomain(LayerInputDomainKind.BooleanMask, 0, 2);

    /// <summary>
    /// Integer indices in <c>[0, vocabularySize)</c>.
    /// </summary>
    /// <param name="vocabularySize">
    /// Exclusive upper bound. A non-positive value yields <see cref="Continuous"/>: a layer
    /// whose vocabulary is not sized yet cannot constrain anything, and returning a degenerate
    /// empty range would make every value illegal rather than every value allowed.
    /// </param>
    public static LayerInputDomain Indices(int vocabularySize) =>
        vocabularySize > 0
            ? new LayerInputDomain(LayerInputDomainKind.IntegerIndices, 0, vocabularySize)
            : Continuous;

    /// <summary>True when this domain constrains values to integer indices.</summary>
    public bool IsIndices => Kind == LayerInputDomainKind.IntegerIndices && MaxExclusive > 0;

    /// <summary>Whether this consumer domain accepts values produced under <paramref name="producer"/>.</summary>
    public bool Accepts(LayerInputDomain producer) =>
        producer.Kind == LayerInputDomainKind.Unspecified
        || Kind == LayerInputDomainKind.Continuous
        || Kind == producer.Kind;

    /// <inheritdoc />
    public override string ToString() => Kind == LayerInputDomainKind.IntegerIndices
        ? $"integer indices in [{MinInclusive}, {MaxExclusive})"
        : Kind switch
        {
            LayerInputDomainKind.BooleanMask => "a Boolean mask",
            LayerInputDomainKind.Unspecified => "an unspecified pass-through domain",
            _ => "continuous values"
        };
}
