namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// The kind of values a layer's input tensor is allowed to hold.
/// </summary>
public enum LayerInputDomainKind
{
    /// <summary>
    /// The producer preserves the domain of a named input. This is a relationship, not a wildcard;
    /// it must be resolved from the graph before a constrained consumer can execute.
    /// </summary>
    Unspecified,

    /// <summary>Any real value. The default for almost every layer.</summary>
    Continuous,

    /// <summary>
    /// Integer token indices in a half-open range. An embedding table is the canonical case.
    /// </summary>
    IntegerIndices,

    /// <summary>A mask whose elements represent false/true membership.</summary>
    BooleanMask,

    /// <summary>An attention mask represented by zero and non-positive additive bias values.</summary>
    AdditiveMask,

    /// <summary>
    /// The domain is known to exist but cannot be bound from the current configuration yet.
    /// Deferred is never executable and never silently degrades to continuous values.
    /// </summary>
    Deferred,

    /// <summary>A user-defined domain whose validator and synthesizer are supplied explicitly.</summary>
    Custom
}

/// <summary>The result of proving whether one value domain can feed another.</summary>
public enum LayerInputDomainCompatibility
{
    /// <summary>The producer is proven to be accepted by the consumer.</summary>
    Compatible,

    /// <summary>The producer is proven not to satisfy the consumer.</summary>
    Incompatible,

    /// <summary>A preserve/deferred/custom relationship must be bound before it can be decided.</summary>
    Deferred
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

    /// <summary>
    /// Source port for a preserve relationship, explanation for a deferred domain, or stable key
    /// for a custom domain. Empty for built-in resolved domains.
    /// </summary>
    public string? Detail { get; }

    private LayerInputDomain(
        LayerInputDomainKind kind,
        int minInclusive,
        int maxExclusive,
        string? detail = null)
    {
        Kind = kind;
        MinInclusive = minInclusive;
        MaxExclusive = maxExclusive;
        Detail = detail;
    }

    /// <summary>Any real value. The default for a layer that expresses no opinion.</summary>
    public static LayerInputDomain Continuous { get; } =
        new LayerInputDomain(LayerInputDomainKind.Continuous, 0, 0);

    /// <summary>
    /// Preserves the primary input domain. Kept under the original name for source compatibility;
    /// unlike the old implementation it is not accepted as an unproved wildcard.
    /// </summary>
    public static LayerInputDomain Unspecified { get; } =
        Preserve("input");

    /// <summary>Preserves the value domain of <paramref name="sourcePort"/>.</summary>
    public static LayerInputDomain Preserve(string sourcePort) =>
        new(
            LayerInputDomainKind.Unspecified,
            0,
            0,
            string.IsNullOrWhiteSpace(sourcePort) ? "input" : sourcePort);

    /// <summary>A Boolean mask represented by the numeric tensor type.</summary>
    public static LayerInputDomain BooleanMask { get; } =
        new LayerInputDomain(LayerInputDomainKind.BooleanMask, 0, 2);

    /// <summary>An additive attention mask: zero keeps an element and a negative value masks it.</summary>
    public static LayerInputDomain AdditiveMask { get; } =
        new LayerInputDomain(LayerInputDomainKind.AdditiveMask, 0, 0);

    /// <summary>
    /// Integer indices in <c>[0, vocabularySize)</c>.
    /// </summary>
    /// <param name="vocabularySize">Exclusive upper bound.</param>
    /// <remarks>
    /// A non-positive bound is deferred rather than treated as continuous. Changing index lookup
    /// into a numerical projection because configuration is incomplete was the silent fallback
    /// behind the D1 failures; callers now receive an actionable binding error instead.
    /// </remarks>
    public static LayerInputDomain Indices(int vocabularySize) =>
        vocabularySize > 0
            ? new LayerInputDomain(LayerInputDomainKind.IntegerIndices, 0, vocabularySize)
            : Deferred($"The exclusive integer bound is {vocabularySize}; it must be positive before input can be created.");

    /// <summary>Creates a non-executable domain with an actionable explanation.</summary>
    public static LayerInputDomain Deferred(string reason) =>
        new(
            LayerInputDomainKind.Deferred,
            0,
            0,
            string.IsNullOrWhiteSpace(reason) ? "The value domain has not been bound yet." : reason);

    /// <summary>Creates a custom domain identified by a stable provider key.</summary>
    public static LayerInputDomain Custom(string providerKey) =>
        string.IsNullOrWhiteSpace(providerKey)
            ? Deferred("A custom value domain must declare a non-empty provider key.")
            : new LayerInputDomain(LayerInputDomainKind.Custom, 0, 0, providerKey);

    /// <summary>True when this domain constrains values to integer indices.</summary>
    public bool IsIndices => Kind == LayerInputDomainKind.IntegerIndices && MaxExclusive > 0;

    /// <summary>Whether the domain can be validated and synthesized without further binding.</summary>
    public bool IsResolved => Kind is LayerInputDomainKind.Continuous
        or LayerInputDomainKind.IntegerIndices
        or LayerInputDomainKind.BooleanMask
        or LayerInputDomainKind.AdditiveMask
        || Kind == LayerInputDomainKind.Custom
           && InputDomainProviderRegistry.TryResolve(Detail, out _);

    /// <summary>Whether the domain represents an identity relationship to another port.</summary>
    public bool IsPreserved => Kind == LayerInputDomainKind.Unspecified;

    /// <summary>Proves whether this consumer accepts values produced under <paramref name="producer"/>.</summary>
    public LayerInputDomainCompatibility CompatibilityWith(LayerInputDomain producer)
    {
        if (!IsResolved || !producer.IsResolved)
            return LayerInputDomainCompatibility.Deferred;

        if (Kind == LayerInputDomainKind.Continuous)
            return LayerInputDomainCompatibility.Compatible;

        if (Kind == LayerInputDomainKind.Custom)
            return InputDomainProviderRegistry.Require(Detail).CompatibilityWith(producer);

        if (Kind != producer.Kind)
            return LayerInputDomainCompatibility.Incompatible;

        if (Kind != LayerInputDomainKind.IntegerIndices)
            return LayerInputDomainCompatibility.Compatible;

        return producer.MinInclusive >= MinInclusive && producer.MaxExclusive <= MaxExclusive
            ? LayerInputDomainCompatibility.Compatible
            : LayerInputDomainCompatibility.Incompatible;
    }

    /// <summary>
    /// Whether this consumer is already proven to accept <paramref name="producer"/>. Preserve,
    /// deferred and custom relations return false until a graph or provider binds them.
    /// </summary>
    public bool Accepts(LayerInputDomain producer) =>
        CompatibilityWith(producer) == LayerInputDomainCompatibility.Compatible;

    /// <inheritdoc />
    public override string ToString() => Kind == LayerInputDomainKind.IntegerIndices
        ? $"integer indices in [{MinInclusive}, {MaxExclusive})"
        : Kind switch
        {
            LayerInputDomainKind.BooleanMask => "a Boolean mask",
            LayerInputDomainKind.AdditiveMask => "an additive attention mask",
            LayerInputDomainKind.Unspecified => $"the domain preserved from '{Detail ?? "input"}'",
            LayerInputDomainKind.Deferred => $"a deferred domain ({Detail})",
            LayerInputDomainKind.Custom => $"the custom domain '{Detail}'",
            _ => "continuous values"
        };
}
