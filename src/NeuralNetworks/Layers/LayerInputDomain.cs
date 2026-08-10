namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// The kind of values a layer's input tensor is allowed to hold.
/// </summary>
public enum LayerInputDomainKind
{
    /// <summary>Any real value. The default for almost every layer.</summary>
    Continuous,

    /// <summary>
    /// Integer token indices in a half-open range. An embedding table in lookup mode
    /// is the canonical case.
    /// </summary>
    IntegerIndices
}

/// <summary>
/// Describes what a layer will ACCEPT in its input tensor, so callers can generate or
/// validate conforming data without inspecting the layer's type.
/// </summary>
/// <remarks>
/// <para>
/// THE COMPLEMENT TO SHAPE-ONLY MODE RESOLUTION, and the reason this type exists rather
/// than the alternative. <see cref="EmbeddingLayer{T}"/> deliberately resolves
/// Indices-vs-Continuous from the input SHAPE alone: inferring it from the VALUES made the
/// layer's output RANK a function of the data, which left the shape contract, chain
/// validation and graph resolution unable to reason about it at all.
/// </para>
/// <para>
/// That rule is correct and this type does not weaken it. It runs the other direction:
/// mode is never inferred from data, so instead the DATA is derived from the declared mode.
/// A caller asks the layer what it accepts and produces values that conform. No value
/// inspection happens anywhere, and the output rank stays a function of shape.
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
}
