namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>Semantic purpose of a tensor port, used in generated diagnostics and tooling.</summary>
public enum TensorPortRole
{
    Unspecified,
    Features,
    TokenIds,
    PositionIds,
    TokenTypeIds,
    Mask,
    EncoderInput,
    EncoderMemory,
    DecoderIds,
    AudioCodes,
    Output
}

/// <summary>Where a tensor port obtains its value.</summary>
public enum TensorPortSource
{
    /// <summary>The caller must supply the tensor.</summary>
    External,

    /// <summary>The layer derives the tensor from another input.</summary>
    Derived,

    /// <summary>The layer supplies a default when the caller omits the tensor.</summary>
    Defaulted,

    /// <summary>The tensor exists only inside the component graph.</summary>
    Internal
}

/// <summary>
/// A small, decidable shape language shared by generated contracts, runtime validation and test
/// synthesis. Zero-valued limits mean unconstrained.
/// </summary>
public sealed record PortShapeConstraint
{
    /// <summary>No extra shape restriction.</summary>
    public static PortShapeConstraint None { get; } = new();

    public int ExactRank { get; init; }
    public int MinimumRank { get; init; }
    public int MaximumRank { get; init; }
    public int MinimumElementCount { get; init; }

    /// <summary>Optional port whose complete shape must match this port.</summary>
    public string? SameShapeAs { get; init; }

    /// <summary>Per-axis minimum sizes. Zero entries leave the corresponding axis unconstrained.</summary>
    public IReadOnlyList<int> MinimumAxisSizes { get; init; } = Array.Empty<int>();

    /// <summary>Per-axis divisors. Zero or one leaves the corresponding axis unconstrained.</summary>
    public IReadOnlyList<int> AxisDivisors { get; init; } = Array.Empty<int>();

    public bool IsConstrained => ExactRank > 0 || MinimumRank > 0 || MaximumRank > 0
        || MinimumElementCount > 0 || !string.IsNullOrWhiteSpace(SameShapeAs)
        || MinimumAxisSizes.Any(value => value > 0) || AxisDivisors.Any(value => value > 1);
}

/// <summary>
/// Declares a named input or output port on a layer.
/// Ports enable multi-input layers (e.g., DiffusionResBlock needs "input" + "time_embed")
/// and provide compile-time documentation of a layer's data contract.
/// </summary>
/// <param name="Name">Port name (e.g., "input", "time_embed", "query", "key", "value").</param>
/// <param name="Shape">Expected tensor shape for this port.</param>
/// <param name="Required">If true, Forward throws when this port is missing. Default: true.</param>
/// <remarks>
/// <para><b>For Beginners:</b> A port is like a labeled plug on the layer.
/// Just as a TV has separate ports for HDMI, USB, and power, a neural network layer
/// can have separate ports for different types of input data.</para>
/// </remarks>
public sealed record LayerPort
{
    public string Name { get; }
    public IReadOnlyList<int> Shape { get; }
    public bool Required { get; }
    public LayerInputDomain ValueDomain { get; }
    public TensorPortRole Role { get; }
    public string StableId { get; }
    public TensorPortSource Source { get; }
    public string Variant { get; }
    public PortShapeConstraint ShapeConstraint { get; }

    public LayerPort(
        string Name,
        int[] Shape,
        bool Required = true,
        LayerInputDomain? ValueDomain = null,
        TensorPortRole Role = TensorPortRole.Unspecified,
        string? StableId = null,
        TensorPortSource Source = TensorPortSource.External,
        string Variant = "default",
        PortShapeConstraint? ShapeConstraint = null)
    {
        this.Name = Name ?? throw new ArgumentNullException(nameof(Name));
        // Defensive copy to prevent callers from mutating the layer's shape
        this.Shape = Shape != null ? (int[])Shape.Clone() : throw new ArgumentNullException(nameof(Shape));
        this.Required = Required;
        this.ValueDomain = ValueDomain ?? LayerInputDomain.Continuous;
        this.Role = Role;
        this.StableId = string.IsNullOrWhiteSpace(StableId) ? this.Name : StableId!;
        this.Source = Source;
        this.Variant = string.IsNullOrWhiteSpace(Variant) ? "default" : Variant;
        this.ShapeConstraint = ShapeConstraint ?? PortShapeConstraint.None;
    }
}
