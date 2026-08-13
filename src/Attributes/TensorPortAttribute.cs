using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.Attributes;

/// <summary>Identifies whether a declared tensor port receives or produces a tensor.</summary>
public enum TensorPortDirection
{
    /// <summary>The port receives a tensor.</summary>
    Input,

    /// <summary>The port produces a tensor.</summary>
    Output
}

/// <summary>
/// Declares a layer tensor port once so its runtime metadata, value-domain validation and generated
/// diagnostics cannot drift into separate hand-written implementations.
/// </summary>
[AttributeUsage(AttributeTargets.Class, AllowMultiple = true, Inherited = true)]
public sealed class TensorPortAttribute : Attribute
{
    /// <summary>Creates a tensor-port declaration.</summary>
    public TensorPortAttribute(
        string name,
        TensorPortDirection direction,
        LayerInputDomainKind domain = LayerInputDomainKind.Continuous)
    {
        Name = name;
        Direction = direction;
        Domain = domain;
    }

    /// <summary>Stable port name used by named-forward APIs and diagnostics.</summary>
    public string Name { get; }

    /// <summary>Whether this is an input or output port.</summary>
    public TensorPortDirection Direction { get; }

    /// <summary>Allowed or produced value domain.</summary>
    public LayerInputDomainKind Domain { get; }

    /// <summary>Semantic role shown to model and layer authors in diagnostics.</summary>
    public TensorPortRole Role { get; set; } = TensorPortRole.Unspecified;

    /// <summary>Whether a named input is mandatory.</summary>
    public bool Required { get; set; } = true;

    /// <summary>
    /// Field or property containing the exclusive integer upper bound. Required for an
    /// <see cref="LayerInputDomainKind.IntegerIndices"/> port unless
    /// <see cref="MaxExclusiveResolver"/> is supplied.
    /// </summary>
    public string? MaxExclusiveMember { get; set; }

    /// <summary>
    /// Method accepting <c>int[]?</c> and returning the exclusive integer upper bound. This supports
    /// packed contracts whose legal range depends on the caller shape without hand-writing a port or
    /// <c>GetInputDomain</c> override.
    /// </summary>
    public string? MaxExclusiveResolver { get; set; }

    /// <summary>Stable registered provider key. Required when <see cref="Domain"/> is Custom.</summary>
    public string? CustomProviderKey { get; set; }

    /// <summary>
    /// Optional field/property/method expression supplying this port's shape. Empty uses
    /// <c>GetInputShape()</c> or <c>GetOutputShape()</c> according to <see cref="Direction"/>.
    /// </summary>
    public string? ShapeMember { get; set; }

    /// <summary>
    /// True only for an identity layer whose output retains the input value domain. The generator
    /// emits the propagation hook; individual layers do not override it.
    /// </summary>
    public bool PropagatesInputDomain { get; set; }

    /// <summary>Stable identifier retained across display-name changes and inheritance.</summary>
    public string? StableId { get; set; }

    /// <summary>Whether the value is caller-supplied, derived, defaulted or internal.</summary>
    public TensorPortSource Source { get; set; } = TensorPortSource.External;

    /// <summary>Alternative input signature. Ports in different variants are never required together.</summary>
    public string Variant { get; set; } = "default";

    public int ExactRank { get; set; }
    public int MinimumRank { get; set; }
    public int MaximumRank { get; set; }
    public int MinimumElementCount { get; set; }
    public string? SameShapeAs { get; set; }
    public int[]? MinimumAxisSizes { get; set; }
    public int[]? AxisDivisors { get; set; }
}

/// <summary>
/// Marks the one method that contains a layer's unique forward logic. Tensor parameters become
/// generated input ports; unannotated parameters use the beginner-friendly continuous default.
/// </summary>
[AttributeUsage(AttributeTargets.Method, AllowMultiple = false, Inherited = false)]
public sealed class GenerateInputContractAttribute : Attribute
{
}

/// <summary>
/// Refines the generated contract for a tensor method parameter. Most numerical layers need no
/// annotation; token IDs and masks opt into their semantic domain here once.
/// </summary>
[AttributeUsage(AttributeTargets.Parameter, AllowMultiple = false, Inherited = false)]
public sealed class TensorInputAttribute : Attribute
{
    public TensorInputAttribute(LayerInputDomainKind domain = LayerInputDomainKind.Continuous)
    {
        Domain = domain;
    }

    public LayerInputDomainKind Domain { get; }
    public string? Name { get; set; }
    public TensorPortRole Role { get; set; } = TensorPortRole.Features;
    public string? MaxExclusiveMember { get; set; }
    public string? MaxExclusiveResolver { get; set; }
    public string? CustomProviderKey { get; set; }
    public TensorPortSource Source { get; set; } = TensorPortSource.External;
    public string Variant { get; set; } = "default";
    public int ExactRank { get; set; }
    public int MinimumRank { get; set; }
    public int MaximumRank { get; set; }
    public int MinimumElementCount { get; set; }
    public string? SameShapeAs { get; set; }
    public int[]? MinimumAxisSizes { get; set; }
    public int[]? AxisDivisors { get; set; }
}

/// <summary>
/// Declares a rank-routed model input. Low-rank inputs enter an index lookup at
/// <see cref="LayerIndex"/> while higher-rank inputs enter the ordinary continuous front end.
/// </summary>
[AttributeUsage(AttributeTargets.Class, AllowMultiple = false, Inherited = false)]
public sealed class RankRoutedInputDomainAttribute : Attribute
{
    /// <summary>Creates a rank-routed declaration.</summary>
    public RankRoutedInputDomainAttribute(int maximumIndexRank, int layerIndex)
    {
        MaximumIndexRank = maximumIndexRank;
        LayerIndex = layerIndex;
    }

    /// <summary>Largest input rank routed to the lookup stream.</summary>
    public int MaximumIndexRank { get; }

    /// <summary>Index of the lookup-stream front layer in the model's registered layer list.</summary>
    public int LayerIndex { get; }
}

/// <summary>
/// Declares minimum fixture geometry from model configuration. The generator exposes the declaration
/// through <c>NeuralNetworkBase.GetInputShapeConstraint()</c>; test scaffolds consume that shared
/// contract instead of adding per-model shape overrides.
/// </summary>
[AttributeUsage(AttributeTargets.Class, AllowMultiple = false, Inherited = false)]
public sealed class ModelInputShapeConstraintAttribute : Attribute
{
    /// <summary>
    /// Exact external tensor rank. Zero means unconstrained. This is useful when a model's first
    /// layer intentionally adds the matrix axis consumed by later operations.
    /// </summary>
    public int ExactRank { get; set; }

    /// <summary>Minimum tensor rank accepted by the model. Zero means unconstrained.</summary>
    public int MinimumRank { get; set; }

    /// <summary>Maximum tensor rank accepted by the model. Zero means unconstrained.</summary>
    public int MaximumRank { get; set; }

    /// <summary>Constant minimum number of input elements. Zero means unconstrained.</summary>
    public int MinimumElementCount { get; set; }

    /// <summary>
    /// Field, property or parameterless method containing the configuration-derived minimum element
    /// count. This takes precedence over <see cref="MinimumElementCount"/>.
    /// </summary>
    public string? MinimumElementCountMember { get; set; }

    /// <summary>Per-axis minimum sizes. Zero entries leave an axis unconstrained.</summary>
    public int[]? MinimumAxisSizes { get; set; }

    /// <summary>Per-axis divisors. Zero or one entries leave an axis unconstrained.</summary>
    public int[]? AxisDivisors { get; set; }
}

/// <summary>
/// Compatibility marker retained for existing layer factories. Sequential iterator factories are
/// now checked automatically, so new code does not need this attribute.
/// </summary>
[AttributeUsage(AttributeTargets.Method, AllowMultiple = false, Inherited = false)]
public sealed class ValidateSequentialLayerDomainsAttribute : Attribute
{
}
