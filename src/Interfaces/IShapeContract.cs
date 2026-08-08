using System;
using System.Collections.Generic;
using AiDotNet.Enums;

namespace AiDotNet.Interfaces;

/// <summary>
/// How ONE output axis is sized from the input axes and the declaring type's own configuration.
/// </summary>
/// <remarks>
/// <para>
/// The symbolic half of the shape system. <c>[TensorLayout]</c> says WHICH axes a layer has and in what
/// order; this says HOW BIG each output axis is. Roles alone cannot distinguish a stride-1 convolution
/// from a stride-2 one — both declare <c>[Batch, Channels, Height, Width]</c> in and out — yet one halves
/// the spatial extent. Without the relation, "automatic shape inference" can only propagate names.
/// </para>
/// <para>
/// A RELATION, NOT A CONSTANT. Every form below is evaluated against the actual input at resolve time
/// and reads the declaring instance's real options, so a layer built with <c>stride: 2</c> reports the
/// halving and one built with <c>stride: 1</c> does not. That is the whole reason this is an interface
/// rather than more attribute arguments: C# attributes take compile-time constants only, and kernel,
/// stride, padding and scale factor are all constructor arguments. A declaration that cannot see the
/// configuration it describes is a declaration that goes stale the first time someone passes a different
/// stride.
/// </para>
/// </remarks>
public sealed class AxisRelation
{
    /// <summary>The forms a relation can take.</summary>
    public enum Form
    {
        /// <summary>Copied unchanged from an input axis.</summary>
        Same,

        /// <summary>Set by configuration, independent of the input (a convolution's output channels).</summary>
        Fixed,

        /// <summary>An input axis times a rational factor (upsampling by 2, downsampling by a stride).</summary>
        Scaled,

        /// <summary>The sliding-window formula — kernel, stride, padding, dilation.</summary>
        Window,

        /// <summary>The product of several input axes, i.e. a flatten.</summary>
        Product,

        /// <summary>Genuinely not derivable from the input shape.</summary>
        Unknown,
    }

    private readonly TensorAxis[] _sources;

    private AxisRelation(
        Form kind,
        TensorAxis[] sources,
        int value = 0,
        int numerator = 1,
        int denominator = 1,
        int kernel = 1,
        int stride = 1,
        int padding = 0,
        int dilation = 1,
        string? reason = null)
    {
        Kind = kind;
        _sources = sources;
        Value = value;
        Numerator = numerator;
        Denominator = denominator;
        Kernel = kernel;
        Stride = stride;
        Padding = padding;
        Dilation = dilation;
        Reason = reason;
    }

    /// <summary>Which form this relation takes.</summary>
    public Form Kind { get; }

    /// <summary>Input axes this output axis is computed from.</summary>
    public IReadOnlyList<TensorAxis> Sources => _sources;

    /// <summary>The size, for <see cref="Form.Fixed"/>.</summary>
    public int Value { get; }

    /// <summary>Scale numerator, for <see cref="Form.Scaled"/>.</summary>
    public int Numerator { get; }

    /// <summary>Scale denominator, for <see cref="Form.Scaled"/>.</summary>
    public int Denominator { get; }

    /// <summary>Window extent, for <see cref="Form.Window"/>.</summary>
    public int Kernel { get; }

    /// <summary>Window step, for <see cref="Form.Window"/>.</summary>
    public int Stride { get; }

    /// <summary>Padding added to BOTH sides, for <see cref="Form.Window"/>.</summary>
    public int Padding { get; }

    /// <summary>Window dilation, for <see cref="Form.Window"/>.</summary>
    public int Dilation { get; }

    /// <summary>Why the size is unknown, for <see cref="Form.Unknown"/>.</summary>
    public string? Reason { get; }

    /// <summary>This output axis copies an input axis unchanged.</summary>
    public static AxisRelation Same(TensorAxis source)
        => new(Form.Same, new[] { source });

    /// <summary>This output axis has a size the configuration fixes, whatever the input.</summary>
    public static AxisRelation Fixed(int size)
    {
        if (size <= 0) throw new ArgumentOutOfRangeException(nameof(size), size, "A fixed axis size must be positive.");
        return new AxisRelation(Form.Fixed, Array.Empty<TensorAxis>(), value: size);
    }

    /// <summary>This output axis is an input axis scaled by <paramref name="numerator"/>/<paramref name="denominator"/>.</summary>
    public static AxisRelation Scaled(TensorAxis source, int numerator, int denominator = 1)
    {
        if (numerator <= 0) throw new ArgumentOutOfRangeException(nameof(numerator), numerator, "Scale numerator must be positive.");
        if (denominator <= 0) throw new ArgumentOutOfRangeException(nameof(denominator), denominator, "Scale denominator must be positive.");
        return new AxisRelation(Form.Scaled, new[] { source }, numerator: numerator, denominator: denominator);
    }

    /// <summary>
    /// This output axis follows the sliding-window formula
    /// <c>floor((in + 2*padding - dilation*(kernel-1) - 1) / stride) + 1</c>.
    /// </summary>
    /// <remarks>
    /// Spelled out rather than reduced to a scale factor because stride alone is wrong at the boundary:
    /// a 3x3 stride-2 convolution over 32 pixels with padding 1 yields 16, but over 33 it yields 17, and
    /// with padding 0 it yields 15. Approximating this as "divide by stride" is right in the common case
    /// and quietly off by one everywhere else — which is exactly the kind of error a shape system is
    /// supposed to catch rather than introduce.
    /// </remarks>
    public static AxisRelation Window(TensorAxis source, int kernel, int stride, int padding, int dilation = 1)
    {
        if (kernel <= 0) throw new ArgumentOutOfRangeException(nameof(kernel), kernel, "Kernel must be positive.");
        if (stride <= 0) throw new ArgumentOutOfRangeException(nameof(stride), stride, "Stride must be positive.");
        if (padding < 0) throw new ArgumentOutOfRangeException(nameof(padding), padding, "Padding cannot be negative.");
        if (dilation <= 0) throw new ArgumentOutOfRangeException(nameof(dilation), dilation, "Dilation must be positive.");
        return new AxisRelation(
            Form.Window, new[] { source }, kernel: kernel, stride: stride, padding: padding, dilation: dilation);
    }

    /// <summary>This output axis is the product of several input axes — a flatten.</summary>
    public static AxisRelation Product(params TensorAxis[] sources)
    {
        if (sources is null) throw new ArgumentNullException(nameof(sources));
        if (sources.Length == 0) throw new ArgumentException("A product needs at least one source axis.", nameof(sources));
        return new AxisRelation(Form.Product, (TensorAxis[])sources.Clone());
    }

    /// <summary>
    /// This output axis genuinely cannot be derived from the input shape.
    /// </summary>
    /// <remarks>
    /// Honest and useful: a CTC decode emits one token per COLLAPSED repeat run, and a detector emits one
    /// row per instance over threshold. Both are data-dependent. Declaring <c>Unknown</c> with a reason
    /// makes that visible and stops inference at a named boundary, which beats both a fabricated formula
    /// and silence.
    /// </remarks>
    public static AxisRelation Unknown(string reason)
    {
        if (string.IsNullOrWhiteSpace(reason))
            throw new ArgumentException("An unknown axis must say why it is unknown.", nameof(reason));
        return new AxisRelation(Form.Unknown, Array.Empty<TensorAxis>(), reason: reason);
    }

    /// <summary>
    /// Computes this axis's size from the input axis sizes.
    /// </summary>
    /// <param name="inputAxes">Input axis role to size.</param>
    /// <param name="size">The resolved size.</param>
    /// <returns><c>false</c> when a source axis is absent or the relation is <see cref="Form.Unknown"/>.</returns>
    public bool TryResolve(IReadOnlyDictionary<TensorAxis, int> inputAxes, out int size)
    {
        if (inputAxes is null) throw new ArgumentNullException(nameof(inputAxes));
        size = 0;

        switch (Kind)
        {
            case Form.Fixed:
                size = Value;
                return true;

            case Form.Same:
                return inputAxes.TryGetValue(_sources[0], out size) && size > 0;

            case Form.Scaled:
            {
                if (!inputAxes.TryGetValue(_sources[0], out int from) || from <= 0) return false;
                // Integer division deliberately: a scale that does not divide evenly is a declaration
                // error, not something to round past silently.
                if ((long)from * Numerator % Denominator != 0) return false;
                size = (int)((long)from * Numerator / Denominator);
                return size > 0;
            }

            case Form.Window:
            {
                if (!inputAxes.TryGetValue(_sources[0], out int from) || from <= 0) return false;
                long effective = (long)Dilation * (Kernel - 1) + 1;
                long numerator = from + (2L * Padding) - effective;
                if (numerator < 0) return false;   // the window does not fit; not a shape, an error
                size = (int)(numerator / Stride) + 1;
                return size > 0;
            }

            case Form.Product:
            {
                long product = 1;
                foreach (var axis in _sources)
                {
                    if (!inputAxes.TryGetValue(axis, out int from) || from <= 0) return false;
                    product *= from;
                    if (product > int.MaxValue) return false;
                }
                size = (int)product;
                return true;
            }

            default:
                return false;
        }
    }

    /// <inheritdoc />
    public override string ToString() => Kind switch
    {
        Form.Same => $"in.{_sources[0]}",
        Form.Fixed => Value.ToString(System.Globalization.CultureInfo.InvariantCulture),
        Form.Scaled => Denominator == 1
            ? $"{Numerator} * in.{_sources[0]}"
            : $"{Numerator} * in.{_sources[0]} / {Denominator}",
        Form.Window => $"floor((in.{_sources[0]} + 2*{Padding} - {Dilation}*({Kernel}-1) - 1) / {Stride}) + 1",
        Form.Product => string.Join(" * ", Array.ConvertAll(_sources, a => $"in.{a}")),
        _ => $"unknown ({Reason})",
    };
}

/// <summary>
/// Declares, symbolically, how a layer or model sizes each of its output axes.
/// </summary>
/// <remarks>
/// <para>
/// Implement this on anything whose output extent depends on its configuration — every convolution,
/// pooling, upsampling, patch-embedding and super-resolution type. With it, an output shape can be
/// COMPUTED from an input shape without running a forward pass, which is what makes shape inference,
/// chain validation and automatic test-fixture shapes possible rather than hand-maintained.
/// </para>
/// <para>
/// THE DECLARATION IS CHECKED, NOT TRUSTED. A conformance test resolves the declared relations and
/// compares them against the shape the type's real forward produces. A relation that drifts from the
/// implementation fails there, which is the only thing that keeps a declaration worth reading — an
/// unverified shape annotation is worse than none, because the next reader believes it.
/// </para>
/// </remarks>
public interface IShapeContract
{
    /// <summary>
    /// One entry per output axis, in output order, for an input of the given rank.
    /// </summary>
    /// <param name="inputRank">Rank of the incoming tensor.</param>
    /// <returns>The output axes and their relations, or <c>null</c> if this rank is not accepted.</returns>
    /// <remarks>
    /// <para>
    /// PARAMETERIZED BY RANK because the output axes genuinely depend on it. A dense layer over
    /// <c>[Batch, Features]</c> emits two axes; the same layer over <c>[Batch, Time, Features]</c> emits
    /// three, and the extra one is a passthrough that simply does not exist in the rank-2 case. The same
    /// is true of every batch-optional layer: fed <c>[C,H,W]</c> a convolution emits three axes, not four
    /// with a fabricated batch. A single fixed axis list would force one of those forms to be declared
    /// wrongly.
    /// </para>
    /// <para>
    /// This deliberately mirrors <c>TensorLayoutAttribute.AxesForRank</c>, so the roles and the relations
    /// are asked the same question in the same way and can be compared entry by entry.
    /// </para>
    /// <para>
    /// Order matters and must match the type's <c>[TensorLayout(Direction = Output)]</c> declaration at
    /// the corresponding rank. The two are cross-checked: a contract listing axes in a different order
    /// than the layout, or a different set of them, is a contradiction between two claims by one type.
    /// </para>
    /// </remarks>
    IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank);
}

/// <summary>One output axis, and the relation that sizes it.</summary>
/// <param name="Axis">The axis's role.</param>
/// <param name="Relation">How it is sized from the input.</param>
public readonly record struct OutputAxisContract(TensorAxis Axis, AxisRelation Relation);
