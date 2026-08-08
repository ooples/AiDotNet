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

        /// <summary>
        /// The product of several DERIVED relations, i.e. a flatten over axes that were themselves
        /// transformed first.
        /// </summary>
        /// <remarks>
        /// <see cref="Product"/> multiplies RAW input axes, which is not enough for the common
        /// patch-embedding shape: <c>numPatches = (H / patch) * (W / patch)</c> windows each spatial
        /// axis and only then multiplies. Two independent layers - SwinPatchEmbeddingLayer and
        /// STCConnectorLayer - were left undeclared for exactly this, so the vocabulary was extended
        /// rather than the layers documented as inexpressible.
        /// </remarks>
        ProductOfRelations,

        /// <summary>An axis of a SPECIFIC input port, for layers with more than one input.</summary>
        /// <remarks>
        /// Every other form reads input port 0, which is all a single-input layer has. A concatenation
        /// sizes its output from the second and later inputs too, and without a way to name them the
        /// relation is not merely hard to write - it is unsayable.
        /// </remarks>
        PortAxis,

        /// <summary>The sum of several relations — what a concatenation does to its joined axis.</summary>
        SumOfRelations,

        /// <summary>A DERIVED relation scaled by a ratio — the counterpart of <see cref="Scaled"/>.</summary>
        /// <remarks>
        /// <see cref="Scaled"/> takes one raw input axis, so it cannot scale something already computed.
        /// A layer that flattens its input and re-batches by a fixed width needs exactly that:
        /// <c>batch = (all input axes multiplied) / featureWidth</c>. With this and
        /// <see cref="ProductOfRelations"/>, relations compose in both directions.
        /// </remarks>
        ScaledRelation,

        /// <summary>Genuinely not derivable from the input shape.</summary>
        Unknown,
    }

    private readonly TensorAxis[] _sources;

    /// <summary>Factors, for <see cref="Form.ProductOfRelations"/>. Null for every other form.</summary>
    private readonly AxisRelation[]? _factors;

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
        _factors = null;
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
    /// This output axis is the product of several DERIVED relations — a flatten over axes that are
    /// each transformed first.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The patch-embedding shape needs this and nothing simpler can state it:
    /// <c>numPatches = (H / patch) * (W / patch)</c> is a <see cref="Window"/> on each spatial axis and
    /// only THEN a product. <see cref="Product"/> multiplies raw input axes, so it cannot express the
    /// windowing; <see cref="Scaled"/> takes one source, so it cannot express the product.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> a "patch embedding" chops an image into equal tiles and treats each tile as
    /// one token. A 224x224 image with 16x16 patches becomes 14 x 14 = 196 tokens. The 14s come from
    /// dividing each side by the patch size, and the 196 from multiplying them — two different kinds of
    /// step, which is why this relation composes rather than doing it in one.
    /// </para>
    /// <para>
    /// Factors compose freely, so a product of products is legal and resolves depth-first. A factor that
    /// cannot resolve makes the whole product decline, which is the correct behaviour: half a product is
    /// not a size.
    /// </para>
    /// </remarks>
    public static AxisRelation ProductOf(params AxisRelation[] factors)
    {
        if (factors is null) throw new ArgumentNullException(nameof(factors));
        if (factors.Length == 0)
            throw new ArgumentException("A product needs at least one factor.", nameof(factors));
        foreach (var factor in factors)
        {
            if (factor is null)
                throw new ArgumentException("A product factor cannot be null.", nameof(factors));
        }

        // Sources are flattened up from the factors so callers that inspect Sources - the discovery
        // fitter and the diagnostics - still see which input axes this output depends on.
        var sources = new List<TensorAxis>();
        foreach (var factor in factors)
        {
            foreach (var source in factor.Sources)
            {
                if (!sources.Contains(source)) sources.Add(source);
            }
        }

        return new AxisRelation(sources.ToArray(), (AxisRelation[])factors.Clone());
    }

    /// <summary>
    /// This output axis is another RELATION scaled by <paramref name="numerator"/>/<paramref name="denominator"/>.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The counterpart of <see cref="Scaled"/>, which only takes a raw input axis. A decoder that
    /// flattens whatever arrives and re-batches it by a fixed feature width computes
    /// <c>batch = (every input axis multiplied) / featureWidth</c> — a product first, a division second.
    /// </para>
    /// <para>
    /// Division is exact, matching <see cref="Scaled"/>: a ratio that does not divide evenly declines
    /// rather than rounding, because a rounded shape is a wrong shape stated confidently.
    /// </para>
    /// </remarks>
    public static AxisRelation ScaledBy(AxisRelation source, int numerator, int denominator = 1)
    {
        if (source is null) throw new ArgumentNullException(nameof(source));
        if (numerator <= 0)
            throw new ArgumentOutOfRangeException(nameof(numerator), numerator, "Scale numerator must be positive.");
        if (denominator <= 0)
            throw new ArgumentOutOfRangeException(nameof(denominator), denominator, "Scale denominator must be positive.");

        var sources = new TensorAxis[source.Sources.Count];
        for (int i = 0; i < sources.Length; i++) sources[i] = source.Sources[i];

        return new AxisRelation(Form.ScaledRelation, sources, new[] { source }, numerator, denominator);
    }

    /// <summary>
    /// This output axis copies an axis of a SPECIFIC input port.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <see cref="Same"/> and friends all read input port 0 — the only input a single-input layer has.
    /// A layer with several inputs needs to name the others, and until it could, a concatenation's
    /// joined axis was not expressible at all: <c>OutputAxesFor</c> saw one rank and no second shape.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> "port" is just which input slot a tensor arrives on. A layer that joins two
    /// tensors has port 0 and port 1, and its output size along the joined axis is the two input sizes
    /// added together — so the relation has to be able to point at each one separately.
    /// </para>
    /// </remarks>
    public static AxisRelation FromPort(int port, TensorAxis axis)
    {
        if (port < 0) throw new ArgumentOutOfRangeException(nameof(port), port, "A port index cannot be negative.");
        return new AxisRelation(Form.PortAxis, new[] { axis }, value: port);
    }

    /// <summary>
    /// This output axis is the SUM of several relations — what a concatenation does to its joined axis.
    /// </summary>
    /// <remarks>
    /// Concatenating <c>[B, 8, D]</c> and <c>[B, 5, D]</c> along the middle axis gives <c>[B, 13, D]</c>.
    /// Every other composed form multiplies or scales; joining adds, and nothing else in the vocabulary
    /// could say so.
    /// </remarks>
    public static AxisRelation SumOf(params AxisRelation[] terms)
    {
        if (terms is null) throw new ArgumentNullException(nameof(terms));
        if (terms.Length == 0) throw new ArgumentException("A sum needs at least one term.", nameof(terms));
        foreach (var term in terms)
        {
            if (term is null) throw new ArgumentException("A sum term cannot be null.", nameof(terms));
        }

        var sources = new List<TensorAxis>();
        foreach (var term in terms)
        {
            foreach (var source in term.Sources)
            {
                if (!sources.Contains(source)) sources.Add(source);
            }
        }

        return new AxisRelation(Form.SumOfRelations, sources.ToArray(), (AxisRelation[])terms.Clone(), 1, 1);
    }

    /// <summary>Builds the composed forms — the only ones carrying sub-relations.</summary>
    private AxisRelation(
        TensorAxis[] sources,
        AxisRelation[] factors)
        : this(Form.ProductOfRelations, sources, factors, 1, 1)
    {
    }

    private AxisRelation(
        Form kind,
        TensorAxis[] sources,
        AxisRelation[] factors,
        int numerator,
        int denominator)
    {
        Kind = kind;
        _sources = sources;
        _factors = factors;
        Value = 0;
        Numerator = numerator;
        Denominator = denominator;
        Kernel = 1;
        Stride = 1;
        Padding = 0;
        Dilation = 1;
        Reason = null;
    }

    /// <summary>The factors, for <see cref="Form.ProductOfRelations"/>; empty for every other form.</summary>
    public IReadOnlyList<AxisRelation> Factors
        => _factors ?? (IReadOnlyList<AxisRelation>)Array.Empty<AxisRelation>();

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
        return TryResolve(new[] { inputAxes }, out size);
    }

    /// <summary>
    /// Computes this axis's size from SEVERAL input ports' axis sizes.
    /// </summary>
    /// <param name="ports">One axis-role-to-size map per input port, in port order.</param>
    /// <param name="size">The resolved size.</param>
    /// <returns><c>false</c> when a source axis or port is absent, or the relation is unknown.</returns>
    /// <remarks>
    /// The single-port overload delegates here, so there is exactly one resolution implementation and a
    /// multi-input relation cannot drift from the single-input one it generalises.
    /// </remarks>
    public bool TryResolve(IReadOnlyList<IReadOnlyDictionary<TensorAxis, int>> ports, out int size)
    {
        if (ports is null) throw new ArgumentNullException(nameof(ports));
        size = 0;
        if (ports.Count == 0) return false;

        // Every form except PortAxis reads port 0 - the only input a single-input layer has.
        var inputAxes = ports[0];
        if (inputAxes is null) return false;

        switch (Kind)
        {
            case Form.PortAxis:
            {
                if (Value < 0 || Value >= ports.Count) return false;
                var port = ports[Value];
                return port is not null && port.TryGetValue(_sources[0], out size) && size > 0;
            }

            case Form.SumOfRelations:
            {
                if (_factors is null || _factors.Length == 0) return false;
                long total = 0;
                foreach (var term in _factors)
                {
                    if (!term.TryResolve(ports, out int from) || from <= 0) return false;
                    total += from;
                    if (total > int.MaxValue) return false;
                }
                size = (int)total;
                return size > 0;
            }
        }

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

            case Form.ScaledRelation:
            {
                if (_factors is null || _factors.Length != 1) return false;
                // ports, not inputAxes: a scaled relation may wrap one that reads a later port.
                if (!_factors[0].TryResolve(ports, out int from) || from <= 0) return false;
                // Exact division, as Form.Scaled: an uneven ratio is a declaration error, not a rounding.
                if ((long)from * Numerator % Denominator != 0) return false;
                size = (int)((long)from * Numerator / Denominator);
                return size > 0;
            }

            case Form.ProductOfRelations:
            {
                if (_factors is null || _factors.Length == 0) return false;
                long product = 1;
                foreach (var factor in _factors)
                {
                    // Depth-first, and over ALL ports: a factor may itself be a product, or may read a
                    // later port. One that declines makes the whole product decline rather than
                    // resolving to a partial size.
                    if (!factor.TryResolve(ports, out int from) || from <= 0) return false;
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
        Form.ProductOfRelations => "(" + string.Join(") * (",
            Array.ConvertAll(_factors ?? Array.Empty<AxisRelation>(), f => f.ToString())) + ")",
        Form.PortAxis => $"in[{Value}].{_sources[0]}",
        Form.SumOfRelations => string.Join(" + ",
            Array.ConvertAll(_factors ?? Array.Empty<AxisRelation>(), f => f.ToString())),
        Form.ScaledRelation => Denominator == 1
            ? $"({_factors?[0]}) * {Numerator}"
            : $"({_factors?[0]}) * {Numerator} / {Denominator}",
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

    /// <summary>
    /// One entry per output axis, for an input of the given rank, saying whether that rank INCLUDES a
    /// batch axis.
    /// </summary>
    /// <param name="inputRank">Rank of the incoming shape.</param>
    /// <param name="isBatched">
    /// <c>true</c> when the leading axis is a batch - a real tensor handed to <c>Forward</c>;
    /// <c>false</c> for a PER-SAMPLE shape, which is what chain resolution propagates.
    /// </param>
    /// <returns>The output axes and their relations, or <c>null</c> if this case is not accepted.</returns>
    /// <remarks>
    /// <para>
    /// Defaults to the rank-only form, so every existing contract answers exactly as before and only a
    /// layer that genuinely needs the distinction has to override.
    /// </para>
    /// <para>
    /// WHY RANK ALONE IS NOT ENOUGH. The same rank means different things to different callers.
    /// <c>Forward</c> receives a batched tensor, so a rank-3 <c>[3,8,9]</c> is one batch axis and two
    /// feature axes. Chain resolution propagates PER-SAMPLE shapes, so a rank-3 <c>[32,7,7]</c> is
    /// <c>[Channels, Height, Width]</c> with no batch at all. FlattenLayer collapses everything after
    /// the batch, so it answers <c>[3,72]</c> for the first and <c>[1568]</c> for the second - both
    /// correct, and unreachable through a signature that cannot tell them apart. Whichever reading such
    /// a contract picked, the other caller saw a wrong shape.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> a "batch" is a stack of samples processed together. Some shapes include
    /// that stacking axis and some describe a single sample, and the numbers alone do not say which -
    /// so the caller has to.
    /// </para>
    /// </remarks>
    // Declared on IBatchAwareShapeContract below, NOT as a default interface method here: this
    // assembly targets net471, where the runtime cannot express one (CS8701). See the remarks on
    // IBatchAwareShapeContract for how a caller reaches it uniformly on every target.
}

/// <summary>
/// Opt-in for a contract whose answer depends on whether the incoming rank INCLUDES a batch axis.
/// </summary>
/// <remarks>
/// <para>
/// Almost no layer needs this, so it is a separate interface rather than a member every contract has
/// to think about. Call it through <see cref="ShapeContractExtensions.OutputAxesFor(IShapeContract,int,bool)"/>,
/// which falls back to the rank-only form for the contracts that do not implement this - so a caller
/// never has to test for it.
/// </para>
/// </remarks>
public interface IBatchAwareShapeContract : IShapeContract
{
    /// <summary>
    /// One entry per output axis, for an input of the given rank, saying whether that rank INCLUDES a
    /// batch axis.
    /// </summary>
    /// <param name="inputRank">Rank of the incoming shape.</param>
    /// <param name="isBatched">
    /// <c>true</c> when the leading axis is a batch - a real tensor handed to <c>Forward</c>;
    /// <c>false</c> for a PER-SAMPLE shape, which is what chain resolution propagates.
    /// </param>
    /// <returns>The output axes and their relations, or <c>null</c> if this case is not accepted.</returns>
    /// <remarks>
    /// <para>
    /// WHY RANK ALONE IS NOT ENOUGH. The same rank means different things to different callers.
    /// <c>Forward</c> receives a batched tensor, so a rank-3 <c>[3,8,9]</c> is one batch axis and two
    /// feature axes. Chain resolution propagates PER-SAMPLE shapes, so a rank-3 <c>[32,7,7]</c> is
    /// <c>[Channels, Height, Width]</c> with no batch at all. FlattenLayer collapses everything after
    /// the batch, so it answers <c>[3,72]</c> for the first and <c>[1568]</c> for the second - both
    /// correct, and unreachable through a signature that cannot tell them apart. Whichever reading such
    /// a contract picked, the other caller saw a wrong shape.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> a "batch" is a stack of samples processed together. Some shapes include
    /// that stacking axis and some describe a single sample, and the numbers alone do not say which -
    /// so the caller has to.
    /// </para>
    /// </remarks>
    IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank, bool isBatched);
}

/// <summary>
/// Opt-in for a contract whose output depends on MORE THAN ONE input port.
/// </summary>
/// <remarks>
/// Reach it through <see cref="ShapeContractExtensions.OutputAxesForPorts"/>, which falls back to the
/// single-input form when exactly one port is fed, so the ~290 single-input contracts need no change.
/// </remarks>
public interface IMultiPortShapeContract : IShapeContract
{
    /// <summary>
    /// One entry per output axis, for a layer fed SEVERAL inputs of the given ranks.
    /// </summary>
    /// <param name="inputRanks">Rank of each incoming tensor, in port order.</param>
    /// <returns>The output axes and their relations, or <c>null</c> if this combination is not accepted.</returns>
    /// <remarks>
    /// <para>
    /// Defaults to the single-input form, so the 290-odd existing contracts need no change: a layer
    /// with one input answers exactly as before, and one with several declines unless it overrides.
    /// </para>
    /// <para>
    /// WHY THIS EXISTS. <see cref="OutputAxesFor(int)"/> is handed one rank and can consult one shape,
    /// which is genuinely enough for almost every layer. It is not enough for a join: concatenating
    /// <c>[B,8,D]</c> and <c>[B,5,D]</c> gives <c>[B,13,D]</c>, and 13 is not a function of either
    /// input alone. Layers like ConcatenateLayer were previously skipped as "inexpressible" when the
    /// truth was narrower - the relation was fine, the interface could not ask the question. Use
    /// <see cref="AxisRelation.FromPort"/> to name a later input's axis and
    /// <see cref="AxisRelation.SumOf"/> to add them.
    /// </para>
    /// </remarks>
    IReadOnlyList<OutputAxisContract>? OutputAxesForPorts(IReadOnlyList<int> inputRanks);
}

/// <summary>
/// Opt-in for a contract describing a layer that returns MORE THAN ONE tensor.
/// </summary>
/// <remarks>
/// Reach it through <see cref="ShapeContractExtensions.OutputsFor"/>, which falls back to wrapping the
/// single-output answer, so every existing contract keeps answering exactly as before.
/// </remarks>
public interface IMultiOutputShapeContract : IShapeContract
{
    /// <summary>
    /// One axis list per OUTPUT tensor, for a layer that returns more than one.
    /// </summary>
    /// <param name="inputRanks">Rank of each incoming tensor, in port order.</param>
    /// <returns>One entry per output tensor, or <c>null</c> if this combination is not accepted.</returns>
    /// <remarks>
    /// <para>
    /// Defaults to wrapping the single-output form, so every existing contract answers exactly as
    /// before and nothing needs changing to adopt this.
    /// </para>
    /// <para>
    /// WHY THIS EXISTS. Both forms above describe ONE output tensor, which is all almost every layer
    /// produces. Autoformer's series decomposition does not: its encoder consumes and returns a
    /// <c>(seasonal, trend)</c> PAIR. Those layers papered over it by declaring an output width of
    /// <c>embeddingDim * 2</c> - a packed placeholder that NO code produces, while the real forward
    /// lives in the model and their own <c>ForwardTraced</c> throws. Declaring that number as a shape
    /// would have been describing a tensor that does not exist, which is worse than declaring nothing.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> most layers take tensors in and give one tensor back. A few give back
    /// several - a decomposition that separates a signal into a trend and a seasonal part hands you
    /// both. This lets such a layer describe the shape of each one, instead of pretending they are a
    /// single wider tensor.
    /// </para>
    /// </remarks>
    IReadOnlyList<IReadOnlyList<OutputAxisContract>>? OutputsFor(IReadOnlyList<int> inputRanks);
}

/// <summary>
/// Uniform access to the optional contract forms, whichever of them a contract actually implements.
/// </summary>
/// <remarks>
/// <para>
/// These were default interface methods until this assembly's net471 target rejected them (CS8701 -
/// the .NET Framework runtime has no such concept). Extension methods give the same "every contract
/// answers, only the ones that care override" behaviour on EVERY target framework, and they keep the
/// fallbacks in exactly one place rather than duplicating them at each call site.
/// </para>
/// </remarks>
public static class ShapeContractExtensions
{
    /// <summary>Batch-aware form, falling back to the rank-only form.</summary>
    public static IReadOnlyList<OutputAxisContract>? OutputAxesFor(
        this IShapeContract contract, int inputRank, bool isBatched)
        => contract is IBatchAwareShapeContract aware
            ? aware.OutputAxesFor(inputRank, isBatched)
            : contract.OutputAxesFor(inputRank);

    /// <summary>Multi-port form, falling back to the single-input form when exactly one port is fed.</summary>
    public static IReadOnlyList<OutputAxisContract>? OutputAxesForPorts(
        this IShapeContract contract, IReadOnlyList<int> inputRanks)
    {
        if (contract is IMultiPortShapeContract multi) return multi.OutputAxesForPorts(inputRanks);
        return inputRanks is { Count: 1 } ? contract.OutputAxesFor(inputRanks[0]) : null;
    }

    /// <summary>Multi-output form, falling back to wrapping the single-output answer.</summary>
    public static IReadOnlyList<IReadOnlyList<OutputAxisContract>>? OutputsFor(
        this IShapeContract contract, IReadOnlyList<int> inputRanks)
    {
        if (contract is IMultiOutputShapeContract multi) return multi.OutputsFor(inputRanks);

        var single = contract.OutputAxesForPorts(inputRanks);
        return single is null ? null : new[] { single };
    }
}

/// <summary>One output axis, and the relation that sizes it.</summary>
/// <param name="Axis">The axis's role.</param>
/// <param name="Relation">How it is sized from the input.</param>
public readonly record struct OutputAxisContract(TensorAxis Axis, AxisRelation Relation);
