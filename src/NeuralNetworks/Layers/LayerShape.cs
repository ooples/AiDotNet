namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// A layer's shape, in which individual axes may be dynamic and may be named.
/// </summary>
/// <remarks>
/// <para>
/// A plain <c>int[]</c> cannot say "this axis is fixed and that one is not", so every layer whose
/// output length depends on its runtime input had to either invent a value or refuse to resolve.
/// Inventing is what went wrong repeatedly: <c>PositionalEncodingLayer</c> declared its encoding
/// table size, <c>CrossAttentionLayer</c> declared a defaulted 64, and
/// <c>TransformerDecoderLayer</c> declared whatever its first input happened to be and then went
/// stale. Each fabricated number was then read as fact by parameter-vector slicing, chain
/// resolution and ONNX export.
/// </para>
/// <para>
/// <b>Where this sits relative to other frameworks.</b> Keras represents an unknown axis as
/// <c>None</c> and ONNX gives it a name; PyTorch has no static shape contract at all. Two things
/// here go past that:
/// </para>
/// <list type="number">
/// <item>
/// Axes can be NAMED, so a relationship is expressible rather than merely a hole. Two axes
/// carrying the same name assert they will be equal at runtime, which is checkable; Keras's
/// anonymous <c>None</c> is not.
/// </item>
/// <item>
/// A dynamic axis cannot be consumed by accident. Reading concrete dimensions requires
/// <see cref="TryGetConcrete"/> or <see cref="RequireConcrete"/>, so a caller that needs real
/// numbers either handles the dynamic case or fails with a message naming the axis and the
/// reason. Keras and PyTorch both hand back plain containers in which a <c>None</c> can be
/// silently misused.
/// </item>
/// </list>
/// <para><b>For Beginners:</b> Some parts of a tensor's shape are known in advance (how many
/// features a layer outputs) and some are not (how many words are in this particular sentence).
/// This type keeps track of which is which, gives the unknown ones names so two of them can be
/// declared equal, and refuses to hand out a number for an axis nobody knows yet.</para>
/// </remarks>
public readonly struct LayerShape : IEquatable<LayerShape>
{
    /// <summary>Marks an axis whose size is not known until a real input arrives.</summary>
    public const int Dynamic = -1;

    private readonly int[]? _axes;
    private readonly string?[]? _names;

    /// <summary>Creates a shape from axis sizes, using <see cref="Dynamic"/> for unknown axes.</summary>
    /// <param name="axes">Axis sizes; any non-positive entry is treated as dynamic.</param>
    public LayerShape(params int[] axes)
        : this(axes, null)
    {
    }

    /// <summary>Creates a shape whose dynamic axes carry names.</summary>
    /// <param name="axes">Axis sizes; any non-positive entry is treated as dynamic.</param>
    /// <param name="names">
    /// Optional per-axis names. Two axes sharing a name assert they are equal at runtime, which
    /// <c>VerifyReportedOutputShape</c> checks against the real forward.
    /// </param>
    public LayerShape(int[]? axes, string?[]? names)
    {
        _axes = axes is null ? null : NormalizeAxes(axes);
        _names = names;
    }

    private static int[] NormalizeAxes(int[] axes)
    {
        var copy = new int[axes.Length];
        for (int i = 0; i < axes.Length; i++) copy[i] = axes[i] > 0 ? axes[i] : Dynamic;
        return copy;
    }

    /// <summary>Number of axes.</summary>
    public int Rank => _axes?.Length ?? 0;

    /// <summary>The size of an axis, or <see cref="Dynamic"/> when it is not yet known.</summary>
    public int this[int axis] => _axes is null ? Dynamic : _axes[axis];

    /// <summary>The name of an axis, or <c>null</c> when it is unnamed.</summary>
    public string? NameOf(int axis)
        => _names is not null && axis < _names.Length ? _names[axis] : null;

    /// <summary><c>true</c> when every axis has a known size.</summary>
    public bool IsFullyConcrete
    {
        get
        {
            if (_axes is null || _axes.Length == 0) return false;
            foreach (var a in _axes) if (a <= 0) return false;
            return true;
        }
    }

    /// <summary><c>true</c> when at least one axis is dynamic.</summary>
    public bool HasDynamicAxis
    {
        get
        {
            if (_axes is null) return true;
            foreach (var a in _axes) if (a <= 0) return true;
            return false;
        }
    }

    /// <summary>Gets the concrete axis sizes, or returns <c>false</c> if any axis is dynamic.</summary>
    /// <param name="concrete">The axis sizes when every axis is known; otherwise <c>null</c>.</param>
    /// <returns><c>true</c> when the shape is fully concrete.</returns>
    public bool TryGetConcrete(out int[]? concrete)
    {
        if (!IsFullyConcrete)
        {
            concrete = null;
            return false;
        }

        concrete = (int[])_axes!.Clone();
        return true;
    }

    /// <summary>Gets the concrete axis sizes, throwing when any axis is dynamic.</summary>
    /// <param name="reason">
    /// What the caller needs concrete dimensions FOR. Included in the exception so the failure
    /// explains itself rather than surfacing as a wrong number somewhere downstream.
    /// </param>
    /// <returns>The axis sizes.</returns>
    /// <exception cref="InvalidOperationException">Thrown when any axis is dynamic.</exception>
    public int[] RequireConcrete(string reason)
    {
        if (TryGetConcrete(out var concrete)) return concrete!;

        var described = new string[Rank];
        for (int i = 0; i < Rank; i++)
        {
            described[i] = this[i] > 0
                ? this[i].ToString(System.Globalization.CultureInfo.InvariantCulture)
                : NameOf(i) is { } n ? $"?{n}" : "?";
        }

        throw new InvalidOperationException(
            $"{reason} requires a fully concrete shape, but this layer's shape is " +
            $"[{string.Join(", ", described)}] — the axes shown as ? depend on the input and are " +
            "not known until a real forward. Resolve the layer from a real input shape first, or " +
            "handle the dynamic case explicitly via TryGetConcrete.");
    }

    /// <summary>The axis sizes as a plain array, with <see cref="Dynamic"/> for unknown axes.</summary>
    /// <remarks>
    /// For callers that genuinely tolerate a dynamic axis. Prefer <see cref="TryGetConcrete"/> or
    /// <see cref="RequireConcrete"/>, which make the decision explicit.
    /// </remarks>
    public int[] ToArray() => _axes is null ? [] : (int[])_axes.Clone();

    /// <inheritdoc/>
    public bool Equals(LayerShape other)
    {
        if (Rank != other.Rank) return false;
        for (int i = 0; i < Rank; i++) if (this[i] != other[i]) return false;
        return true;
    }

    /// <inheritdoc/>
    public override bool Equals(object? obj) => obj is LayerShape s && Equals(s);

    /// <inheritdoc/>
    public override int GetHashCode()
    {
        int hash = 17;
        for (int i = 0; i < Rank; i++) hash = (hash * 31) + this[i];
        return hash;
    }

    /// <inheritdoc/>
    public override string ToString()
    {
        if (Rank == 0) return "[]";
        var parts = new string[Rank];
        for (int i = 0; i < Rank; i++)
        {
            parts[i] = this[i] > 0
                ? this[i].ToString(System.Globalization.CultureInfo.InvariantCulture)
                : NameOf(i) is { } n ? $"?{n}" : "?";
        }
        return "[" + string.Join(", ", parts) + "]";
    }
}
