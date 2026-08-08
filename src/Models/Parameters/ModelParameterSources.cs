using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Models.Parameters;

/// <summary>
/// A single scalar exposed as a one-element parameter surface, optionally gated so it contributes
/// nothing until the model is fitted.
/// </summary>
/// <remarks>
/// <para>
/// Reads and writes go through delegates rather than a captured value, because the scalar lives in
/// a field the model keeps mutating; capturing it once would freeze the surface at whatever the
/// value was when the component was registered.
/// </para>
/// <para>
/// The gate exists for detectors whose threshold is only meaningful after fitting. It applies to the
/// COUNT as well as the vector, so the two never disagree: an unfitted model reports zero from both.
/// </para>
/// </remarks>
public sealed class ScalarParameterSource<T> : IParameterSource<T>
{
    private readonly Func<T> _get;
    private readonly Action<T> _set;
    private readonly Func<bool>? _isPresent;

    /// <summary>Creates a source over one scalar.</summary>
    /// <param name="get">Reads the current value.</param>
    /// <param name="set">Writes a restored value.</param>
    /// <param name="isPresent">Optional gate; when it returns false the scalar contributes nothing.</param>
    public ScalarParameterSource(Func<T> get, Action<T> set, Func<bool>? isPresent = null)
    {
        _get = get ?? throw new ArgumentNullException(nameof(get));
        _set = set ?? throw new ArgumentNullException(nameof(set));
        _isPresent = isPresent;
    }

    /// <inheritdoc />
    public long ParameterCount => _isPresent is null || _isPresent() ? 1 : 0;

    /// <inheritdoc />
    public Vector<T> GetParameters()
    {
        if (ParameterCount == 0) return new Vector<T>(0);
        var result = new Vector<T>(1);
        result[0] = _get();
        return result;
    }

    /// <inheritdoc />
    public void SetParameters(Vector<T> parameters)
    {
        if (parameters is null) throw new ArgumentNullException(nameof(parameters));
        if (parameters.Length == 0) return;
        _set(parameters[0]);
    }
}

/// <summary>
/// A whole <see cref="Vector{T}"/> field exposed as a parameter surface.
/// </summary>
/// <remarks>
/// Takes a setter as well as a getter because the models that hold parameters this way REPLACE the
/// vector on restore rather than writing into it. A source that only read the field would restore
/// into a vector the model no longer references.
/// </remarks>
public sealed class VectorFieldParameterSource<T> : IParameterSource<T>
{
    private readonly Func<Vector<T>?> _get;
    private readonly Action<Vector<T>> _set;

    /// <summary>Creates a source over a vector-valued field.</summary>
    public VectorFieldParameterSource(Func<Vector<T>?> get, Action<Vector<T>> set)
    {
        _get = get ?? throw new ArgumentNullException(nameof(get));
        _set = set ?? throw new ArgumentNullException(nameof(set));
    }

    /// <inheritdoc />
    public long ParameterCount => _get()?.Length ?? 0;

    /// <inheritdoc />
    public Vector<T> GetParameters()
    {
        var current = _get();
        if (current is null) return new Vector<T>(0);
        var result = new Vector<T>(current.Length);
        for (int i = 0; i < current.Length; i++) result[i] = current[i];
        return result;
    }

    /// <inheritdoc />
    public void SetParameters(Vector<T> parameters)
    {
        if (parameters is null) throw new ArgumentNullException(nameof(parameters));
        _set(parameters);
    }
}

/// <summary>
/// Adapts a component that already exposes count / read / write but does not implement
/// <see cref="IParameterSource{T}"/>.
/// </summary>
/// <remarks>
/// For types the registry cannot hold directly because they sit outside the interface hierarchy --
/// a projector head, a helper network, an internal block. Preferable to widening those types'
/// interfaces when they are not ours to change; where they ARE ours, implementing
/// <see cref="IParameterSource{T}"/> on them directly is cleaner and this adapter is unnecessary.
/// </remarks>
public sealed class DelegatingParameterSource<T> : IParameterSource<T>
{
    private readonly Func<long> _count;
    private readonly Func<Vector<T>> _get;
    private readonly Action<Vector<T>> _set;

    /// <summary>Creates a source from count, read and write delegates.</summary>
    public DelegatingParameterSource(Func<long> count, Func<Vector<T>> get, Action<Vector<T>> set)
    {
        _count = count ?? throw new ArgumentNullException(nameof(count));
        _get = get ?? throw new ArgumentNullException(nameof(get));
        _set = set ?? throw new ArgumentNullException(nameof(set));
    }

    /// <inheritdoc />
    public long ParameterCount => _count();

    /// <inheritdoc />
    public Vector<T> GetParameters() => _get();

    /// <inheritdoc />
    public void SetParameters(Vector<T> parameters)
    {
        if (parameters is null) throw new ArgumentNullException(nameof(parameters));
        _set(parameters);
    }
}
