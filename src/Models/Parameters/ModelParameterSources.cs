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
        int expected = _isPresent is null || _isPresent() ? 1 : 0;
        if (parameters.Length != expected)
            throw new ArgumentException(
                $"Expected exactly {expected} values for the scalar parameter, got {parameters.Length}.",
                nameof(parameters));
        if (expected == 0) return;
        _set(parameters[0]);
    }
}

/// <summary>
/// A component whose width is decided BY the vector it is restored from, not checked against it.
/// </summary>
/// <remarks>
/// For models that genuinely do not know their own size until they are fitted -- a propensity
/// model has no coefficients yet, a Kaplan-Meier curve has as many points as the data had event
/// times. A fresh instance of one declares zero parameters, and restoring a checkpoint INTO a
/// fresh instance is the whole point of a checkpoint, so a strict length check made every such
/// load throw. The registry gives a component marked this way whatever the fixed-size components
/// leave; at most one may be registered, and it must be last.
/// </remarks>
public interface IVariableLengthParameterSource<T> : IParameterSource<T>
{
    /// <summary>
    /// Gets whether this source may learn a different width from the next restore payload.
    /// </summary>
    /// <remarks>
    /// Some sources are genuinely variable for their whole lifetime. A replaceable vector field is
    /// different: it needs one deferred restore while empty, then its materialized width becomes an
    /// exact contract. Keeping that distinction here prevents a later malformed checkpoint from
    /// silently resizing an already initialized model.
    /// </remarks>
    bool CanResizeOnRestore { get; }
}

/// <summary>
/// A whole <see cref="Vector{T}"/> field exposed as a parameter surface.
/// </summary>
/// <remarks>
/// Takes a setter as well as a getter because the models that hold parameters this way REPLACE the
/// vector on restore rather than writing into it. A source that only read the field would restore
/// into a vector the model no longer references.
/// </remarks>
public sealed class VectorFieldParameterSource<T> : IVariableLengthParameterSource<T>, IParameterLayoutSource
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
    public bool CanResizeOnRestore => ParameterCount == 0;

    /// <inheritdoc />
    public IReadOnlyList<ParameterSlotDescriptor> GetParameterLayout()
    {
        var value = _get();
        return new[]
        {
            new ParameterSlotDescriptor(
                "$", ParameterSlotRole.Trainable,
                value is null ? ParameterReadiness.ShapeDeferred
                    : value.Length == 0 ? ParameterReadiness.ShapeResolvedUnmaterialized
                    : ParameterReadiness.Materialized,
                value is null ? null : (long?)value.Length,
                shape: value is null ? null : new[] { value.Length },
                elementType: typeof(T).FullName)
        };
    }

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

/// <summary>
/// Like <see cref="DelegatingParameterSource{T}"/>, but sized BY the vector it is restored from.
/// </summary>
/// <remarks>
/// For a model that packs several pieces into one vector whose width it does not know until it is
/// fitted -- an online learner writes its weights and then a bias, and learns how many weights
/// there are from the first vector it sees. Splitting that into two components would put the
/// variable-length piece FIRST, which the registry cannot slice; owning the whole packing in one
/// component keeps the layout exactly as it was and still leaves one place that decides it.
/// </remarks>
public sealed class VariableLengthParameterSource<T> : IVariableLengthParameterSource<T>
{
    private readonly Func<long> _count;
    private readonly Func<Vector<T>> _get;
    private readonly Action<Vector<T>> _set;

    /// <summary>Creates a source from count, read and write delegates.</summary>
    public VariableLengthParameterSource(Func<long> count, Func<Vector<T>> get, Action<Vector<T>> set)
    {
        _count = count ?? throw new ArgumentNullException(nameof(count));
        _get = get ?? throw new ArgumentNullException(nameof(get));
        _set = set ?? throw new ArgumentNullException(nameof(set));
    }

    /// <inheritdoc />
    public long ParameterCount => _count();

    /// <inheritdoc />
    public bool CanResizeOnRestore => true;

    /// <inheritdoc />
    public Vector<T> GetParameters() => _get();

    /// <inheritdoc />
    public void SetParameters(Vector<T> parameters)
    {
        if (parameters is null) throw new ArgumentNullException(nameof(parameters));
        _set(parameters);
    }
}

/// <summary>
/// One or more tensor lists exposed as a single parameter surface, concatenated in the order the
/// lists are given and, within each list, in index order.
/// </summary>
/// <remarks>
/// For models that hold weights as bare <c>List&lt;Tensor&lt;T&gt;&gt;</c> rather than layers -- a
/// feature-pyramid neck keeps a lateral weight and bias per level, and an output pair per level.
/// The tensors are written THROUGH, never replaced, so a restore reaches the same instances the
/// forward pass reads.
/// </remarks>
public sealed class TensorListParameterSource<T> : IParameterSource<T>
{
    private readonly Func<IReadOnlyList<Tensor<T>>>[] _lists;

    /// <summary>Creates a source over the given tensor lists, in order.</summary>
    public TensorListParameterSource(params Func<IReadOnlyList<Tensor<T>>>[] lists)
    {
        _lists = lists ?? throw new ArgumentNullException(nameof(lists));
    }

    private IEnumerable<Tensor<T>> Tensors()
    {
        foreach (var list in _lists)
        {
            var items = list();
            if (items is null) continue;
            foreach (var t in items)
            {
                if (t is not null) yield return t;
            }
        }
    }

    /// <inheritdoc />
    public long ParameterCount
    {
        get
        {
            long total = 0;
            foreach (var t in Tensors()) total += t.Length;
            return total;
        }
    }

    /// <inheritdoc />
    public Vector<T> GetParameters()
    {
        var result = new Vector<T>(checked((int)ParameterCount));
        int idx = 0;
        foreach (var t in Tensors())
        {
            for (int i = 0; i < t.Length; i++) result[idx++] = t[i];
        }
        return result;
    }

    /// <inheritdoc />
    public void SetParameters(Vector<T> parameters)
    {
        if (parameters is null) throw new ArgumentNullException(nameof(parameters));
        int expected = checked((int)ParameterCount);
        if (parameters.Length != expected)
            throw new ArgumentException(
                $"Expected exactly {expected} values for the tensor lists, got {parameters.Length}.",
                nameof(parameters));
        int idx = 0;
        foreach (var t in Tensors())
        {
            for (int i = 0; i < t.Length; i++) t[i] = parameters[idx++];
        }
    }
}
