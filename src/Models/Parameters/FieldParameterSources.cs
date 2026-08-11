using System;
using System.Collections.Generic;
using AiDotNet.Interfaces;
using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Models.Parameters;

// Parameter sources over a single weight-bearing FIELD, for the code TrainableParameterGenerator
// emits on a model's behalf. These exist because the adapters written for hand-registration do not
// fit generated code -- two properties are required and no existing source has both.
//
// NULL TOLERANCE. 157 of the 339 model weight fields in this library are nullable, because a fitted
// model allocates them in Fit rather than its constructor. A source that dereferenced the field
// would throw from ParameterCount on any unfitted model -- and ParameterCount is exactly what the
// contract tests call first. An absent field reports zero and contributes nothing to the flat
// vector, which is the truthful answer: there are no parameters there yet.
//
// WRITE-THROUGH. These write INTO the instance the field already holds rather than replacing it.
// That matters twice over. A Tensor<T> aliases its storage, so the forward pass and the tape may
// hold the same buffer -- rebinding the field would leave them computing against the old one, the
// stale-weights defect that made every copy-on-write clone of a model containing a DenseLayer
// compute with pre-restore values. It also means a readonly field is registerable, where a
// replacing source would not even compile (CS0191).
//
// The consequence is that a null field cannot be restored INTO: there is no shape to restore
// against, and a flat length cannot imply one for a matrix or a tensor. That is consistent rather
// than lossy -- such a field also contributed nothing on the way out, so the vector holds no values
// destined for it.

/// <summary>A <see cref="Tensor{T}"/> field exposed as a parameter surface, written through.</summary>
/// <typeparam name="T">The numeric type of the values.</typeparam>
public sealed class TensorFieldParameterSource<T> : IParameterSource<T>
{
    private readonly Func<Tensor<T>?> _get;

    /// <summary>Creates a source over whatever tensor <paramref name="accessor"/> currently returns.</summary>
    public TensorFieldParameterSource(Func<Tensor<T>?> accessor)
    {
        _get = accessor ?? throw new ArgumentNullException(nameof(accessor));
    }

    /// <inheritdoc />
    public long ParameterCount => _get()?.Length ?? 0;

    /// <inheritdoc />
    public Vector<T> GetParameters()
    {
        var t = _get();
        if (t is null) return new Vector<T>(0);
        var result = new Vector<T>(t.Length);
        for (int i = 0; i < t.Length; i++) result[i] = t[i];
        return result;
    }

    /// <inheritdoc />
    public void SetParameters(Vector<T> parameters)
    {
        if (parameters is null) throw new ArgumentNullException(nameof(parameters));
        var t = _get();
        if (t is null) return;
        int n = Math.Min(t.Length, parameters.Length);
        for (int i = 0; i < n; i++) t[i] = parameters[i];
    }
}

/// <summary>A <see cref="Matrix{T}"/> field exposed as a parameter surface, written through.</summary>
/// <typeparam name="T">The numeric type of the values.</typeparam>
/// <remarks>
/// Row-major, matching <see cref="AiDotNet.ReinforcementLearning.Parameters.MatrixParameterSource{T}"/>,
/// so a model that moves between the two keeps its serialization layout.
/// </remarks>
public sealed class MatrixFieldParameterSource<T> : IParameterSource<T>
{
    private readonly Func<Matrix<T>?> _get;

    /// <summary>Creates a source over whatever matrix <paramref name="accessor"/> currently returns.</summary>
    public MatrixFieldParameterSource(Func<Matrix<T>?> accessor)
    {
        _get = accessor ?? throw new ArgumentNullException(nameof(accessor));
    }

    /// <inheritdoc />
    public long ParameterCount
    {
        get { var m = _get(); return m is null ? 0 : (long)m.Rows * m.Columns; }
    }

    /// <inheritdoc />
    public Vector<T> GetParameters()
    {
        var m = _get();
        if (m is null) return new Vector<T>(0);
        var result = new Vector<T>(m.Rows * m.Columns);
        int idx = 0;
        for (int r = 0; r < m.Rows; r++)
        {
            for (int c = 0; c < m.Columns; c++) result[idx++] = m[r, c];
        }
        return result;
    }

    /// <inheritdoc />
    public void SetParameters(Vector<T> parameters)
    {
        if (parameters is null) throw new ArgumentNullException(nameof(parameters));
        var m = _get();
        if (m is null) return;
        int idx = 0;
        for (int r = 0; r < m.Rows; r++)
        {
            for (int c = 0; c < m.Columns && idx < parameters.Length; c++) m[r, c] = parameters[idx++];
        }
    }
}

/// <summary>A <see cref="Vector{T}"/> field exposed as a parameter surface, written through.</summary>
/// <typeparam name="T">The numeric type of the values.</typeparam>
/// <remarks>
/// Distinct from <see cref="VectorFieldParameterSource{T}"/>, which REPLACES the vector on restore
/// and therefore needs a setter. This one writes into the existing instance, so it works on a
/// <c>readonly</c> field and cannot leave a caller holding a detached vector.
/// </remarks>
public sealed class VectorFieldWriteThroughSource<T> : IParameterSource<T>
{
    private readonly Func<Vector<T>?> _get;

    /// <summary>Creates a source over whatever vector <paramref name="accessor"/> currently returns.</summary>
    public VectorFieldWriteThroughSource(Func<Vector<T>?> accessor)
    {
        _get = accessor ?? throw new ArgumentNullException(nameof(accessor));
    }

    /// <inheritdoc />
    public long ParameterCount => _get()?.Length ?? 0;

    /// <inheritdoc />
    public Vector<T> GetParameters()
    {
        var v = _get();
        if (v is null) return new Vector<T>(0);
        var result = new Vector<T>(v.Length);
        for (int i = 0; i < v.Length; i++) result[i] = v[i];
        return result;
    }

    /// <inheritdoc />
    public void SetParameters(Vector<T> parameters)
    {
        if (parameters is null) throw new ArgumentNullException(nameof(parameters));
        var v = _get();
        if (v is null) return;
        int n = Math.Min(v.Length, parameters.Length);
        for (int i = 0; i < n; i++) v[i] = parameters[i];
    }
}

/// <summary>
/// A COLLECTION of parameterized components exposed as one parameter surface, concatenated in
/// enumeration order.
/// </summary>
/// <typeparam name="T">The numeric type of the values.</typeparam>
/// <remarks>
/// <para>
/// For a model whose parameters live in a variable set of sub-models rather than in fields: an
/// ensemble, a mixture of experts, a stacked or boosted collection. Every <c>IFullModel</c> is an
/// <see cref="IParameterSource{T}"/> already, so nothing had to be adapted -- what was missing was
/// a way to register the COLLECTION rather than a fixed number of members.
/// </para>
/// <para>
/// The collection is re-read on every call rather than captured. Registration happens once and
/// lazily, so a source that snapshotted the members would freeze whatever the ensemble held at that
/// instant and then silently disagree with itself the moment a member was added or replaced --
/// which for an ensemble is the normal case, not an edge one.
/// </para>
/// </remarks>
public sealed class ComponentCollectionParameterSource<T> : IParameterSource<T>
{
    private readonly Func<IEnumerable<IParameterSource<T>>?> _get;

    /// <summary>Creates a source over whatever components <paramref name="accessor"/> returns.</summary>
    public ComponentCollectionParameterSource(Func<IEnumerable<IParameterSource<T>>?> accessor)
    {
        _get = accessor ?? throw new ArgumentNullException(nameof(accessor));
    }

    private IEnumerable<IParameterSource<T>> Members()
    {
        var items = _get();
        if (items is null) yield break;
        foreach (var m in items)
        {
            if (m is not null) yield return m;
        }
    }

    /// <inheritdoc />
    public long ParameterCount
    {
        get
        {
            long total = 0;
            foreach (var m in Members()) total += m.ParameterCount;
            return total;
        }
    }

    /// <inheritdoc />
    public Vector<T> GetParameters()
    {
        var parts = new List<Vector<T>>();
        int total = 0;
        foreach (var m in Members())
        {
            var p = m.GetParameters();
            parts.Add(p);
            total += p.Length;
        }

        var result = new Vector<T>(total);
        int at = 0;
        for (int i = 0; i < parts.Count; i++)
        {
            for (int j = 0; j < parts[i].Length; j++) result[at++] = parts[i][j];
        }
        return result;
    }

    /// <inheritdoc />
    public void SetParameters(Vector<T> parameters)
    {
        if (parameters is null) throw new ArgumentNullException(nameof(parameters));
        int at = 0;
        foreach (var m in Members())
        {
            int n = (int)m.ParameterCount;
            if (at + n > parameters.Length) break;
            var slice = new Vector<T>(n);
            for (int j = 0; j < n; j++) slice[j] = parameters[at++];
            m.SetParameters(slice);
        }
    }
}

// Sources over RAW double storage in a generic model. Several time-series models keep their
// weights as double[] or double[][] even though the model is generic over T -- the values are
// converted at the boundary. That is a precision decision those models already made; these let
// such storage take part in the parameter surface without first rewriting it, so the surface is
// not held hostage to a separate refactor.

/// <summary>A <c>double[]</c> field exposed as a parameter surface, written through.</summary>
/// <typeparam name="T">The numeric type of the surface.</typeparam>
public sealed class DoubleArrayParameterSource<T> : IParameterSource<T>
{
    private static readonly INumericOperations<T> Ops = MathHelper.GetNumericOperations<T>();
    private readonly Func<double[]?> _get;

    /// <summary>Creates a source over whatever array <paramref name="accessor"/> returns.</summary>
    public DoubleArrayParameterSource(Func<double[]?> accessor)
    {
        _get = accessor ?? throw new ArgumentNullException(nameof(accessor));
    }

    /// <inheritdoc />
    public long ParameterCount => _get()?.Length ?? 0;

    /// <inheritdoc />
    public Vector<T> GetParameters()
    {
        var a = _get();
        if (a is null) return new Vector<T>(0);
        var result = new Vector<T>(a.Length);
        for (int i = 0; i < a.Length; i++) result[i] = Ops.FromDouble(a[i]);
        return result;
    }

    /// <inheritdoc />
    public void SetParameters(Vector<T> parameters)
    {
        if (parameters is null) throw new ArgumentNullException(nameof(parameters));
        var a = _get();
        if (a is null) return;
        int n = Math.Min(a.Length, parameters.Length);
        for (int i = 0; i < n; i++) a[i] = Ops.ToDouble(parameters[i]);
    }
}

/// <summary>A jagged <c>double[][]</c> field exposed as a parameter surface, row-major.</summary>
/// <typeparam name="T">The numeric type of the surface.</typeparam>
public sealed class DoubleJaggedParameterSource<T> : IParameterSource<T>
{
    private static readonly INumericOperations<T> Ops = MathHelper.GetNumericOperations<T>();
    private readonly Func<double[][]?> _get;

    /// <summary>Creates a source over whatever jagged array <paramref name="accessor"/> returns.</summary>
    public DoubleJaggedParameterSource(Func<double[][]?> accessor)
    {
        _get = accessor ?? throw new ArgumentNullException(nameof(accessor));
    }

    /// <inheritdoc />
    public long ParameterCount
    {
        get
        {
            var rows = _get();
            if (rows is null) return 0;
            long total = 0;
            for (int i = 0; i < rows.Length; i++) total += rows[i]?.Length ?? 0;
            return total;
        }
    }

    /// <inheritdoc />
    public Vector<T> GetParameters()
    {
        var rows = _get();
        if (rows is null) return new Vector<T>(0);
        var result = new Vector<T>((int)ParameterCount);
        int at = 0;
        for (int i = 0; i < rows.Length; i++)
        {
            var row = rows[i];
            if (row is null) continue;
            for (int j = 0; j < row.Length; j++) result[at++] = Ops.FromDouble(row[j]);
        }
        return result;
    }

    /// <inheritdoc />
    public void SetParameters(Vector<T> parameters)
    {
        if (parameters is null) throw new ArgumentNullException(nameof(parameters));
        var rows = _get();
        if (rows is null) return;
        int at = 0;
        for (int i = 0; i < rows.Length; i++)
        {
            var row = rows[i];
            if (row is null) continue;
            for (int j = 0; j < row.Length && at < parameters.Length; j++)
                row[j] = Ops.ToDouble(parameters[at++]);
        }
    }
}

/// <summary>A single <c>double</c> field exposed as a one-element parameter surface.</summary>
/// <typeparam name="T">The numeric type of the surface.</typeparam>
public sealed class DoubleScalarParameterSource<T> : IParameterSource<T>
{
    private static readonly INumericOperations<T> Ops = MathHelper.GetNumericOperations<T>();
    private readonly Func<double> _get;
    private readonly Action<double> _set;

    /// <summary>Creates a source over a scalar double field.</summary>
    public DoubleScalarParameterSource(Func<double> get, Action<double> set)
    {
        _get = get ?? throw new ArgumentNullException(nameof(get));
        _set = set ?? throw new ArgumentNullException(nameof(set));
    }

    /// <inheritdoc />
    public long ParameterCount => 1;

    /// <inheritdoc />
    public Vector<T> GetParameters()
    {
        var v = new Vector<T>(1);
        v[0] = Ops.FromDouble(_get());
        return v;
    }

    /// <inheritdoc />
    public void SetParameters(Vector<T> parameters)
    {
        if (parameters is null) throw new ArgumentNullException(nameof(parameters));
        if (parameters.Length > 0) _set(Ops.ToDouble(parameters[0]));
    }
}

// Sequences of weight-bearing values. A model whose parameters live in a LIST or ARRAY of vectors
// or matrices -- a mixture model's per-component means, a boosted model's per-feature shape
// functions, a meta-learner's per-block parameters -- had no way to reach the surface, because a
// single field source describes one value and the layer path describes layers.
//
// Concatenated in enumeration order. For a Dictionary the caller must impose an order (see the
// generated registration, which sorts by key): dictionary enumeration order is an implementation
// detail, and serialization layout cannot rest on one.

/// <summary>A sequence of <see cref="Vector{T}"/> exposed as one parameter surface.</summary>
/// <typeparam name="T">The numeric type of the values.</typeparam>
public sealed class VectorSequenceParameterSource<T> : IParameterSource<T>
{
    private readonly Func<IEnumerable<Vector<T>>?> _get;

    /// <summary>Creates a source over whatever sequence <paramref name="accessor"/> returns.</summary>
    public VectorSequenceParameterSource(Func<IEnumerable<Vector<T>>?> accessor)
    {
        _get = accessor ?? throw new ArgumentNullException(nameof(accessor));
    }

    private IEnumerable<Vector<T>> Items()
    {
        var items = _get();
        if (items is null) yield break;
        foreach (var v in items)
        {
            if (v is not null) yield return v;
        }
    }

    /// <inheritdoc />
    public long ParameterCount
    {
        get { long n = 0; foreach (var v in Items()) n += v.Length; return n; }
    }

    /// <inheritdoc />
    public Vector<T> GetParameters()
    {
        var result = new Vector<T>((int)ParameterCount);
        int at = 0;
        foreach (var v in Items())
        {
            for (int i = 0; i < v.Length; i++) result[at++] = v[i];
        }
        return result;
    }

    /// <inheritdoc />
    public void SetParameters(Vector<T> parameters)
    {
        if (parameters is null) throw new ArgumentNullException(nameof(parameters));
        int at = 0;
        foreach (var v in Items())
        {
            for (int i = 0; i < v.Length && at < parameters.Length; i++) v[i] = parameters[at++];
        }
    }
}

/// <summary>A sequence of <see cref="Matrix{T}"/> exposed as one parameter surface, row-major.</summary>
/// <typeparam name="T">The numeric type of the values.</typeparam>
public sealed class MatrixSequenceParameterSource<T> : IParameterSource<T>
{
    private readonly Func<IEnumerable<Matrix<T>>?> _get;

    /// <summary>Creates a source over whatever sequence <paramref name="accessor"/> returns.</summary>
    public MatrixSequenceParameterSource(Func<IEnumerable<Matrix<T>>?> accessor)
    {
        _get = accessor ?? throw new ArgumentNullException(nameof(accessor));
    }

    private IEnumerable<Matrix<T>> Items()
    {
        var items = _get();
        if (items is null) yield break;
        foreach (var m in items)
        {
            if (m is not null) yield return m;
        }
    }

    /// <inheritdoc />
    public long ParameterCount
    {
        get { long n = 0; foreach (var m in Items()) n += (long)m.Rows * m.Columns; return n; }
    }

    /// <inheritdoc />
    public Vector<T> GetParameters()
    {
        var result = new Vector<T>((int)ParameterCount);
        int at = 0;
        foreach (var m in Items())
        {
            for (int r = 0; r < m.Rows; r++)
            for (int c = 0; c < m.Columns; c++) result[at++] = m[r, c];
        }
        return result;
    }

    /// <inheritdoc />
    public void SetParameters(Vector<T> parameters)
    {
        if (parameters is null) throw new ArgumentNullException(nameof(parameters));
        int at = 0;
        foreach (var m in Items())
        {
            for (int r = 0; r < m.Rows; r++)
            for (int c = 0; c < m.Columns && at < parameters.Length; c++) m[r, c] = parameters[at++];
        }
    }
}

/// <summary>A sequence of <see cref="Tensor{T}"/> exposed as one parameter surface.</summary>
/// <typeparam name="T">The numeric type of the values.</typeparam>
/// <remarks>
/// Written THROUGH, so a tensor's aliased storage stays valid for whatever else holds it -- the
/// same reason the single-field tensor source does not rebind.
/// </remarks>
public sealed class TensorSequenceParameterSource<T> : IParameterSource<T>
{
    private readonly Func<IEnumerable<Tensor<T>>?> _get;

    /// <summary>Creates a source over whatever sequence <paramref name="accessor"/> returns.</summary>
    public TensorSequenceParameterSource(Func<IEnumerable<Tensor<T>>?> accessor)
    {
        _get = accessor ?? throw new ArgumentNullException(nameof(accessor));
    }

    private IEnumerable<Tensor<T>> Items()
    {
        var items = _get();
        if (items is null) yield break;
        foreach (var t in items)
        {
            if (t is not null) yield return t;
        }
    }

    /// <inheritdoc />
    public long ParameterCount
    {
        get { long n = 0; foreach (var t in Items()) n += t.Length; return n; }
    }

    /// <inheritdoc />
    public Vector<T> GetParameters()
    {
        var result = new Vector<T>((int)ParameterCount);
        int at = 0;
        foreach (var t in Items())
        {
            for (int i = 0; i < t.Length; i++) result[at++] = t[i];
        }
        return result;
    }

    /// <inheritdoc />
    public void SetParameters(Vector<T> parameters)
    {
        if (parameters is null) throw new ArgumentNullException(nameof(parameters));
        int at = 0;
        foreach (var t in Items())
        {
            for (int i = 0; i < t.Length && at < parameters.Length; i++) t[i] = parameters[at++];
        }
    }
}
