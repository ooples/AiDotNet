using System;
using AiDotNet.Interfaces;
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
