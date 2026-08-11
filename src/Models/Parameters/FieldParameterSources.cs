using System;
using System.Collections.Generic;
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
public sealed class TensorFieldParameterSource<T> : IParameterSource<T>, IParameterLayoutSource
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
                value is null ? null : (long?)value.Length)
        };
    }

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
        if (t is null)
        {
            // NOT A SILENT RETURN. This source reads its tensor through an accessor and has no
            // setter, so with the field still null there is nowhere to put these values and no way
            // to create it. Dropping them here is how a restored model came back with its declared
            // shape and none of its learned weights -- the round-trip reported success and the
            // model predicted differently. An empty vector is genuinely nothing to do; anything
            // else is a caller restoring before the owner has materialized its storage.
            if (parameters.Length == 0) return;
            throw new ParameterLayoutNotReadyException(
                "restore", new ParameterLayoutSnapshot(GetParameterLayout()));
        }
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
public sealed class MatrixFieldParameterSource<T> : IParameterSource<T>, IParameterLayoutSource
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
    public IReadOnlyList<ParameterSlotDescriptor> GetParameterLayout()
    {
        var value = _get();
        long? count = value is null ? null : (long)value.Rows * value.Columns;
        return new[]
        {
            new ParameterSlotDescriptor(
                "$", ParameterSlotRole.Trainable,
                value is null ? ParameterReadiness.ShapeDeferred
                    : count == 0 ? ParameterReadiness.ShapeResolvedUnmaterialized
                    : ParameterReadiness.Materialized,
                count)
        };
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
        if (m is null)
        {
            // NOT A SILENT RETURN. This source reads its matrix through an accessor and has no
            // setter, so with the field still null there is nowhere to put these values and no way
            // to create it. Dropping them here is how a restored model came back with its declared
            // shape and none of its learned weights -- the round-trip reported success and the
            // model predicted differently. An empty vector is genuinely nothing to do; anything
            // else is a caller restoring before the owner has materialized its storage.
            if (parameters.Length == 0) return;
            throw new ParameterLayoutNotReadyException(
                "restore", new ParameterLayoutSnapshot(GetParameterLayout()));
        }
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
public sealed class VectorFieldWriteThroughSource<T> : IParameterSource<T>, IParameterLayoutSource
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
                value is null ? null : (long?)value.Length)
        };
    }

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
        if (v is null)
        {
            // NOT A SILENT RETURN. This source reads its vector through an accessor and has no
            // setter, so with the field still null there is nowhere to put these values and no way
            // to create it. Dropping them here is how a restored model came back with its declared
            // shape and none of its learned weights -- the round-trip reported success and the
            // model predicted differently. An empty vector is genuinely nothing to do; anything
            // else is a caller restoring before the owner has materialized its storage.
            if (parameters.Length == 0) return;
            throw new ParameterLayoutNotReadyException(
                "restore", new ParameterLayoutSnapshot(GetParameterLayout()));
        }
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
public sealed class ComponentCollectionParameterSource<T> : IParameterSource<T>, IParameterLayoutSource
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
    public IReadOnlyList<ParameterSlotDescriptor> GetParameterLayout()
    {
        var slots = new List<ParameterSlotDescriptor>();
        int index = 0;
        foreach (var member in Members())
        {
            long count = member.ParameterCount;
            slots.Add(new ParameterSlotDescriptor(
                $"index={index++:D8}", ParameterSlotRole.Trainable,
                count == 0 ? ParameterReadiness.ParameterFree : ParameterReadiness.Materialized,
                count));
        }
        if (slots.Count == 0)
        {
            slots.Add(new ParameterSlotDescriptor(
                "$", ParameterSlotRole.Trainable, ParameterReadiness.ParameterFree, 0));
        }
        return slots;
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

/// <summary>
/// A lazily obtained component whose identity is registered before the component itself exists.
/// </summary>
public sealed class ComponentAccessorParameterSource<T> : IParameterSource<T>, IParameterLayoutSource
{
    private readonly Func<IParameterSource<T>?> _get;

    /// <summary>Creates a source that re-reads the component on every operation.</summary>
    public ComponentAccessorParameterSource(Func<IParameterSource<T>?> get)
    {
        _get = get ?? throw new ArgumentNullException(nameof(get));
    }

    /// <inheritdoc />
    public long ParameterCount => _get()?.ParameterCount ?? 0;

    /// <inheritdoc />
    public IReadOnlyList<ParameterSlotDescriptor> GetParameterLayout()
    {
        var component = _get();
        if (component is null)
        {
            return new[]
            {
                new ParameterSlotDescriptor(
                    "$", ParameterSlotRole.Trainable, ParameterReadiness.ShapeDeferred, null)
            };
        }

        if (component is IParameterManifestProvider manifest)
        {
            var layout = manifest.ParameterLayout;
            return new[]
            {
                new ParameterSlotDescriptor(
                    "$", ParameterSlotRole.Trainable, layout.Readiness, layout.ParameterCount)
            };
        }
        if (component is IParameterLayoutSource source)
            return source.GetParameterLayout();

        long count = component.ParameterCount;
        return new[]
        {
            new ParameterSlotDescriptor(
                "$", ParameterSlotRole.Trainable,
                count == 0 ? ParameterReadiness.ParameterFree : ParameterReadiness.Materialized,
                count)
        };
    }

    /// <inheritdoc />
    public Vector<T> GetParameters()
    {
        var component = _get();
        if (component is null)
            throw new ParameterLayoutNotReadyException("read", new ParameterLayoutSnapshot(GetParameterLayout()));
        return component.GetParameters();
    }

    /// <inheritdoc />
    public void SetParameters(Vector<T> parameters)
    {
        if (parameters is null) throw new ArgumentNullException(nameof(parameters));
        var component = _get();
        if (component is null)
            throw new ParameterLayoutNotReadyException("restore", new ParameterLayoutSnapshot(GetParameterLayout()));
        component.SetParameters(parameters);
    }
}
