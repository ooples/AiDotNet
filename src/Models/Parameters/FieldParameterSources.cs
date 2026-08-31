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
                value is null ? null : (long?)value.Length,
                shape: value is null ? null : value.Shape.ToArray(),
                elementType: typeof(T).FullName)
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
            throw new ParameterLayoutNotReadyException(
                "restore", new ParameterLayoutSnapshot(GetParameterLayout()));
        if (parameters.Length != t.Length)
            throw new ArgumentException(
                $"Expected exactly {t.Length} values for the tensor field, got {parameters.Length}.",
                nameof(parameters));
        parameters.AsSpan().CopyTo(t.AsWritableSpan());
    }
}

/// <summary>
/// A tensor field whose one unresolved axis is learned from the first restore payload.
/// </summary>
/// <remarks>
/// Fit-sized models can declare a placeholder such as <c>[0]</c> or <c>[5, 0]</c>. The fixed
/// dimensions preserve the tensor's structure while the single zero dimension identifies the
/// axis whose width is data-dependent. Once restored, the source becomes fixed-size and every
/// later restore is validated exactly like <see cref="TensorFieldParameterSource{T}"/>.
/// </remarks>
public sealed class ResizableTensorFieldParameterSource<T> :
    IVariableLengthParameterSource<T>, IParameterLayoutSource
{
    private readonly Func<Tensor<T>?> _get;
    private readonly Action<Tensor<T>> _set;

    /// <summary>Creates a source over a replaceable, fit-sized tensor field.</summary>
    public ResizableTensorFieldParameterSource(Func<Tensor<T>?> get, Action<Tensor<T>> set)
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
        bool unresolved = value is null;
        if (value is not null)
        {
            for (int axis = 0; axis < value.Shape.Length; axis++)
                unresolved |= value.Shape[axis] <= 0;
        }
        return new[]
        {
            new ParameterSlotDescriptor(
                "$", ParameterSlotRole.Trainable,
                unresolved ? ParameterReadiness.ShapeDeferred
                    : value!.Length == 0 ? ParameterReadiness.ShapeResolvedUnmaterialized
                    : ParameterReadiness.Materialized,
                unresolved ? null : (long?)value!.Length,
                shape: value?.Shape.ToArray(),
                elementType: typeof(T).FullName)
        };
    }

    /// <inheritdoc />
    public Vector<T> GetParameters()
    {
        var value = _get();
        if (value is null) return new Vector<T>(0);
        var result = new Vector<T>(value.Length);
        value.AsSpan().CopyTo(result.AsWritableSpan());
        return result;
    }

    /// <inheritdoc />
    public void SetParameters(Vector<T> parameters)
    {
        if (parameters is null) throw new ArgumentNullException(nameof(parameters));
        var current = _get();
        if (current is not null && current.Length > 0)
        {
            if (parameters.Length != current.Length)
                throw new ArgumentException(
                    $"Expected exactly {current.Length} values for the tensor field, got {parameters.Length}.",
                    nameof(parameters));
            parameters.AsSpan().CopyTo(current.AsWritableSpan());
            return;
        }

        var declaredShape = current?.Shape.ToArray() ?? new[] { 0 };
        int unresolvedAxis = -1;
        long fixedProduct = 1;
        for (int axis = 0; axis < declaredShape.Length; axis++)
        {
            if (declaredShape[axis] <= 0)
            {
                if (unresolvedAxis >= 0)
                    throw new ParameterLayoutNotReadyException(
                        "restore", new ParameterLayoutSnapshot(GetParameterLayout()));
                unresolvedAxis = axis;
            }
            else
            {
                fixedProduct = checked(fixedProduct * declaredShape[axis]);
            }
        }

        if (unresolvedAxis < 0)
        {
            if (parameters.Length != 0)
                throw new ArgumentException(
                    $"Expected an empty tensor payload, got {parameters.Length} values.", nameof(parameters));
        }
        else
        {
            if (fixedProduct == 0 || parameters.Length % fixedProduct != 0)
                throw new ArgumentException(
                    $"A {parameters.Length}-value payload cannot resolve tensor shape " +
                    $"[{string.Join(", ", declaredShape)}].", nameof(parameters));
            declaredShape[unresolvedAxis] = checked((int)(parameters.Length / fixedProduct));
        }

        var replacement = new Tensor<T>(declaredShape);
        parameters.AsSpan().CopyTo(replacement.AsWritableSpan());
        _set(replacement);
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
                count,
                shape: value is null ? null : new[] { value.Rows, value.Columns },
                elementType: typeof(T).FullName)
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
            throw new ParameterLayoutNotReadyException(
                "restore", new ParameterLayoutSnapshot(GetParameterLayout()));
        int expected = checked(m.Rows * m.Columns);
        if (parameters.Length != expected)
            throw new ArgumentException(
                $"Expected exactly {expected} values for the matrix field, got {parameters.Length}.",
                nameof(parameters));
        int idx = 0;
        for (int r = 0; r < m.Rows; r++)
        {
            for (int c = 0; c < m.Columns; c++) m[r, c] = parameters[idx++];
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
                value is null ? null : (long?)value.Length,
                shape: value is null ? null : new[] { value.Length },
                elementType: typeof(T).FullName)
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
            throw new ParameterLayoutNotReadyException(
                "restore", new ParameterLayoutSnapshot(GetParameterLayout()));
        if (parameters.Length != v.Length)
            throw new ArgumentException(
                $"Expected exactly {v.Length} values for the vector field, got {parameters.Length}.",
                nameof(parameters));
        parameters.AsSpan().CopyTo(v.AsWritableSpan());
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
public sealed class ComponentCollectionParameterSource<T> : IParameterSource<T>, IParameterLayoutSource,
    IParameterSurfaceLifecycle
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

    /// <summary>The currently present collection members, evaluated lazily.</summary>
    internal IEnumerable<IParameterSource<T>> Current => Members();

    /// <summary>Whether this live collection currently owns a particular component instance.</summary>
    internal bool ContainsCurrent(IParameterSource<T>? candidate)
    {
        if (candidate is null) return false;
        foreach (var member in Members())
        {
            if (ReferenceEquals(member, candidate)) return true;
        }
        return false;
    }

    /// <inheritdoc />
    public void PrepareParameterSurface(ParameterSurfaceIntent intent)
    {
        foreach (var member in Members())
        {
            if (member is IParameterSurfaceLifecycle lifecycle)
                lifecycle.PrepareParameterSurface(intent);
            else if (intent != ParameterSurfaceIntent.Describe
                && member is IParameterMaterializationSource materializer)
                materializer.MaterializeParameters();
        }
    }

    /// <inheritdoc />
    public long ParameterCount
    {
        get
        {
            long total = 0;
            foreach (var member in Members())
            {
                var layout = GetMemberLayout(member);
                if (!TryGetResolvedCount(layout, out long count))
                    count = member.ParameterCount;
                total = checked(total + count);
            }
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
            string prefix = $"index={index++:D8}";
            var memberSlots = GetMemberLayout(member);
            if (memberSlots.Count == 0)
            {
                slots.Add(new ParameterSlotDescriptor(
                    prefix, ParameterSlotRole.Trainable, ParameterReadiness.ParameterFree, 0));
                continue;
            }

            for (int slotIndex = 0; slotIndex < memberSlots.Count; slotIndex++)
            {
                var memberSlot = memberSlots[slotIndex];
                string stableId = memberSlot.StableId == "$"
                    ? prefix
                    : prefix + "/" + memberSlot.StableId;
                slots.Add(new ParameterSlotDescriptor(
                    stableId,
                    memberSlot.Role,
                    memberSlot.Readiness,
                    memberSlot.ParameterCount,
                    shape: memberSlot.Shape,
                    elementType: memberSlot.ElementType,
                    updatePolicy: memberSlot.UpdatePolicy,
                    persistence: memberSlot.Persistence,
                    ownership: memberSlot.Ownership,
                    availability: memberSlot.Availability,
                    materializedParameterCount: memberSlot.MaterializedParameterCount));
            }
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
        var members = new List<IParameterSource<T>>(Members());
        var memberCounts = new int[members.Count];
        long expectedLong = 0;
        for (int i = 0; i < members.Count; i++)
        {
            var layout = GetMemberLayout(members[i]);
            if (!TryGetResolvedCount(layout, out long count))
                throw new ParameterLayoutNotReadyException(
                    "restore component collection", new ParameterLayoutSnapshot(GetParameterLayout()));
            memberCounts[i] = checked((int)count);
            expectedLong = checked(expectedLong + count);
        }
        int expected = checked((int)expectedLong);
        if (parameters.Length != expected)
            throw new ArgumentException(
                $"Expected exactly {expected} values for the component collection, got {parameters.Length}.",
                nameof(parameters));

        int at = 0;
        for (int i = 0; i < members.Count; i++)
        {
            int n = memberCounts[i];
            var slice = new Vector<T>(n);
            for (int j = 0; j < n; j++) slice[j] = parameters[at++];
            members[i].SetParameters(slice);
        }
    }

    private static IReadOnlyList<ParameterSlotDescriptor> GetMemberLayout(IParameterSource<T> member)
    {
        if (member is IParameterManifestProvider manifestProvider)
            return manifestProvider.ParameterLayout.Slots;
        if (member is IParameterLayoutSource layoutSource)
            return layoutSource.GetParameterLayout();

        long count = member.ParameterCount;
        return new[]
        {
            new ParameterSlotDescriptor(
                "$", ParameterSlotRole.Trainable,
                count == 0 ? ParameterReadiness.ParameterFree : ParameterReadiness.Materialized,
                count,
                materializedParameterCount: count)
        };
    }

    private static bool TryGetResolvedCount(
        IReadOnlyList<ParameterSlotDescriptor> slots, out long count)
    {
        count = 0;
        for (int i = 0; i < slots.Count; i++)
        {
            if (slots[i].Readiness == ParameterReadiness.ShapeDeferred
                || !slots[i].ParameterCount.HasValue)
            {
                count = 0;
                return false;
            }

            count = checked(count + slots[i].ParameterCount!.Value);
        }
        return true;
    }
}

/// <summary>
/// A lazily obtained component whose identity is registered before the component itself exists.
/// </summary>
public sealed class ComponentAccessorParameterSource<T> : IParameterSource<T>, IParameterLayoutSource,
    IParameterSurfaceLifecycle
{
    private readonly Func<IParameterSource<T>?> _get;
    private readonly bool _optional;

    /// <summary>Creates a source that re-reads the component on every operation.</summary>
    /// <param name="get">Returns the component, or <c>null</c> if it is not available.</param>
    /// <param name="optional">
    /// What a <c>null</c> component MEANS. <c>false</c> (the default) means "not constructed yet",
    /// so the slot's shape is genuinely unknown. <c>true</c> means the owner legitimately does not
    /// have this component -- an unconditional diffusion model with no conditioner, or one whose
    /// conditioner is not a parameter source -- which is a RESOLVED fact and contributes zero
    /// parameters.
    /// </param>
    /// <remarks>
    /// <para>
    /// The distinction is not cosmetic. <c>ParameterManifest</c> marks a layout unresolved when any
    /// slot is <c>ShapeDeferred</c> OR has no parameter count, and an unresolved layout makes
    /// <c>ParameterCount</c> THROW for the entire model rather than return a number. Reporting an
    /// absent optional component as deferred therefore does not degrade one slot; it takes the whole
    /// model's parameter surface offline.
    /// </para>
    /// <para>
    /// Defaults to <c>false</c> deliberately: every existing registration means "not constructed
    /// yet", and treating those as absent would report a confident zero for a component whose real
    /// size is simply not known yet.
    /// </para>
    /// </remarks>
    public ComponentAccessorParameterSource(Func<IParameterSource<T>?> get, bool optional = false)
    {
        _get = get ?? throw new ArgumentNullException(nameof(get));
        _optional = optional;
    }

    /// <summary>The current component, used internally to deduplicate and materialize its storage.</summary>
    internal IParameterSource<T>? Current => _get();

    /// <inheritdoc />
    public void PrepareParameterSurface(ParameterSurfaceIntent intent)
    {
        var component = _get();
        if (component is IParameterSurfaceLifecycle lifecycle)
            lifecycle.PrepareParameterSurface(intent);
        else if (intent != ParameterSurfaceIntent.Describe
            && component is IParameterMaterializationSource materializer)
            materializer.MaterializeParameters();
    }

    /// <inheritdoc />
    public long ParameterCount => _get()?.ParameterCount ?? 0;

    /// <inheritdoc />
    public IReadOnlyList<ParameterSlotDescriptor> GetParameterLayout()
    {
        var component = _get();
        if (component is null)
        {
            // ParameterFree with a count of 0, not ShapeDeferred with no count.
            //
            // The count matters because ParameterManifest computes
            //     unresolved = shapeDeferred || fitDeferred || slots.Any(s => !s.ParameterCount.HasValue)
            // so a null count alone keeps the whole manifest unresolved and makes ParameterCount
            // throw for the entire model.
            //
            // The READINESS matters for a second, subtler reason. ParameterManifest folds slot
            // readiness into one value for the model, and in that fold ConditionalAbsent outranks
            // the fallback: a single absent slot relabels the WHOLE model ConditionalAbsent whenever
            // no other slot happens to be materialized yet, and the read paths reject that with
            // "Cannot read parameters while the layout is ConditionalAbsent". ConditionalAbsent is
            // the right word for a slot that a condition removed from a layout that still describes
            // it; it is the wrong word here, where the owner simply has no such component. It also
            // must not be able to speak for the model.
            //
            // ParameterFree -- "the owner deliberately has no parameters" -- says exactly that, and
            // sets none of the dominating flags in the fold, so an absent optional component stays
            // invisible to the model's aggregate readiness instead of overriding it.
            return new[]
            {
                _optional
                    ? new ParameterSlotDescriptor(
                        "$", ParameterSlotRole.Trainable, ParameterReadiness.ParameterFree, 0L)
                    : new ParameterSlotDescriptor(
                        "$", ParameterSlotRole.Trainable, ParameterReadiness.ShapeDeferred, null)
            };
        }

        if (component is IParameterManifestProvider manifest)
        {
            var layout = manifest.ParameterLayout;
            var slots = new List<ParameterSlotDescriptor>(layout.Slots.Count);
            for (int i = 0; i < layout.Slots.Count; i++)
            {
                var slot = layout.Slots[i];
                slots.Add(new ParameterSlotDescriptor(
                    slot.StableId,
                    slot.Role,
                    slot.Readiness,
                    slot.ParameterCount,
                    shape: slot.Shape,
                    elementType: slot.ElementType,
                    updatePolicy: slot.UpdatePolicy,
                    persistence: slot.Persistence,
                    ownership: slot.Ownership,
                    availability: slot.Availability));
            }
            return slots;
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
