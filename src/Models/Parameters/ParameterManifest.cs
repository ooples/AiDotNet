using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Models.Parameters;

/// <summary>Describes how a numeric slot participates in model state and optimization.</summary>
public enum ParameterSlotRole
{
    /// <summary>A value an optimizer is allowed to update.</summary>
    Trainable,

    /// <summary>Fitted state that is restored but is not updated by a gradient optimizer.</summary>
    LearnedState,

    /// <summary>A restorable value intentionally excluded from optimization.</summary>
    Frozen,

    /// <summary>A second name for storage owned by another slot.</summary>
    Alias,

    /// <summary>A gradient accumulator rather than model state.</summary>
    Gradient,

    /// <summary>Transient working storage that is neither restored nor optimized.</summary>
    Scratch,

    /// <summary>State owned by an external runtime, such as a loaded ONNX graph.</summary>
    External
}

/// <summary>
/// One concrete, ordered model-state chunk together with the manifest identity and semantic role
/// that govern it.
/// </summary>
/// <remarks>
/// <para>
/// A bare <see cref="Tensor{T}"/> cannot say whether an optimizer may mutate it. That ambiguity
/// made persistent buffers such as batch-normalization running statistics look like trainable
/// weights whenever the chunked checkpoint surface was compared with the flat state surface.
/// Carrying the role beside the tensor lets serialization enumerate the complete state while an
/// optimizer selects only <see cref="ParameterSlotRole.Trainable"/> chunks.
/// </para>
/// <para>
/// <see cref="StableId"/> is durable manifest identity, not a reflection or registration index.
/// <see cref="Tensor"/> is normally live backing storage. Sources whose native storage is scalar,
/// matrix, tree, or another non-tensor representation may return a payload tensor that is committed
/// through their ordinary <c>SetParameters</c> path.
/// </para>
/// </remarks>
public sealed class ParameterChunk<T>
{
    /// <summary>Creates one role-aware state chunk.</summary>
    public ParameterChunk(string stableId, ParameterSlotRole role, Tensor<T> tensor)
    {
        if (string.IsNullOrWhiteSpace(stableId))
            throw new ArgumentException("A parameter chunk requires a stable ID.", nameof(stableId));
        StableId = stableId;
        Role = role;
        Tensor = tensor ?? throw new ArgumentNullException(nameof(tensor));
    }

    /// <summary>The durable path of this chunk in the owning model manifest.</summary>
    public string StableId { get; }

    /// <summary>Whether and how this chunk participates in optimization and persistence.</summary>
    public ParameterSlotRole Role { get; }

    /// <summary>The concrete payload, in the same scalar order as the flat state surface.</summary>
    public Tensor<T> Tensor { get; }
}

/// <summary>
/// Exposes the complete persistent state as role-aware chunks without conflating it with the
/// trainable-only optimizer view.
/// </summary>
public interface IParameterChunkSource<T>
{
    /// <summary>Yields state chunks in the exact order used by count, flat read, and restore.</summary>
    IEnumerable<ParameterChunk<T>> GetParameterStateChunks();
}

/// <summary>States whether a parameter layout can be inspected or restored without allocation.</summary>
public enum ParameterReadiness
{
    /// <summary>The owner deliberately has no parameters.</summary>
    ParameterFree,

    /// <summary>A required dimension is unknown, so the slot count is not yet meaningful.</summary>
    ShapeDeferred,

    /// <summary>The shape is known but its storage has not yet been allocated.</summary>
    ShapeResolvedUnmaterialized,

    /// <summary>The slot has concrete storage and can be read or restored.</summary>
    Materialized
}

/// <summary>An immutable, stable description of one model-state slot.</summary>
public sealed class ParameterSlotDescriptor
{
    /// <summary>Creates a slot descriptor.</summary>
    public ParameterSlotDescriptor(
        string stableId,
        ParameterSlotRole role,
        ParameterReadiness readiness,
        long? parameterCount,
        long? offset = null)
    {
        if (string.IsNullOrWhiteSpace(stableId))
            throw new ArgumentException("A parameter slot requires a non-empty stable ID.", nameof(stableId));
        if (parameterCount < 0)
            throw new ArgumentOutOfRangeException(nameof(parameterCount));

        StableId = stableId;
        Role = role;
        Readiness = readiness;
        ParameterCount = parameterCount;
        Offset = offset;
    }

    /// <summary>A durable field/component path, independent of reflection and declaration order.</summary>
    public string StableId { get; }

    /// <summary>The semantic role of this slot.</summary>
    public ParameterSlotRole Role { get; }

    /// <summary>Whether the slot has a fully usable layout.</summary>
    public ParameterReadiness Readiness { get; }

    /// <summary>The resolved count, or <c>null</c> while the shape is deferred.</summary>
    public long? ParameterCount { get; }

    /// <summary>The offset in the selected flat layout, or <c>null</c> when any preceding count is unresolved.</summary>
    public long? Offset { get; }
}

/// <summary>A single deterministic snapshot consumed by count, vector, restore and checkpoint code.</summary>
public sealed class ParameterLayoutSnapshot
{
    /// <summary>Creates a layout snapshot from already ordered slots.</summary>
    public ParameterLayoutSnapshot(IReadOnlyList<ParameterSlotDescriptor> slots)
    {
        Slots = slots ?? throw new ArgumentNullException(nameof(slots));

        bool deferred = false;
        bool unmaterialized = false;
        bool materialized = false;
        long total = 0;
        for (int i = 0; i < slots.Count; i++)
        {
            var slot = slots[i];
            deferred |= slot.Readiness == ParameterReadiness.ShapeDeferred || !slot.ParameterCount.HasValue;
            unmaterialized |= slot.Readiness == ParameterReadiness.ShapeResolvedUnmaterialized;
            materialized |= slot.Readiness == ParameterReadiness.Materialized && slot.ParameterCount > 0;
            if (slot.ParameterCount.HasValue) total = checked(total + slot.ParameterCount.Value);
        }

        Readiness = slots.Count == 0 || (!deferred && !unmaterialized && !materialized && total == 0)
            ? ParameterReadiness.ParameterFree
            : deferred
                ? ParameterReadiness.ShapeDeferred
                : unmaterialized
                    ? ParameterReadiness.ShapeResolvedUnmaterialized
                    : ParameterReadiness.Materialized;
        ParameterCount = deferred ? null : total;
    }

    /// <summary>Slots in stable-ID order.</summary>
    public IReadOnlyList<ParameterSlotDescriptor> Slots { get; }

    /// <summary>The aggregate readiness of this snapshot.</summary>
    public ParameterReadiness Readiness { get; }

    /// <summary>The exact total, or <c>null</c> rather than a false zero when a shape is deferred.</summary>
    public long? ParameterCount { get; }
}

/// <summary>Implemented by sources that can describe their local slots without allocating storage.</summary>
public interface IParameterLayoutSource
{
    /// <summary>
    /// Returns local slot descriptors. IDs are relative to the component ID assigned by the owner.
    /// </summary>
    IReadOnlyList<ParameterSlotDescriptor> GetParameterLayout();
}

/// <summary>Implemented by models and layers that expose the generated parameter manifest.</summary>
public interface IParameterManifestProvider
{
    /// <summary>Gets a fresh, non-allocating snapshot of the current parameter layout.</summary>
    ParameterLayoutSnapshot ParameterLayout { get; }
}

/// <summary>
/// Implemented by generated partial classes so automated registration composes with hand-written
/// exceptional registration instead of either surface suppressing the other.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
public interface IGeneratedParameterRegistrar<T>
{
    /// <summary>Registers generated fields and components into the owning base's registry.</summary>
    void RegisterGeneratedParameters(ParameterComponentRegistry<T> registry);
}

/// <summary>Thrown when an exact vector operation is requested before every slot has a shape.</summary>
public sealed class ParameterLayoutNotReadyException : InvalidOperationException
{
    /// <summary>Creates a readiness error for the supplied operation.</summary>
    public ParameterLayoutNotReadyException(string operation, ParameterLayoutSnapshot layout)
        : base($"Cannot {operation} parameters while the layout is {layout.Readiness}. " +
               "Resolve model shapes or explicitly materialize parameters first; an unresolved " +
               "layout is not an empty parameter vector.")
    {
        Layout = layout ?? throw new ArgumentNullException(nameof(layout));
    }

    /// <summary>The layout that prevented the operation.</summary>
    public ParameterLayoutSnapshot Layout { get; }
}
