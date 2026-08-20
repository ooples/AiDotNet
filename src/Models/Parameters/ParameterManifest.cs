using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;
using System.Security.Cryptography;
using System.Text;

namespace AiDotNet.Models.Parameters;

/// <summary>Defines the canonical grammar used by durable parameter-manifest identities.</summary>
/// <remarks>
/// Stable IDs are opaque, ordinally sorted paths. A path segment that represents a numeric index
/// must contain exactly eight decimal digits (for example <c>layers/00000002</c>). Requiring one
/// width prevents adding index 10 from moving ahead of index 2 and changing every later checkpoint
/// offset. Semantic numbers must be named instead (for example <c>year=2024</c>) so they cannot be
/// mistaken for positional identity.
/// </remarks>
public static class ParameterStableId
{
    /// <summary>The fixed decimal width of an indexed path segment.</summary>
    public const int IndexWidth = 8;

    /// <summary>Formats a non-negative positional index as a canonical path segment.</summary>
    public static string IndexSegment(int index)
    {
        if (index < 0 || index > 99_999_999)
            throw new ArgumentOutOfRangeException(nameof(index),
                $"A parameter index must be between 0 and 99,999,999 for the {IndexWidth}-digit stable-ID grammar.");
        return index.ToString("D8", System.Globalization.CultureInfo.InvariantCulture);
    }

    internal static void Validate(string stableId, string parameterName)
    {
        if (string.IsNullOrWhiteSpace(stableId))
            throw new ArgumentException("A parameter component requires a stable ID.", parameterName);

        string[] segments = stableId.Split('/');
        for (int i = 0; i < segments.Length; i++)
        {
            string segment = segments[i];
            if (segment.Length == 0)
                throw new ArgumentException("A parameter stable ID cannot contain an empty path segment.", parameterName);
            if (segment == "." || segment == "..")
                throw new ArgumentException("A parameter stable ID cannot contain relative path segments.", parameterName);

            bool numeric = true;
            for (int j = 0; j < segment.Length; j++)
            {
                if (segment[j] < '0' || segment[j] > '9')
                {
                    numeric = false;
                    break;
                }
            }
            if (numeric && segment.Length != IndexWidth)
                throw new ArgumentException(
                    $"Numeric parameter path segment '{segment}' must contain exactly {IndexWidth} digits. " +
                    $"Use {nameof(ParameterStableId)}.{nameof(IndexSegment)} to create indexed identities.",
                    parameterName);
        }
    }
}

/// <summary>Describes how a numeric slot participates in model state and optimization.</summary>
public enum ParameterSlotRole
{
    /// <summary>A value an optimizer is allowed to update.</summary>
    Trainable,

    /// <summary>Fitted state that is restored but is not updated by a gradient optimizer.</summary>
    LearnedState,

    /// <summary>
    /// Fitted state whose extent comes from the caller's data, so it is restored by name but never
    /// joins the flat parameter vector.
    /// </summary>
    /// <remarks>
    /// <see cref="LearnedState"/> is sized once at construction, which is what lets it travel in
    /// the flat vector alongside the weights. A graph layer's adjacency matrix is sized by the
    /// graph it was handed, so counting it there would make ParameterCount change under a forward
    /// pass and would bind a checkpoint to the node count it was trained on. Same persistence, but
    /// deliberately outside the count.
    /// </remarks>
    InputSizedState,

    /// <summary>Persistent auxiliary state that is neither fitted nor optimizer-updated.</summary>
    Buffer,

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

/// <summary>Declares which mechanism is allowed to change a numeric state slot.</summary>
public enum ParameterUpdatePolicy
{
    Optimizer,
    Fit,
    Forward,
    Never,
    External
}

/// <summary>Declares whether a numeric state slot belongs in a durable checkpoint.</summary>
public enum ParameterPersistence
{
    Persistent,
    Transient
}

/// <summary>Declares whether this manifest owns the storage it names.</summary>
public enum ParameterOwnership
{
    Owned,
    Alias,
    External
}

/// <summary>Declares when a slot is expected to become available.</summary>
public enum ParameterAvailability
{
    Construction,
    ShapeResolution,
    Fit,
    Conditional,
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

    /// <summary>The slot is intentionally unavailable until the model has been fitted.</summary>
    FitDeferred,

    /// <summary>An explicitly conditional slot is absent from this concrete layout.</summary>
    ConditionalAbsent,

    /// <summary>The storage belongs to an external runtime and has no local payload.</summary>
    External,

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
        long? offset = null,
        IReadOnlyList<int>? shape = null,
        string? elementType = null,
        ParameterUpdatePolicy? updatePolicy = null,
        ParameterPersistence? persistence = null,
        ParameterOwnership? ownership = null,
        ParameterAvailability availability = ParameterAvailability.Construction,
        long? materializedParameterCount = null)
    {
        if (string.IsNullOrWhiteSpace(stableId))
            throw new ArgumentException("A parameter slot requires a non-empty stable ID.", nameof(stableId));
        if (parameterCount < 0)
            throw new ArgumentOutOfRangeException(nameof(parameterCount));

        long concreteCount = materializedParameterCount
            ?? (readiness == ParameterReadiness.Materialized ? parameterCount ?? 0 : 0);
        if (concreteCount < 0)
            throw new ArgumentOutOfRangeException(nameof(materializedParameterCount));
        if (parameterCount.HasValue && concreteCount > parameterCount.Value)
        {
            throw new ArgumentException(
                "A slot cannot materialize more values than its declared layout contains.",
                nameof(materializedParameterCount));
        }

        StableId = stableId;
        Role = role;
        Readiness = readiness;
        ParameterCount = parameterCount;
        MaterializedParameterCount = concreteCount;
        Offset = offset;
        if (shape is not null)
        {
            var immutableShape = new int[shape.Count];
            for (int i = 0; i < shape.Count; i++) immutableShape[i] = shape[i];
            Shape = Array.AsReadOnly(immutableShape);
        }
        ElementType = string.IsNullOrWhiteSpace(elementType) ? null : elementType;
        UpdatePolicy = updatePolicy ?? DefaultUpdatePolicy(role);
        Persistence = persistence ?? DefaultPersistence(role);
        Ownership = ownership ?? DefaultOwnership(role);
        Availability = availability;
    }

    /// <summary>A durable field/component path, independent of reflection and declaration order.</summary>
    public string StableId { get; }

    /// <summary>The semantic role of this slot.</summary>
    public ParameterSlotRole Role { get; }

    /// <summary>Whether the slot has a fully usable layout.</summary>
    public ParameterReadiness Readiness { get; }

    /// <summary>The resolved count, or <c>null</c> while the shape is deferred.</summary>
    public long? ParameterCount { get; }

    /// <summary>
    /// The exact number of values currently backed by readable storage. Unlike
    /// <see cref="ParameterCount"/>, this never anticipates a future lazy allocation.
    /// </summary>
    public long MaterializedParameterCount { get; }

    /// <summary>The offset in the selected flat layout, or <c>null</c> when any preceding count is unresolved.</summary>
    public long? Offset { get; }

    /// <summary>
    /// The concrete or symbolic tensor shape. A negative dimension is unresolved; <c>null</c>
    /// means the source can describe only a scalar count and must not claim shape conformance.
    /// </summary>
    public IReadOnlyList<int>? Shape { get; }

    /// <summary>
    /// Canonical numeric element type used by shape-aware checkpoint and schema validation.
    /// </summary>
    public string? ElementType { get; }

    /// <summary>The only mechanism permitted to mutate this slot.</summary>
    public ParameterUpdatePolicy UpdatePolicy { get; }

    /// <summary>Whether this slot participates in durable checkpoint state.</summary>
    public ParameterPersistence Persistence { get; }

    /// <summary>Whether this slot owns, aliases, or externally references its storage.</summary>
    public ParameterOwnership Ownership { get; }

    /// <summary>The lifecycle point at which this slot is expected to become available.</summary>
    public ParameterAvailability Availability { get; }

    private static ParameterUpdatePolicy DefaultUpdatePolicy(ParameterSlotRole role) => role switch
    {
        ParameterSlotRole.Trainable => ParameterUpdatePolicy.Optimizer,
        ParameterSlotRole.LearnedState => ParameterUpdatePolicy.Fit,
        ParameterSlotRole.Gradient or ParameterSlotRole.Scratch => ParameterUpdatePolicy.Forward,
        ParameterSlotRole.External => ParameterUpdatePolicy.External,
        _ => ParameterUpdatePolicy.Never
    };

    private static ParameterPersistence DefaultPersistence(ParameterSlotRole role) => role switch
    {
        ParameterSlotRole.Gradient or ParameterSlotRole.Scratch or ParameterSlotRole.Alias
            or ParameterSlotRole.External => ParameterPersistence.Transient,
        _ => ParameterPersistence.Persistent
    };

    private static ParameterOwnership DefaultOwnership(ParameterSlotRole role) => role switch
    {
        ParameterSlotRole.Alias => ParameterOwnership.Alias,
        ParameterSlotRole.External => ParameterOwnership.External,
        _ => ParameterOwnership.Owned
    };
}

/// <summary>Shared structural comparisons for parameter-manifest shapes.</summary>
internal static class ParameterShapeComparer
{
    /// <summary>Returns whether two optional shape declarations are element-for-element equal.</summary>
    internal static bool AreEqual(IReadOnlyList<int>? left, IReadOnlyList<int>? right)
    {
        if (left is null || right is null) return left is null && right is null;
        if (left.Count != right.Count) return false;
        for (int i = 0; i < left.Count; i++)
        {
            if (left[i] != right[i]) return false;
        }
        return true;
    }
}

/// <summary>A single deterministic snapshot consumed by count, vector, restore and checkpoint code.</summary>
public sealed class ParameterLayoutSnapshot
{
    /// <summary>The canonical manifest schema used to compute <see cref="Fingerprint"/>.</summary>
    public const int CurrentSchemaVersion = 5;

    /// <summary>Creates a layout snapshot from already ordered slots.</summary>
    public ParameterLayoutSnapshot(IReadOnlyList<ParameterSlotDescriptor> slots)
    {
        if (slots is null) throw new ArgumentNullException(nameof(slots));
        var immutableSlots = new List<ParameterSlotDescriptor>(slots.Count);
        var stableIds = new HashSet<string>(StringComparer.Ordinal);
        for (int i = 0; i < slots.Count; i++)
        {
            var slot = slots[i] ?? throw new ArgumentException(
                $"Parameter manifest slot {i} is null.",
                nameof(slots));
            ParameterStableId.Validate(slot.StableId, nameof(slots));
            if (!stableIds.Add(slot.StableId))
            {
                throw new ArgumentException(
                    $"Parameter manifest contains duplicate stable identity '{slot.StableId}'. " +
                    "Every persistent-state slot must have one unique, durable owner path.",
                    nameof(slots));
            }
            immutableSlots.Add(slot);
        }
        Slots = immutableSlots.AsReadOnly();

        bool shapeDeferred = false;
        bool fitDeferred = false;
        bool conditionalAbsent = false;
        bool external = false;
        bool unmaterialized = false;
        bool materialized = false;
        long declaredTotal = 0;
        long materializedTotal = 0;
        for (int i = 0; i < slots.Count; i++)
        {
            var slot = slots[i];
            shapeDeferred |= slot.Readiness == ParameterReadiness.ShapeDeferred;
            fitDeferred |= slot.Readiness == ParameterReadiness.FitDeferred;
            conditionalAbsent |= slot.Readiness == ParameterReadiness.ConditionalAbsent;
            external |= slot.Readiness == ParameterReadiness.External;
            unmaterialized |= slot.Readiness == ParameterReadiness.ShapeResolvedUnmaterialized;
            materialized |= slot.Readiness == ParameterReadiness.Materialized && slot.ParameterCount > 0;
            if (slot.ParameterCount.HasValue)
                declaredTotal = checked(declaredTotal + slot.ParameterCount.Value);
            materializedTotal = checked(materializedTotal + slot.MaterializedParameterCount);
        }

        bool unresolved = shapeDeferred || fitDeferred
            || slots.Any(slot => !slot.ParameterCount.HasValue);
        Readiness = slots.Count == 0 || (!unresolved && !unmaterialized && !materialized
                                        && !conditionalAbsent && !external && declaredTotal == 0)
            ? ParameterReadiness.ParameterFree
            : shapeDeferred
                ? ParameterReadiness.ShapeDeferred
                : fitDeferred
                    ? ParameterReadiness.FitDeferred
                : unmaterialized
                    ? ParameterReadiness.ShapeResolvedUnmaterialized
                    : materialized
                        ? ParameterReadiness.Materialized
                        : conditionalAbsent
                            ? ParameterReadiness.ConditionalAbsent
                            : external
                                ? ParameterReadiness.External
                                : ParameterReadiness.ParameterFree;
        KnownParameterCount = declaredTotal;
        ParameterCount = unresolved ? null : declaredTotal;
        MaterializedParameterCount = materializedTotal;
        Fingerprint = ComputeFingerprint(immutableSlots);
    }

    /// <summary>Slots in stable-ID order.</summary>
    public IReadOnlyList<ParameterSlotDescriptor> Slots { get; }

    /// <summary>The aggregate readiness of this snapshot.</summary>
    public ParameterReadiness Readiness { get; }

    /// <summary>The exact total, or <c>null</c> rather than a false zero when a shape is deferred.</summary>
    public long? ParameterCount { get; }

    /// <summary>
    /// The exact width of every slot whose size is currently known, including resolved slots that
    /// have deliberately not allocated their backing storage yet.
    /// </summary>
    /// <remarks>
    /// This equals <see cref="ParameterCount"/> when the complete layout is resolved. While one or
    /// more slots remain deferred it is an explicit lower bound, not a guessed total: deferred
    /// slots contribute nothing, while independently resolved slots retain their real width. Flat
    /// read boundaries use this value because those resolved slots materialize on demand even when
    /// an unrelated slot is still waiting for shape or fit information.
    /// </remarks>
    public long KnownParameterCount { get; }

    /// <summary>
    /// The structural total when every slot can be sized, or <c>null</c> while any required
    /// shape is unresolved. This names the planning meaning of <see cref="ParameterCount"/>.
    /// </summary>
    public long? DeclaredParameterCount => ParameterCount;

    /// <summary>
    /// The exact number of values the live flat parameter surface can emit without allocating
    /// lazy storage. This value is always concrete, including for partially built graphs.
    /// </summary>
    public long MaterializedParameterCount { get; }

    /// <summary>The version of the canonical manifest representation.</summary>
    public int SchemaVersion => CurrentSchemaVersion;

    /// <summary>
    /// A SHA-256 digest of stable IDs, semantic roles and resolved counts in canonical order.
    /// Checkpoints can persist this value and reject a layout mismatch before applying any values.
    /// </summary>
    public string Fingerprint { get; }

    private static string ComputeFingerprint(IReadOnlyList<ParameterSlotDescriptor> slots)
    {
        var canonical = new StringBuilder();
        canonical.Append("parameter-manifest-v").Append(CurrentSchemaVersion).Append('\n');
        for (int i = 0; i < slots.Count; i++)
        {
            var slot = slots[i];
            canonical.Append(slot.StableId.Length).Append(':').Append(slot.StableId).Append('|')
                .Append((int)slot.Role).Append('|')
                .Append((int)slot.Readiness).Append('|')
                .Append((int)slot.UpdatePolicy).Append('|')
                .Append((int)slot.Persistence).Append('|')
                .Append((int)slot.Ownership).Append('|')
                .Append((int)slot.Availability).Append('|')
                .Append(slot.ParameterCount.HasValue ? slot.ParameterCount.Value.ToString(
                    System.Globalization.CultureInfo.InvariantCulture) : "?").Append('|')
                .Append(slot.MaterializedParameterCount.ToString(
                    System.Globalization.CultureInfo.InvariantCulture)).Append('|')
                .Append(slot.ElementType ?? "?").Append('|');
            if (slot.Shape is null)
            {
                canonical.Append('?');
            }
            else
            {
                canonical.Append(slot.Shape.Count).Append(':');
                for (int axis = 0; axis < slot.Shape.Count; axis++)
                {
                    if (axis > 0) canonical.Append(',');
                    canonical.Append(slot.Shape[axis].ToString(
                        System.Globalization.CultureInfo.InvariantCulture));
                }
            }
            canonical
                .Append('\n');
        }

        using var sha256 = SHA256.Create();
        byte[] digest = sha256.ComputeHash(Encoding.UTF8.GetBytes(canonical.ToString()));
        return BitConverter.ToString(digest).Replace("-", string.Empty).ToLowerInvariant();
    }
}

/// <summary>
/// Thrown when a parameter source's concrete vector disagrees with the manifest snapshot that
/// describes it.
/// </summary>
/// <remarks>
/// Continuing after this condition would shift every following component and silently restore
/// values into the wrong owner. The registry therefore fails before returning or applying a
/// partial vector.
/// </remarks>
public sealed class ParameterContractViolationException : InvalidOperationException
{
    /// <summary>Creates a count-drift failure for one stable component identity.</summary>
    public ParameterContractViolationException(
        string operation,
        string stableId,
        long expectedCount,
        long actualCount)
        : base($"Cannot {operation} parameter component '{stableId}': its captured manifest " +
               $"declares {expectedCount} values but the source supplied {actualCount}. " +
               "The operation was stopped to prevent checkpoint offset corruption.")
    {
        Operation = operation;
        StableId = stableId;
        ExpectedCount = expectedCount;
        ActualCount = actualCount;
    }

    /// <summary>The operation that detected the disagreement.</summary>
    public string Operation { get; }

    /// <summary>The durable component identity whose source violated the contract.</summary>
    public string StableId { get; }

    /// <summary>The count captured from the manifest.</summary>
    public long ExpectedCount { get; }

    /// <summary>The concrete count returned by the source.</summary>
    public long ActualCount { get; }
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
    /// <summary>Gets a fresh immutable snapshot of the current parameter layout.</summary>
    ParameterLayoutSnapshot ParameterLayout { get; }
}

/// <summary>
/// Implemented by lazy parameter sources that can make every shape-resolved slot writable.
/// </summary>
/// <remarks>
/// Read operations may remain allocation-free and report
/// <see cref="ParameterReadiness.ShapeResolvedUnmaterialized"/>. A restore is an explicit write,
/// so registries call this hook before capturing the destination layout; otherwise a zero-sized
/// lazy snapshot can accept a vector and then change its boundaries midway through application.
/// </remarks>
public interface IParameterMaterializationSource
{
    /// <summary>Allocates every parameter whose shape is already known.</summary>
    void MaterializeParameters();
}

/// <summary>Identifies why a caller is preparing a parameter surface.</summary>
/// <remarks>
/// A shape-only metadata query and a concrete value read are deliberately different operations.
/// Treating both as "materialize everything" makes foundation-scale model inspection allocate
/// billions of values; treating both as "describe whatever exists now" lets the first value read
/// grow the manifest after its offsets have already been captured. This intent is the common
/// vocabulary used by generated component accessors, model bases and the registry transaction.
/// </remarks>
public enum ParameterSurfaceIntent
{
    /// <summary>Resolve construction-known structure and shapes without allocating values.</summary>
    Describe,

    /// <summary>Prepare all shape-resolved values for one exact flat read.</summary>
    Read,

    /// <summary>Prepare values for stable-ID ordered, potentially streaming enumeration.</summary>
    EnumerateChunks,

    /// <summary>Prepare a destination before applying an exact checkpoint payload.</summary>
    Restore,

    /// <summary>Prepare durable state before a checkpoint manifest is captured.</summary>
    Checkpoint,

    /// <summary>Prepare optimizer-visible state before the first training snapshot.</summary>
    Train
}

/// <summary>
/// Implemented by parameter sources whose structure or storage becomes ready in lifecycle stages.
/// </summary>
/// <remarks>
/// <para>
/// This is a capability contract, not a trainability inference. A source may contain trainable,
/// frozen, fitted, buffered, or external slots; their explicit manifest roles remain authoritative.
/// </para>
/// <para>
/// Implementations must be idempotent. <see cref="ParameterSurfaceIntent.Describe"/> must not
/// allocate parameter storage, while concrete intents may materialize only slots whose shapes are
/// already resolved. Truly data-dependent shapes remain deferred and produce the normal readiness
/// diagnostic rather than being guessed from a dummy forward.
/// </para>
/// </remarks>
public interface IParameterSurfaceLifecycle
{
    /// <summary>Advances this source only as far as the requested operation requires.</summary>
    void PrepareParameterSurface(ParameterSurfaceIntent intent);
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

/// <summary>Thrown when an exact vector operation is requested before every slot is ready.</summary>
public sealed class ParameterLayoutNotReadyException : InvalidOperationException
{
    /// <summary>Creates a readiness error for the supplied operation.</summary>
    public ParameterLayoutNotReadyException(string operation, ParameterLayoutSnapshot layout)
        : base($"Cannot {operation} parameters while the layout is {layout.Readiness}. " +
               $"Unresolved slots: {DescribeUnresolvedSlots(layout)}. " +
               ReadinessGuidance(layout))
    {
        Layout = layout ?? throw new ArgumentNullException(nameof(layout));
    }

    /// <summary>The layout that prevented the operation.</summary>
    public ParameterLayoutSnapshot Layout { get; }

    private static string DescribeUnresolvedSlots(ParameterLayoutSnapshot? layout)
    {
        if (layout is null) return "<unknown>";
        var ids = new List<string>();
        for (int i = 0; i < layout.Slots.Count && ids.Count < 8; i++)
        {
            var slot = layout.Slots[i];
            if (slot.Readiness is ParameterReadiness.ShapeDeferred
                or ParameterReadiness.ShapeResolvedUnmaterialized
                or ParameterReadiness.FitDeferred
                or ParameterReadiness.External
                || !slot.ParameterCount.HasValue)
                ids.Add(slot.StableId);
        }
        if (ids.Count == 0) return "<none>";
        return string.Join(", ", ids) + (ids.Count < 8 ? string.Empty : ", ...");
    }

    private static string ReadinessGuidance(ParameterLayoutSnapshot? layout)
        => layout?.Readiness == ParameterReadiness.FitDeferred
            ? "Fit the model before reading or restoring fitted parameter state; fit-deferred state is not an empty parameter vector."
            : "Resolve model shapes or explicitly materialize parameters first; an unresolved layout is not an empty parameter vector.";
}
