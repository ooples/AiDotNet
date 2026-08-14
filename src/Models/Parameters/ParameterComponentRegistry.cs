using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;
using System.Security.Cryptography;
using System.Text;

namespace AiDotNet.Models.Parameters;

/// <summary>
/// Owns a model's deterministic parameter manifest and derives count, flat values and restore from
/// the same ordered component snapshot.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
public sealed class ParameterComponentRegistry<T> : IParameterManifestProvider
{
    private sealed class Entry
    {
        public Entry(string stableId, IParameterSource<T>? source, ParameterSlotRole role,
            ParameterAvailability availability)
        {
            StableId = stableId;
            Source = source;
            Role = role;
            Availability = availability;
        }

        public string StableId { get; }
        public IParameterSource<T>? Source { get; }
        public ParameterSlotRole Role { get; }
        public ParameterAvailability Availability { get; }
    }

    private sealed class CapturedEntry
    {
        public CapturedEntry(Entry entry, IReadOnlyList<ParameterSlotDescriptor> localSlots,
                             long? parameterCount)
        {
            Entry = entry;
            LocalSlots = localSlots;
            ParameterCount = parameterCount;
        }

        public Entry Entry { get; }
        public IReadOnlyList<ParameterSlotDescriptor> LocalSlots { get; }
        public long? ParameterCount { get; }
    }

    private sealed class CapturedLayout
    {
        public CapturedLayout(ParameterLayoutSnapshot snapshot, IReadOnlyList<CapturedEntry> entries)
        {
            Snapshot = snapshot;
            Entries = entries;
        }

        public ParameterLayoutSnapshot Snapshot { get; }
        public IReadOnlyList<CapturedEntry> Entries { get; }
    }

    private readonly List<Entry> _entries = new();
    private readonly Dictionary<string, int> _legacyOccurrences = new(StringComparer.Ordinal);

    /// <summary>The registered, currently present sources in stable-ID order.</summary>
    public IReadOnlyList<IParameterSource<T>> Components
    {
        get
        {
            var ordered = OrderedEntries();
            var result = new List<IParameterSource<T>>(ordered.Count);
            for (int i = 0; i < ordered.Count; i++)
            {
                if (ordered[i].Source is not null) result.Add(ordered[i].Source!);
            }
            return result;
        }
    }

    /// <summary>True once at least one component identity has been registered.</summary>
    public bool HasComponents => _entries.Count > 0;

    /// <summary>
    /// True when the manifest contains state that defines the model itself rather than auxiliary
    /// buffers. Used by compatibility bases while their remaining packed state is migrated.
    /// </summary>
    public bool HasPrimaryParameterComponents => _entries.Any(entry => entry.Role is
        ParameterSlotRole.Trainable or ParameterSlotRole.LearnedState or ParameterSlotRole.Frozen);

    /// <summary>
    /// True only when the manifest explicitly contains optimizer-updatable state.
    /// </summary>
    /// <remarks>
    /// Count is deliberately not used as a proxy. Learned thresholds, frozen weights and replay
    /// buffers are persistent numbers, but randomizing them as trainable parameters is a semantic
    /// error even when their count is non-zero.
    /// </remarks>
    public bool HasOptimizerUpdatableComponents =>
        _entries.Any(entry => entry.Role == ParameterSlotRole.Trainable);

    /// <summary>
    /// True when optimizer-owned state exists and the complete flat layout is resolved now.
    /// </summary>
    /// <remarks>
    /// This is the capability gate consumed by model bases. It deliberately combines semantic
    /// ownership with lifecycle readiness: a fitted coefficient is not optimizer-owned, and an
    /// optimizer-owned slot whose shape is still deferred cannot safely accept an initialization
    /// vector yet. Neither fact can be inferred from a raw numeric count.
    /// </remarks>
    public bool CanInitializeOptimizerParameters
    {
        get
        {
            var layout = CaptureLayout().Snapshot;
            if (!layout.ParameterCount.HasValue) return false;
            return layout.Slots.Any(slot =>
                slot.Role == ParameterSlotRole.Trainable
                && slot.ParameterCount.GetValueOrDefault() > 0);
        }
    }

    /// <summary>
    /// Registers a legacy component using its source type as a deterministic identity seed.
    /// Generated and model-owned code should use an explicit stable ID or
    /// <see cref="RegisterLegacy(string, string?, string?, IParameterSource{T}?)"/> so unrelated
    /// registrations cannot renumber it.
    /// </summary>
    public void Register(IParameterSource<T>? component)
    {
        string sourceType = component?.GetType().FullName ?? "null";
        RegisterLegacy(sourceType, nameof(Register), sourceType, component);
    }

    /// <summary>
    /// Registers an un-migrated caller under a deterministic compatibility identity.
    /// </summary>
    /// <remarks>
    /// The old compatibility path assigned one global sequence number, so inserting an unrelated
    /// component changed every following checkpoint offset. This seed isolates identity by owning
    /// model, registration member and caller expression. Repeated entries from the same expression
    /// (for example a collection loop) receive a fixed-width local occurrence index; those callers
    /// should still migrate to explicit semantic collection IDs when such IDs exist.
    /// </remarks>
    internal void RegisterLegacy(
        string ownerIdentity,
        string? memberName,
        string? argumentExpression,
        IParameterSource<T>? component)
    {
        for (int i = 0; i < _entries.Count; i++)
        {
            if (component is not null && ReferencesSameSource(_entries[i].Source, component)) return;
        }

        string seed = "legacy-v1\n" + ownerIdentity + "\n" +
            (memberName ?? "unknown-member") + "\n" +
            (argumentExpression ?? "unknown-expression");
        string digest;
        using (var sha256 = SHA256.Create())
        {
            byte[] bytes = sha256.ComputeHash(Encoding.UTF8.GetBytes(seed));
            digest = BitConverter.ToString(bytes).Replace("-", string.Empty).ToLowerInvariant();
        }

        int occurrence = _legacyOccurrences.TryGetValue(digest, out int current) ? current : 0;
        _legacyOccurrences[digest] = checked(occurrence + 1);
        Register($"legacy-v1/{digest}/{ParameterStableId.IndexSegment(occurrence)}", component,
            ParameterSlotRole.Trainable);
    }

    /// <summary>Registers a component by durable identity and semantic role.</summary>
    public void Register(string stableId, IParameterSource<T>? component,
                         ParameterSlotRole role = ParameterSlotRole.Trainable,
                         ParameterAvailability availability = ParameterAvailability.Construction)
    {
        ParameterStableId.Validate(stableId, nameof(stableId));

        for (int i = 0; i < _entries.Count; i++)
        {
            var existing = _entries[i];
            if (component is not null && ReferencesSameSource(existing.Source, component))
            {
                if (existing.Role == role && existing.Availability == availability) return;
                throw new InvalidOperationException(
                    $"Parameter component '{stableId}' refers to storage already registered as " +
                    $"'{existing.StableId}' with role {existing.Role} and availability " +
                    $"{existing.Availability}; the same storage cannot also be registered with " +
                    $"role {role} and availability {availability}.");
            }

            if (!string.Equals(existing.StableId, stableId, StringComparison.Ordinal)) continue;
            if (ReferenceEquals(existing.Source, component) && existing.Role == role
                && existing.Availability == availability) return;
            throw new InvalidOperationException(
                $"Parameter component ID '{stableId}' was registered more than once. Stable IDs " +
                "must identify exactly one owner; use an Alias role rather than duplicate storage.");
        }

        _entries.Add(new Entry(stableId, component, role, availability));
    }

    /// <inheritdoc />
    public ParameterLayoutSnapshot ParameterLayout
        => CaptureLayout().Snapshot;

    /// <summary>
    /// The exact resolved count. An unresolved layout throws rather than masquerading as a partial
    /// or zero-sized model; shape-aware callers can inspect <see cref="ParameterLayout"/> first.
    /// </summary>
    public long ParameterCount
    {
        get
        {
            if (TryComputeCount(out long total)) return total;
            throw new ParameterLayoutNotReadyException("count", ParameterLayout);
        }
    }

    /// <summary>
    /// Sums the same slots <see cref="ParameterLayout"/> describes, without materializing them.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Applies the identical deferred rule the snapshot applies -- a slot defers the whole model if
    /// its readiness says <see cref="ParameterReadiness.ShapeDeferred"/> OR it carries no count --
    /// so this and the snapshot cannot disagree about either the number or whether there is one.
    /// </para>
    /// <para>
    /// Exists because building the snapshot allocates a descriptor per slot, and the surfaces this
    /// registry replaced were a plain sum that callers read once per training step. The full
    /// snapshot is still built on the failure path, where the caller wants it for diagnostics.
    /// </para>
    /// </remarks>
    private bool TryComputeCount(out long total)
    {
        total = 0;
        var ordered = OrderedEntries();

        for (int i = 0; i < ordered.Count; i++)
        {
            var entry = ordered[i];
            var source = entry.Source;
            if (source is null) return false;

            if (source is IParameterLayoutSource layoutSource)
            {
                var local = layoutSource.GetParameterLayout();
                for (int j = 0; j < local.Count; j++)
                {
                    var slot = local[j];
                    if (slot.Readiness == ParameterReadiness.ShapeDeferred
                        || !slot.ParameterCount.HasValue)
                    {
                        bool concreteAbsence = entry.Availability == ParameterAvailability.Conditional
                            || entry.Availability == ParameterAvailability.External
                            || (entry.Role == ParameterSlotRole.Buffer
                                && entry.Availability != ParameterAvailability.Fit);
                        if (concreteAbsence)
                            continue;
                        return false;
                    }

                    total = checked(total + slot.ParameterCount.Value);
                }
            }
            else
            {
                total = checked(total + source.ParameterCount);
            }
        }

        return true;
    }

    /// <summary>Concatenates the same stable-ID ordered entries described by the manifest.</summary>
    public Vector<T> GetParameters()
    {
        var captured = CaptureLayout();
        var layout = captured.Snapshot;
        if (!layout.ParameterCount.HasValue)
            throw new ParameterLayoutNotReadyException("read", layout);

        int total = checked((int)layout.ParameterCount.Value);
        var result = new Vector<T>(total);
        int offset = 0;
        for (int i = 0; i < captured.Entries.Count; i++)
        {
            var item = captured.Entries[i];
            var source = item.Entry.Source;
            if (source is null) continue;
            int expected = checked((int)item.ParameterCount!.Value);
            if (expected == 0) continue;
            var part = source.GetParameters();
            if (part.Length != expected)
                throw new ParameterContractViolationException(
                    "read", item.Entry.StableId, expected, part.Length);

            part.AsSpan().CopyTo(result.AsWritableSpan().Slice(offset, expected));
            offset += expected;
        }
        return result;
    }

    /// <summary>
    /// Yields the registered state in stable-ID order. Tensor-backed child sources keep their
    /// zero-copy chunks; classical sources receive one exact flat payload chunk.
    /// </summary>
    public IEnumerable<ParameterChunk<T>> GetParameterStateChunks()
    {
        var captured = CaptureLayout();
        if (!captured.Snapshot.ParameterCount.HasValue)
            throw new ParameterLayoutNotReadyException("enumerate chunks", captured.Snapshot);

        for (int i = 0; i < captured.Entries.Count; i++)
        {
            var item = captured.Entries[i];
            var entry = item.Entry;
            var source = entry.Source;
            if (source is null) continue;
            int expected = checked((int)item.ParameterCount!.Value);
            if (expected == 0) continue;

            if (source is IParameterChunkSource<T> chunkSource)
            {
                int actual = 0;
                foreach (var chunk in chunkSource.GetParameterStateChunks())
                {
                    if (chunk is null || chunk.Tensor.Length == 0) continue;
                    actual = checked(actual + chunk.Tensor.Length);
                    var role = entry.Role == ParameterSlotRole.Trainable
                        ? chunk.Role
                        : entry.Role;
                    string localId = chunk.StableId == "$"
                        ? entry.StableId
                        : entry.StableId + "/" + chunk.StableId;
                    yield return new ParameterChunk<T>(localId, role, chunk.Tensor);
                }
                if (actual != expected)
                    throw new ParameterContractViolationException(
                        "enumerate chunks", entry.StableId, expected, actual);
                continue;
            }

            var flat = source.GetParameters();
            if (flat.Length != expected)
                throw new ParameterContractViolationException(
                    "enumerate chunks", entry.StableId, expected, flat.Length);
            if (expected == 0) continue;

            if (source is IParameterLayoutSource)
            {
                int offset = 0;
                for (int j = 0; j < item.LocalSlots.Count; j++)
                {
                    var slot = item.LocalSlots[j];
                    if (!slot.ParameterCount.HasValue)
                        throw new ParameterLayoutNotReadyException(
                            "enumerate chunks", captured.Snapshot);
                    int count = checked((int)slot.ParameterCount.Value);
                    if (count == 0) continue;
                    if (offset + count > flat.Length)
                        throw new InvalidOperationException(
                            $"Parameter layout for '{entry.StableId}' describes more values than " +
                            "its flat parameter source contains.");

                    var values = new Vector<T>(count);
                    flat.AsSpan().Slice(offset, count).CopyTo(values.AsWritableSpan());
                    string localId = slot.StableId == "$"
                        ? entry.StableId
                        : entry.StableId + "/" + slot.StableId;
                    var role = entry.Role == ParameterSlotRole.Trainable ? slot.Role : entry.Role;
                    yield return new ParameterChunk<T>(
                        localId, role, new Tensor<T>(new[] { count }, values));
                    offset += count;
                }
                if (offset != flat.Length)
                    throw new InvalidOperationException(
                        $"Parameter layout for '{entry.StableId}' describes {offset} values but " +
                        $"its flat parameter source contains {flat.Length}.");
                continue;
            }

            // Tensor(Vector) shares the Vector's storage. For native tensor sources the branch
            // above supplies the model's real backing tensor; this fallback is the explicit,
            // immutable-payload style used by scalar/tree/classical sources.
            yield return new ParameterChunk<T>(entry.StableId, entry.Role,
                new Tensor<T>(new[] { flat.Length }, flat));
        }
    }

    /// <summary>Restores slices using the exact manifest snapshot used to validate the vector.</summary>
    public void SetParameters(Vector<T> parameters)
    {
        if (parameters is null) throw new ArgumentNullException(nameof(parameters));

        var captured = CaptureLayout();
        int variableIndex = FindVariableEntryIndex(captured.Entries);
        if (variableIndex >= 0 &&
            (!captured.Snapshot.ParameterCount.HasValue ||
             parameters.Length != captured.Snapshot.ParameterCount.Value))
        {
            // A variable tail gets whatever remains after every fixed component. Materialize fixed
            // neural networks first so their checkpoint spans are known before the remainder is
            // calculated; the variable source itself deliberately learns its width from the slice.
            for (int i = 0; i < variableIndex; i++)
            {
                if (TryGetNetwork(captured.Entries[i].Entry.Source, out var network))
                    network.MaterializeParameters();
            }
            captured = CaptureLayout();
            variableIndex = FindVariableEntryIndex(captured.Entries);
        }
        else if (captured.Snapshot.ParameterCount.HasValue &&
            parameters.Length != captured.Snapshot.ParameterCount.Value)
        {
            // Materialize in stable flat-vector order and stop as soon as the incoming length is
            // explained. A checkpoint may honestly contain only the prefix of networks that have
            // executed (A3C can run its policy before its value network); materializing every fresh
            // component would invent a longer target layout than the source actually emitted.
            for (int i = 0; i < captured.Entries.Count; i++)
            {
                if (TryGetNetwork(captured.Entries[i].Entry.Source, out var network))
                    network.MaterializeParameters();

                captured = CaptureLayout();
                long? recapturedCount = captured.Snapshot.ParameterCount;
                if (!recapturedCount.HasValue || recapturedCount.Value >= parameters.Length) break;
            }
        }
        var layout = captured.Snapshot;
        long fixedParameterCount;
        if (variableIndex < 0)
        {
            if (!layout.ParameterCount.HasValue)
                throw new ParameterLayoutNotReadyException("restore", layout);
            fixedParameterCount = layout.ParameterCount.Value;
        }
        else
        {
            fixedParameterCount = 0;
            for (int i = 0; i < captured.Entries.Count; i++)
            {
                if (i == variableIndex) continue;
                if (!captured.Entries[i].ParameterCount.HasValue)
                    throw new ParameterLayoutNotReadyException("restore", layout);
                fixedParameterCount = checked(
                    fixedParameterCount + captured.Entries[i].ParameterCount!.Value);
            }
        }
        if (variableIndex < 0 && parameters.Length != fixedParameterCount)
            throw new ArgumentException(
                $"Expected {fixedParameterCount} parameters, got {parameters.Length}.",
                nameof(parameters));
        if (variableIndex >= 0 && parameters.Length < fixedParameterCount)
            throw new ArgumentException(
                $"Expected at least {fixedParameterCount} parameters for the fixed components, " +
                $"got {parameters.Length}.", nameof(parameters));

        // Slice by the captured layout, never by re-querying live source counts. The same immutable
        // snapshot validated the total above and records each source's declared span, so lazy
        // materialization cannot shift every component that follows it during restore.
        int offset = 0;
        for (int i = 0; i < captured.Entries.Count; i++)
        {
            var item = captured.Entries[i];
            var source = item.Entry.Source;
            if (source is null) continue;
            int count = i == variableIndex
                ? checked(parameters.Length - offset)
                : checked((int)item.ParameterCount!.Value);
            if (count == 0) continue;
            var slice = new Vector<T>(count);
            parameters.AsSpan().Slice(offset, count).CopyTo(slice.AsWritableSpan());
            offset += count;
            source.SetParameters(slice);
        }
    }

    /// <summary>
    /// Materializes nested neural-network sources before a checkpoint manifest is captured.
    /// </summary>
    /// <remarks>
    /// Neural-network serialization materializes lazy weights. A containing model must do the
    /// same before capturing its own flat vector; otherwise the nested serializer can grow the
    /// layout after the outer snapshot has already been written.
    /// </remarks>
    public void MaterializeCheckpointSources()
    {
        foreach (var entry in OrderedEntries())
        {
            if (TryGetNetwork(entry.Source, out var network))
                network.MaterializeParameters();
        }
    }

    private CapturedLayout CaptureLayout()
    {
        var ordered = OrderedEntries();
        var slots = new List<ParameterSlotDescriptor>();
        var captured = new List<CapturedEntry>(ordered.Count);
        var identities = new HashSet<string>(StringComparer.Ordinal);
        long offset = 0;
        bool offsetKnown = true;

        for (int i = 0; i < ordered.Count; i++)
        {
            var entry = ordered[i];
            IReadOnlyList<ParameterSlotDescriptor> local;
            if (entry.Source is null)
            {
                local = new[]
                {
                    new ParameterSlotDescriptor(
                        "$", entry.Role, ParameterReadiness.ShapeDeferred, null)
                };
            }
            else if (entry.Source is IParameterManifestProvider manifestProvider)
            {
                local = manifestProvider.ParameterLayout.Slots;
            }
            else if (entry.Source is IParameterLayoutSource layoutSource)
            {
                local = layoutSource.GetParameterLayout()
                    ?? throw new InvalidOperationException(
                        $"Parameter component '{entry.StableId}' returned a null layout.");
            }
            else
            {
                long count = entry.Source.ParameterCount;
                if (count < 0)
                    throw new ParameterContractViolationException(
                        "capture layout", entry.StableId, 0, count);
                local = new[]
                {
                    new ParameterSlotDescriptor(
                        "$", entry.Role,
                        count == 0 ? ParameterReadiness.ParameterFree : ParameterReadiness.Materialized,
                        count)
                };
            }

            long entryCount = 0;
            bool entryCountKnown = true;
            for (int j = 0; j < local.Count; j++)
            {
                var localSlot = local[j] ?? throw new InvalidOperationException(
                    $"Parameter component '{entry.StableId}' returned a null layout slot.");
                var normalizedReadiness = localSlot.Readiness;
                long? normalizedCount = localSlot.ParameterCount;
                if (!normalizedCount.HasValue)
                {
                    if (entry.Availability == ParameterAvailability.Fit)
                    {
                        // A fit-produced buffer is not an optional architecture branch. Treating
                        // all absent buffers as ConditionalAbsent made fresh fitted models look
                        // like concrete zero-parameter models and let gates pass vacuously.
                        normalizedReadiness = ParameterReadiness.FitDeferred;
                    }
                    else if (entry.Availability == ParameterAvailability.Conditional
                        || entry.Role == ParameterSlotRole.Buffer)
                    {
                        // An absent optional/buffer slot contributes no values to this concrete
                        // snapshot. It may appear in a later snapshot, but must not block unrelated
                        // parameter reads as though a trainable shape were missing.
                        normalizedReadiness = ParameterReadiness.ConditionalAbsent;
                        normalizedCount = 0;
                    }
                    else if (entry.Availability == ParameterAvailability.External)
                    {
                        normalizedReadiness = ParameterReadiness.External;
                        normalizedCount = 0;
                    }
                }
                string id = localSlot.StableId == "$"
                    ? entry.StableId
                    : entry.StableId + "/" + localSlot.StableId;
                ParameterStableId.Validate(id, nameof(localSlot.StableId));
                if (!identities.Add(id))
                    throw new InvalidOperationException(
                        $"Parameter manifest contains duplicate stable slot ID '{id}'.");

                var role = localSlot.Role == ParameterSlotRole.Trainable
                    ? entry.Role
                    : localSlot.Role;
                slots.Add(new ParameterSlotDescriptor(
                    id, role, normalizedReadiness, normalizedCount,
                    offsetKnown ? offset : (long?)null,
                    localSlot.Shape,
                    localSlot.ElementType,
                    role == localSlot.Role ? localSlot.UpdatePolicy : null,
                    role == localSlot.Role ? localSlot.Persistence : null,
                    role == localSlot.Role ? localSlot.Ownership : null,
                    entry.Availability,
                    materializedParameterCount: localSlot.MaterializedParameterCount));

                if (normalizedCount.HasValue)
                {
                    entryCount = checked(entryCount + normalizedCount.Value);
                    if (offsetKnown) offset = checked(offset + normalizedCount.Value);
                }
                else
                {
                    entryCountKnown = false;
                    offsetKnown = false;
                }
            }
            captured.Add(new CapturedEntry(entry, local,
                entryCountKnown ? entryCount : (long?)null));
        }

        var capturedEntries = captured.AsReadOnly();
        return new CapturedLayout(new ParameterLayoutSnapshot(slots), capturedEntries);
    }

    private static bool ReferencesSameSource(IParameterSource<T>? registered, IParameterSource<T> candidate)
        => ReferenceEquals(registered, candidate)
        || registered is ComponentAccessorParameterSource<T> accessor
           && ReferenceEquals(accessor.Current,
               candidate is ComponentAccessorParameterSource<T> candidateAccessor
                   ? candidateAccessor.Current
                   : candidate)
        || candidate is ComponentAccessorParameterSource<T> reverseAccessor
           && ReferenceEquals(reverseAccessor.Current, registered);

    private static bool TryGetNetwork(
        IParameterSource<T>? source,
        out NeuralNetworks.NeuralNetworkBase<T> network)
    {
        var current = source is ComponentAccessorParameterSource<T> accessor
            ? accessor.Current
            : source;
        if (current is NeuralNetworks.NeuralNetworkBase<T> found)
        {
            network = found;
            return true;
        }

        network = null!;
        return false;
    }

    private static int FindVariableEntryIndex(IReadOnlyList<CapturedEntry> entries)
    {
        int found = -1;
        for (int i = 0; i < entries.Count; i++)
        {
            if (entries[i].Entry.Source is not IVariableLengthParameterSource<T> variable) continue;
            // A replaceable vector stops being a variable restore target once materialized. It
            // remains in the marker family so its canonical order never changes across lifecycle
            // states, but it is now an ordinary exact-width component. More than one such fixed
            // vector is therefore harmless; more than one ACTIVE variable target is ambiguous.
            if (!variable.CanResizeOnRestore) continue;
            if (found >= 0)
                throw new InvalidOperationException(
                    "A parameter manifest may contain at most one resizable variable-length component.");
            found = i;
        }

        if (found >= 0 && found != entries.Count - 1)
            throw new InvalidOperationException(
                $"Variable-length parameter component '{entries[found].Entry.StableId}' must be " +
                "last in stable-ID order so its restore slice is unambiguous.");
        return found;
    }

    /// <summary>
    /// Materializes shape-resolved child sources before an operation that reads or writes concrete
    /// values. The manifest property itself remains allocation-free, so callers can still inspect
    /// readiness without paying for storage.
    /// </summary>
    private void MaterializeSources()
    {
        for (int i = 0; i < _entries.Count; i++)
        {
            if (_entries[i].Source is IParameterMaterializationSource materializer)
                materializer.MaterializeParameters();
        }
    }

    private List<Entry> OrderedEntries()
    {
        var ordered = new List<Entry>(_entries);
        ordered.Sort((left, right) =>
        {
            // A restore cannot know where a variable-width component ends unless it owns the
            // tail. Enforce that centrally instead of requiring every author and generator to
            // encode this mechanical rule into an otherwise semantic stable ID.
            bool leftVariable = left.Source is IVariableLengthParameterSource<T>;
            bool rightVariable = right.Source is IVariableLengthParameterSource<T>;
            if (leftVariable != rightVariable) return leftVariable ? 1 : -1;
            return CompareStableIds(left.StableId, right.StableId);
        });
        return ordered;
    }

    /// <summary>
    /// Orders stable IDs segment by segment, comparing all-digit segments NUMERICALLY.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Flat parameter order is this comparison, so it decides the layout of every checkpoint. A plain
    /// ordinal sort is lexicographic, which puts <c>layer/10</c> before <c>layer/2</c> -- so a model
    /// gaining its TENTH component silently reorders its whole parameter vector and every checkpoint
    /// written before that point restores into the wrong slots. Nothing would report an error; the
    /// lengths still match.
    /// </para>
    /// <para>
    /// Fixing it here rather than by zero-padding at the emitter is deliberate. Padding works only
    /// while every generator, every hand-written registration and every future emitter remembers to
    /// do it, and the failure mode for forgetting is silent checkpoint corruption. Ordering is the
    /// single choke point that all of them pass through.
    /// </para>
    /// <para>
    /// Existing data is unaffected: the legacy IDs are already <c>:D8</c> zero-padded, and numeric
    /// comparison of equal-width zero-padded digits gives the same order ordinal comparison did.
    /// </para>
    /// </remarks>
    internal static int CompareStableIds(string left, string right)
    {
        int li = 0, ri = 0;
        while (li < left.Length && ri < right.Length)
        {
            int lEnd = left.IndexOf('/', li);
            if (lEnd < 0) lEnd = left.Length;
            int rEnd = right.IndexOf('/', ri);
            if (rEnd < 0) rEnd = right.Length;

            int cmp = CompareSegment(left, li, lEnd, right, ri, rEnd);
            if (cmp != 0) return cmp;

            li = lEnd + 1;
            ri = rEnd + 1;
        }

        // A prefix sorts before the id that extends it: "layer" before "layer/0".
        return (left.Length - li).CompareTo(right.Length - ri);
    }

    private static int CompareSegment(string a, int aStart, int aEnd, string b, int bStart, int bEnd)
    {
        bool aNumeric = IsAllDigits(a, aStart, aEnd);
        bool bNumeric = IsAllDigits(b, bStart, bEnd);

        if (aNumeric && bNumeric)
        {
            // Compare by value, using length after skipping leading zeros so arbitrarily long
            // numbers work without parsing (and without overflowing).
            int aDigits = SkipLeadingZeros(a, aStart, aEnd);
            int bDigits = SkipLeadingZeros(b, bStart, bEnd);
            int aLen = aEnd - aDigits, bLen = bEnd - bDigits;
            if (aLen != bLen) return aLen.CompareTo(bLen);

            for (int i = 0; i < aLen; i++)
            {
                int d = a[aDigits + i].CompareTo(b[bDigits + i]);
                if (d != 0) return d;
            }

            // Equal in value: fall back to the raw text so the order stays total and deterministic
            // when two ids differ only by zero padding.
            return string.CompareOrdinal(a, aStart, b, bStart, Math.Max(aEnd - aStart, bEnd - bStart));
        }

        // A numeric segment sorts before an alphabetic one, so mixed sets stay deterministic.
        if (aNumeric != bNumeric) return aNumeric ? -1 : 1;

        int len = Math.Min(aEnd - aStart, bEnd - bStart);
        int ordinal = string.CompareOrdinal(a, aStart, b, bStart, len);
        return ordinal != 0 ? ordinal : (aEnd - aStart).CompareTo(bEnd - bStart);
    }

    private static bool IsAllDigits(string value, int start, int end)
    {
        if (start >= end) return false;
        for (int i = start; i < end; i++)
        {
            if (value[i] < '0' || value[i] > '9') return false;
        }

        return true;
    }

    private static int SkipLeadingZeros(string value, int start, int end)
    {
        int i = start;
        while (i < end - 1 && value[i] == '0') i++;
        return i;
    }
}
