using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

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
        public Entry(string stableId, IParameterSource<T>? source, ParameterSlotRole role)
        {
            StableId = stableId;
            Source = source;
            Role = role;
        }

        public string StableId { get; }
        public IParameterSource<T>? Source { get; }
        public ParameterSlotRole Role { get; }
    }

    private readonly List<Entry> _entries = new();
    private int _legacyId;

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
    /// Registers a legacy component. Its generated numeric ID preserves historical registration
    /// order; generated code should use the stable-ID overload.
    /// </summary>
    public void Register(IParameterSource<T>? component)
        => Register($"legacy/{_legacyId++:D8}", component, ParameterSlotRole.Trainable);

    /// <summary>Registers a component by durable identity and semantic role.</summary>
    public void Register(string stableId, IParameterSource<T>? component,
                         ParameterSlotRole role = ParameterSlotRole.Trainable)
    {
        if (string.IsNullOrWhiteSpace(stableId))
            throw new ArgumentException("A parameter component requires a stable ID.", nameof(stableId));

        for (int i = 0; i < _entries.Count; i++)
        {
            if (!string.Equals(_entries[i].StableId, stableId, StringComparison.Ordinal)) continue;
            if (ReferenceEquals(_entries[i].Source, component) && _entries[i].Role == role) return;
            throw new InvalidOperationException(
                $"Parameter component ID '{stableId}' was registered more than once. Stable IDs " +
                "must identify exactly one owner; use an Alias role rather than duplicate storage.");
        }

        _entries.Add(new Entry(stableId, component, role));
    }

    /// <inheritdoc />
    public ParameterLayoutSnapshot ParameterLayout
    {
        get
        {
            var ordered = OrderedEntries();
            var slots = new List<ParameterSlotDescriptor>();
            long offset = 0;
            bool offsetKnown = true;

            for (int i = 0; i < ordered.Count; i++)
            {
                var entry = ordered[i];
                if (entry.Source is null)
                {
                    slots.Add(new ParameterSlotDescriptor(
                        entry.StableId, entry.Role, ParameterReadiness.ShapeDeferred, null,
                        offsetKnown ? offset : (long?)null));
                    offsetKnown = false;
                    continue;
                }

                var local = entry.Source is IParameterLayoutSource layoutSource
                    ? layoutSource.GetParameterLayout()
                    : new[]
                    {
                        new ParameterSlotDescriptor(
                            "$", entry.Role,
                            entry.Source.ParameterCount == 0
                                ? ParameterReadiness.ParameterFree
                                : ParameterReadiness.Materialized,
                            entry.Source.ParameterCount)
                    };

                for (int j = 0; j < local.Count; j++)
                {
                    var item = local[j];
                    string id = item.StableId == "$"
                        ? entry.StableId
                        : entry.StableId + "/" + item.StableId;
                    var role = item.Role == ParameterSlotRole.Trainable ? entry.Role : item.Role;
                    slots.Add(new ParameterSlotDescriptor(
                        id, role, item.Readiness, item.ParameterCount,
                        offsetKnown ? offset : (long?)null));
                    if (item.ParameterCount.HasValue && offsetKnown)
                        offset = checked(offset + item.ParameterCount.Value);
                    else
                        offsetKnown = false;
                }
            }

            return new ParameterLayoutSnapshot(slots);
        }
    }

    /// <summary>
    /// The exact resolved count. An unresolved layout throws rather than masquerading as a partial
    /// or zero-sized model; shape-aware callers can inspect <see cref="ParameterLayout"/> first.
    /// </summary>
    public long ParameterCount
    {
        get
        {
            var layout = ParameterLayout;
            if (layout.ParameterCount.HasValue) return layout.ParameterCount.Value;
            throw new ParameterLayoutNotReadyException("count", layout);
        }
    }

    /// <summary>Concatenates the same stable-ID ordered entries described by the manifest.</summary>
    public Vector<T> GetParameters()
    {
        var layout = ParameterLayout;
        if (!layout.ParameterCount.HasValue)
            throw new ParameterLayoutNotReadyException("read", layout);

        var ordered = OrderedEntries();
        var parts = new List<Vector<T>>(ordered.Count);
        int total = 0;
        for (int i = 0; i < ordered.Count; i++)
        {
            if (ordered[i].Source is null) continue;
            var part = ordered[i].Source!.GetParameters();
            parts.Add(part);
            total = checked(total + part.Length);
        }

        var result = new Vector<T>(total);
        int offset = 0;
        for (int i = 0; i < parts.Count; i++)
        {
            for (int j = 0; j < parts[i].Length; j++) result[offset++] = parts[i][j];
        }
        return result;
    }

    /// <summary>
    /// Yields the registered state in stable-ID order. Tensor-backed child sources keep their
    /// zero-copy chunks; classical sources receive one exact flat payload chunk.
    /// </summary>
    public IEnumerable<ParameterChunk<T>> GetParameterStateChunks()
    {
        var ordered = OrderedEntries();
        for (int i = 0; i < ordered.Count; i++)
        {
            var entry = ordered[i];
            var source = entry.Source;
            if (source is null) continue;

            if (source is IParameterChunkSource<T> chunkSource)
            {
                foreach (var chunk in chunkSource.GetParameterStateChunks())
                {
                    if (chunk is null || chunk.Tensor.Length == 0) continue;
                    var role = entry.Role == ParameterSlotRole.Trainable
                        ? chunk.Role
                        : entry.Role;
                    string localId = chunk.StableId == "$"
                        ? entry.StableId
                        : entry.StableId + "/" + chunk.StableId;
                    yield return new ParameterChunk<T>(localId, role, chunk.Tensor);
                }
                continue;
            }

            var flat = source.GetParameters();
            if (flat.Length == 0) continue;

            if (source is IParameterLayoutSource layoutSource)
            {
                var layout = layoutSource.GetParameterLayout();
                int offset = 0;
                for (int j = 0; j < layout.Count; j++)
                {
                    var slot = layout[j];
                    if (!slot.ParameterCount.HasValue)
                        throw new ParameterLayoutNotReadyException(
                            "enumerate chunks", new ParameterLayoutSnapshot(layout));
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

        var layout = ParameterLayout;
        if (!layout.ParameterCount.HasValue)
            throw new ParameterLayoutNotReadyException("restore", layout);
        if (parameters.Length != layout.ParameterCount.Value)
            throw new ArgumentException(
                $"Expected {layout.ParameterCount.Value} parameters, got {parameters.Length}.",
                nameof(parameters));

        var spans = DeclaredSpans();
        int offset = 0;
        for (int i = 0; i < spans.Count; i++)
        {
            var source = spans[i].Key.Source;
            if (source is null) continue;

            // The declared span, never source.ParameterCount. See DeclaredSpans for why.
            long? declared = spans[i].Value;
            if (!declared.HasValue) throw new ParameterLayoutNotReadyException("restore", layout);

            int count = checked((int)declared.Value);
            var slice = new Vector<T>(count);
            for (int j = 0; j < count; j++) slice[j] = parameters[offset++];
            source.SetParameters(slice);
        }

        // The spans are what the vector was validated against, so a shortfall here means a source
        // described a different span than the snapshot did. Report it rather than leaving a
        // partially-restored model that predicts plausibly and wrongly.
        if (offset != parameters.Length)
            throw new InvalidOperationException(
                $"Restore consumed {offset} of {parameters.Length} values. A component's declared " +
                "layout disagrees with the manifest snapshot the vector was validated against.");
    }

    /// <summary>
    /// The declared span of each ordered entry, computed from the same local layout the manifest
    /// snapshot is built from.
    /// </summary>
    /// <remarks>
    /// Restore must slice by this rather than by <c>source.ParameterCount</c>. The two can
    /// disagree — a source that implements <see cref="IParameterLayoutSource"/> reports its span
    /// through the layout, and while a shape is deferred that layout is authoritative while the
    /// scalar count is not. Slicing by the source advances the offset by a different amount than
    /// the total the vector was just validated against, so every later component silently receives
    /// another component's values: the model restores without error and predicts differently. That
    /// is the round-trip defect this walk exists to prevent.
    /// </remarks>
    private List<KeyValuePair<Entry, long?>> DeclaredSpans()
    {
        var ordered = OrderedEntries();
        var spans = new List<KeyValuePair<Entry, long?>>(ordered.Count);
        for (int i = 0; i < ordered.Count; i++)
        {
            var entry = ordered[i];
            if (entry.Source is null)
            {
                spans.Add(new KeyValuePair<Entry, long?>(entry, null));
                continue;
            }

            if (entry.Source is IParameterLayoutSource layoutSource)
            {
                long total = 0;
                bool known = true;
                var local = layoutSource.GetParameterLayout();
                for (int j = 0; j < local.Count; j++)
                {
                    // Hoisted: indexing twice loses the HasValue proof, since the indexer is not
                    // guaranteed to return the same instance on a second call.
                    var slot = local[j];
                    long? slotCount = slot.ParameterCount;
                    if (!slotCount.HasValue) { known = false; break; }
                    total = checked(total + slotCount.Value);
                }
                spans.Add(new KeyValuePair<Entry, long?>(entry, known ? total : (long?)null));
                continue;
            }

            spans.Add(new KeyValuePair<Entry, long?>(entry, entry.Source.ParameterCount));
        }
        return spans;
    }

    private List<Entry> OrderedEntries()
    {
        var ordered = new List<Entry>(_entries);
        ordered.Sort((left, right) => StringComparer.Ordinal.Compare(left.StableId, right.StableId));
        return ordered;
    }
}
