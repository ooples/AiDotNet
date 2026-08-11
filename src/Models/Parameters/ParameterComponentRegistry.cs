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

        // Slice by the LAYOUT, not by re-asking each source how long it is.
        //
        // The vector was validated against layout.ParameterCount immediately above, so the layout is
        // the only description of it that is known to be consistent. Re-deriving each slice from
        // source.ParameterCount asks a second, independent question -- and the two answers diverge
        // in exactly the case this registry exists to handle: a source whose declared slots and
        // whose live count disagree, i.e. anything lazily materialized. When they diverge every
        // slice after the first mismatch is shifted, so weights land in the wrong component and the
        // restore reports success. That is the clone/serialize round-trip losing weights.
        //
        // A source may contribute several slots (IParameterLayoutSource), and the layout emits them
        // consecutively under "<entryId>/<slotId>", so its own span is the run of slots carrying its
        // id. Summing the DECLARED counts over that run gives the length the vector was measured
        // with, which is the only length that can be correct here.
        var ordered = OrderedEntries();
        var slots = layout.Slots;
        int slotIndex = 0;
        int offset = 0;

        for (int i = 0; i < ordered.Count; i++)
        {
            var entry = ordered[i];

            long declared = 0;
            while (slotIndex < slots.Count && OwnsSlot(entry.StableId, slots[slotIndex].StableId))
            {
                // Non-null by construction: a slot with no count makes layout.ParameterCount null,
                // and the guard above already rejected that.
                declared += slots[slotIndex].ParameterCount ?? 0;
                slotIndex++;
            }

            var source = entry.Source;
            if (source is null) continue;

            int count = checked((int)declared);
            var slice = new Vector<T>(count);
            for (int j = 0; j < count; j++) slice[j] = parameters[offset++];
            source.SetParameters(slice);
        }
    }

    /// <summary>
    /// True when <paramref name="slotId"/> is the entry's own slot or one it contributed, matching
    /// the "&lt;entryId&gt;/&lt;slotId&gt;" composition the layout builder uses.
    /// </summary>
    private static bool OwnsSlot(string entryId, string slotId)
        => string.Equals(slotId, entryId, StringComparison.Ordinal)
           || (slotId.Length > entryId.Length
               && slotId[entryId.Length] == '/'
               && slotId.StartsWith(entryId, StringComparison.Ordinal));

    private List<Entry> OrderedEntries()
    {
        var ordered = new List<Entry>(_entries);
        ordered.Sort((left, right) => CompareStableIds(left.StableId, right.StableId));
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
