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

        var ordered = OrderedEntries();
        int offset = 0;
        for (int i = 0; i < ordered.Count; i++)
        {
            var source = ordered[i].Source;
            if (source is null) continue;
            int count = checked((int)source.ParameterCount);
            var slice = new Vector<T>(count);
            for (int j = 0; j < count; j++) slice[j] = parameters[offset++];
            source.SetParameters(slice);
        }
    }

    private List<Entry> OrderedEntries()
    {
        var ordered = new List<Entry>(_entries);
        ordered.Sort((left, right) => StringComparer.Ordinal.Compare(left.StableId, right.StableId));
        return ordered;
    }
}
