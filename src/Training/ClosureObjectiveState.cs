using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Training;

/// <summary>Why a closure-objective step could not replay a cached compiled plan.</summary>
internal enum ClosureFallbackReason
{
    /// <summary>The objective read one of its own values on the host while probed.</summary>
    Aborted,

    /// <summary>The objective captured state whose value cannot be compared.</summary>
    Unsignable,

    /// <summary>The objective's structure changed on too many of its first steps to be worth caching.</summary>
    Dynamic,
}

/// <summary>
/// The compiled plans one owner's closure objective (a <c>Func&lt;Tensor&lt;T&gt;&gt;</c> loss that builds its per-step
/// data inside itself) has replayed, keyed by <see cref="ObjectiveSignature"/>, and the rule that stops caching a
/// genuinely dynamic objective.
/// </summary>
/// <remarks>
/// <para>At most <see cref="MaxPlans"/> plans are kept, least recently used evicted first. A signature change (the
/// short last batch of an epoch) compiles a second plan that then continues the first one's optimizer; switching
/// back replays the cached one.</para>
/// <para>Dynamic rule: an objective whose probe aborts, is unsignable, or misses the cache on more than
/// <see cref="MaxEarlyFailures"/> of its first <see cref="ObservationSteps"/> steps stops probing and caching. Its
/// steps still run as compiled plans traced from that step (never a stale replay, and the optimizer's moments carry
/// over), at the price of a compile per step; the one diagnostic says so.</para>
/// </remarks>
internal sealed class ClosureObjectiveState<T>
{
    internal const int MaxPlans = 4;
    internal const int ObservationSteps = 8;
    internal const int MaxEarlyFailures = 2;

    private readonly List<Entry> _entries = new();   // most recently used first

    internal sealed class Entry
    {
        // The tensors the plan reads that its trace did not produce, in the guard's order. They are the tensors of the
        // step the plan was traced from, so the caller may still own them (a pre-built batch it reuses every epoch).
        private readonly object[] _leaves;
        private Dictionary<object, int>? _leafIndex;
        private object?[]? _saved;
        private bool[]? _overwritten;

        internal Entry(ICompiledTrainingPlan<T> plan, ClosureObjectiveGuard<T>? guard, object[]? leaves = null)
        {
            Plan = plan;
            _leaves = leaves ?? Array.Empty<object>();
            // Without the plan's own leaves a replay could not give the caller's tensors back, so it does not replay.
            Guard = guard is not null && leaves is not null && guard.LeafCount == leaves.Length ? guard : null;
        }

        internal ICompiledTrainingPlan<T> Plan { get; }

        /// <summary>Null for a plan traced from one step that cannot be replayed (kept only until evicted).</summary>
        internal ClosureObjectiveGuard<T>? Guard { get; }

        /// <summary>
        /// Copies this step's leaves (<paramref name="probedLeaves"/>, from a probe whose signature equals the plan's)
        /// into the plan's leaves, after saving what each overwritten leaf held: <see cref="RestoreLeaves"/> puts it
        /// back once the step has run.
        /// </summary>
        /// <remarks>
        /// The plan's leaves are the tensors of the step it was traced from. Left overwritten, a caller that reuses
        /// its batches finds the first one holding the last step's data, and trains on that when the batch comes
        /// round again (the same tensor is then not copied at all).
        /// </remarks>
        /// <returns>False, with nothing changed, when the plan cannot take this step's data; the step then compiles
        /// from its own trace.</returns>
        internal bool TryRebind(object[] probedLeaves, out string? reason)
        {
            if (probedLeaves is null) throw new ArgumentNullException(nameof(probedLeaves));
            if (Guard is null)
            {
                reason = "the plan is not replayable";
                return false;
            }

            if (probedLeaves.Length != _leaves.Length)
            {
                reason = $"the probe has {probedLeaves.Length} leaves, the plan {_leaves.Length}";
                return false;
            }

            if (_leafIndex is null)
            {
                _leafIndex = new Dictionary<object, int>(_leaves.Length, ReferenceComparer.Instance);
                for (int i = 0; i < _leaves.Length; i++)
                    _leafIndex[_leaves[i]] = i;
            }

            // A tensor the plan reads in one position and this step supplies in another (the caller swapped two
            // batches) cannot be copied in place: the first copy would overwrite the second copy's source.
            for (int i = 0; i < _leaves.Length; i++)
            {
                if (ReferenceEquals(_leaves[i], probedLeaves[i])) continue;
                if (_leafIndex.TryGetValue(probedLeaves[i], out int other))
                {
                    reason = $"leaf {i} of this step is the plan's leaf {other}";
                    return false;
                }
            }

            _saved ??= new object?[_leaves.Length];
            _overwritten ??= new bool[_leaves.Length];
            try
            {
                for (int i = 0; i < _leaves.Length; i++)
                {
                    if (ReferenceEquals(_leaves[i], probedLeaves[i])) continue;   // read live, never written
                    if (_leaves[i] is not ISignatureTensor leaf)
                    {
                        reason = $"leaf {i} is not a tensor";
                        return false;
                    }

                    var saved = (ISignatureTensor)(_saved[i] ??= Activator.CreateInstance(
                        _leaves[i].GetType(), new object[] { (int[])leaf.SignatureShape.Clone() })
                        ?? throw new InvalidOperationException($"leaf {i} ({_leaves[i].GetType().Name}) cannot be saved"));
                    saved.CopyLeafFrom(leaf);
                    _overwritten[i] = true;
                }
            }
            catch (Exception ex) when (ex is MissingMethodException || ex is MemberAccessException
                || ex is System.Reflection.TargetInvocationException || ex is InvalidOperationException
                || ex is ArgumentException || ex is ObjectDisposedException || ex is InvalidCastException)
            {
                Array.Clear(_overwritten, 0, _overwritten.Length);   // nothing has been written yet
                reason = $"a leaf of the plan could not be saved ({ex.GetType().Name}: {ex.Message})";
                return false;
            }

            if (!Guard.TryCopyLeaves(probedLeaves, out reason))
            {
                RestoreLeaves();
                return false;
            }

            return true;
        }

        /// <summary>Gives every leaf <see cref="TryRebind"/> overwrote its own contents back.</summary>
        internal void RestoreLeaves()
        {
            if (_overwritten is null || _saved is null) return;
            for (int i = 0; i < _overwritten.Length; i++)
            {
                if (!_overwritten[i]) continue;
                _overwritten[i] = false;
                if (_leaves[i] is ISignatureTensor leaf && _saved[i] is ISignatureTensor saved)
                {
                    try
                    {
                        leaf.CopyLeafFrom(saved);
                    }
                    catch (Exception ex) when (ex is ObjectDisposedException || ex is InvalidOperationException || ex is ArgumentException)
                    {
                        // The caller disposed the tensor during the step: there is nothing left to give back.
                    }
                }
            }
        }
    }

    private sealed class ReferenceComparer : IEqualityComparer<object>
    {
        internal static readonly ReferenceComparer Instance = new();
        public new bool Equals(object? x, object? y) => ReferenceEquals(x, y);
        public int GetHashCode(object obj) => System.Runtime.CompilerServices.RuntimeHelpers.GetHashCode(obj);
    }

    /// <summary>Steps this objective has taken through the closure path.</summary>
    internal int StepsSeen { get; private set; }

    /// <summary>Probe aborts, unsignable probes and cache misses within the first <see cref="ObservationSteps"/>.</summary>
    internal int EarlyFailures { get; private set; }

    /// <summary>True once the dynamic rule fired: no more probing or caching.</summary>
    internal bool IsDynamic { get; private set; }

    /// <summary>Set when the objective proved unsignable on its first step: it trains on the eager tape.</summary>
    internal string? UnsignableReason { get; private set; }

    /// <summary>The last reason a step did not replay a cached plan.</summary>
    internal string? LastReason { get; private set; }

    /// <summary>The fallback reasons already reported, so each is diagnosed once per owner.</summary>
    internal HashSet<ClosureFallbackReason> ReportedReasons { get; } = new();

    /// <summary>Plans currently held (replayable or not).</summary>
    internal int PlanCount => _entries.Count;

    /// <summary>Replayable plans currently held.</summary>
    internal int ReplayablePlanCount
    {
        get
        {
            int n = 0;
            foreach (var e in _entries) if (e.Guard is not null) n++;
            return n;
        }
    }

    /// <summary>The replayable plan compiled for <paramref name="signature"/>, made most recently used; or null.</summary>
    internal Entry? Find(ObjectiveSignature signature)
    {
        for (int i = 0; i < _entries.Count; i++)
        {
            var entry = _entries[i];
            if (entry.Guard is { } guard && guard.Signature.Equals(signature))
            {
                if (i > 0)
                {
                    _entries.RemoveAt(i);
                    _entries.Insert(0, entry);
                }
                return entry;
            }
        }
        return null;
    }

    /// <summary>
    /// Adds a plan as most recently used and disposes whatever falls past <see cref="MaxPlans"/>, never the plan
    /// <paramref name="keepAlive"/> (the configured plan the new one continues its optimizer from).
    /// </summary>
    internal void Add(Entry entry, object? keepAlive)
    {
        if (entry is null) throw new ArgumentNullException(nameof(entry));
        _entries.Insert(0, entry);
        for (int i = _entries.Count - 1; i >= 0 && _entries.Count > MaxPlans; i--)
        {
            var victim = _entries[i];
            if (ReferenceEquals(victim.Plan, keepAlive) || ReferenceEquals(victim, entry)) continue;
            _entries.RemoveAt(i);
            victim.Plan.Dispose();
        }
    }

    /// <summary>Drops a plan that can no longer take new data (the caller disposed a leaf it was compiled on, or
    /// supplies one of its leaves in another position).</summary>
    internal void Remove(Entry entry, object? keepAlive)
    {
        if (entry is null) throw new ArgumentNullException(nameof(entry));
        if (_entries.Remove(entry) && !ReferenceEquals(entry.Plan, keepAlive))
            entry.Plan.Dispose();
    }

    /// <summary>Counts one step; <paramref name="failed"/> when it could not replay a cached plan.</summary>
    internal void RecordStep(bool failed, string? reason)
    {
        StepsSeen++;
        if (failed)
        {
            LastReason = reason;
            if (StepsSeen <= ObservationSteps && ++EarlyFailures > MaxEarlyFailures)
                IsDynamic = true;
        }
    }

    internal void MarkUnsignable(string reason) => UnsignableReason = reason;

    /// <summary>Disposes every held plan except <paramref name="keepAlive"/>.</summary>
    internal void Clear(object? keepAlive)
    {
        foreach (var entry in _entries)
            if (!ReferenceEquals(entry.Plan, keepAlive))
                entry.Plan.Dispose();
        _entries.Clear();
    }
}
