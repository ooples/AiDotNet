namespace AiDotNet.Diffusion.NoisePredictors;

/// <summary>
/// Who holds the process-wide streaming session of <see cref="AiDotNet.Tensors.LinearAlgebra.WeightRegistry"/>
/// among noise predictors, of every element type.
/// </summary>
/// <remarks>
/// <para>
/// The registry is one per process: configuring or resetting it replaces the pool and the parameter file every
/// streamed predictor reads its weights from. Whether that is safe cannot be read off the registry. Parameter IO
/// unregisters each weight once it is mapped, so a predictor in the middle of a parameter walk has no registered
/// entry between two of its weights and looks exactly like an idle registry. A predictor therefore owns the session
/// from the moment it engages streaming until it is disposed or collected, and a second predictor may take the
/// registry only when no other owner is alive.
/// </para>
/// <para>
/// Owners are held weakly: a predictor that is dropped without Dispose must not keep the session, and the
/// parameter file behind it, for the life of the process.
/// </para>
/// </remarks>
internal static class WeightStreamingSessions
{
    private static readonly object Gate = new();
    private static readonly List<WeakReference<object>> Owners = new();
    private static long _version;

    /// <summary>Changes whenever an owner is added or removed.</summary>
    internal static long Version
    {
        get { lock (Gate) return _version; }
    }

    /// <summary>The owners that are still reachable.</summary>
    internal static int LiveOwnerCount
    {
        get
        {
            lock (Gate)
            {
                PruneCollectedOwners();
                return Owners.Count;
            }
        }
    }

    /// <summary>
    /// Gives <paramref name="owner"/> the session when no other live predictor holds it: under one gate, checks the
    /// owners, lets <paramref name="claim"/> veto, runs <paramref name="configure"/> and records the owner.
    /// </summary>
    /// <param name="owner">The predictor engaging streaming.</param>
    /// <param name="collectUnreachableOwners">
    /// Whether to run a blocking collection when another owner looks alive, so that one which is merely not yet
    /// collected does not keep the session. A foundation-scale engagement is rare and the alternative is tens of
    /// gigabytes held on disk and a predictor that runs fully resident; a caller that was refused by the same
    /// owner set before passes <c>false</c>.
    /// </param>
    /// <param name="claim">Further conditions, evaluated under the gate; <c>false</c> refuses.</param>
    /// <param name="configure">Reconfigures the registry for the new session, under the gate.</param>
    /// <returns><c>true</c> when <paramref name="owner"/> now holds the session.</returns>
    internal static bool TryBegin(object owner, bool collectUnreachableOwners, Func<bool> claim, Action configure)
    {
        if (owner is null) throw new ArgumentNullException(nameof(owner));
        if (claim is null) throw new ArgumentNullException(nameof(claim));
        if (configure is null) throw new ArgumentNullException(nameof(configure));

        bool otherOwnerSeen;
        lock (Gate)
        {
            otherOwnerSeen = HasOtherLiveOwner(owner);
        }

        if (otherOwnerSeen)
        {
            if (!collectUnreachableOwners) return false;
            // Outside the gate: a finalizer must be free to take it.
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();
        }

        lock (Gate)
        {
            if (HasOtherLiveOwner(owner)) return false;
            if (!claim()) return false;
            configure();
            if (!IsOwner(owner))
            {
                Owners.Add(new WeakReference<object>(owner));
                _version++;
            }

            return true;
        }
    }

    /// <summary>Gives up <paramref name="owner"/>'s session, if it holds one.</summary>
    internal static void End(object owner)
    {
        if (owner is null) return;
        lock (Gate)
        {
            for (int i = Owners.Count - 1; i >= 0; i--)
            {
                if (Owners[i].TryGetTarget(out var target) && ReferenceEquals(target, owner))
                {
                    Owners.RemoveAt(i);
                    _version++;
                }
            }
        }
    }

    /// <summary>Forgets every owner. For tests that reset the registry themselves.</summary>
    internal static void Clear()
    {
        lock (Gate)
        {
            if (Owners.Count == 0) return;
            Owners.Clear();
            _version++;
        }
    }

    private static bool HasOtherLiveOwner(object owner)
    {
        PruneCollectedOwners();
        foreach (var reference in Owners)
        {
            if (reference.TryGetTarget(out var target) && !ReferenceEquals(target, owner)) return true;
        }

        return false;
    }

    private static bool IsOwner(object owner)
    {
        foreach (var reference in Owners)
        {
            if (reference.TryGetTarget(out var target) && ReferenceEquals(target, owner)) return true;
        }

        return false;
    }

    private static void PruneCollectedOwners()
    {
        int before = Owners.Count;
        Owners.RemoveAll(static reference => !reference.TryGetTarget(out _));
        if (Owners.Count != before) _version++;
    }
}
