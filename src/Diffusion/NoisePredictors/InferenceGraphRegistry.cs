using System;
using System.Collections.Generic;
using System.Threading;

namespace AiDotNet.Diffusion.NoisePredictors;

/// <summary>
/// A predictor that keeps a captured CUDA inference graph (and the engine-wide resident capture scope that graph
/// needs) alive between calls.
/// </summary>
internal interface IInferenceGraphOwner
{
    /// <summary>Destroys the captured graph and ends its capture scope; the next inference captures afresh.</summary>
    void ReleaseInferenceGraph();
}

/// <summary>
/// Process-wide registry of live captured inference graphs. A graph's capture scope changes engine-wide state
/// (eviction suspended, inference-capture binding, weight-cache invalidation deferred while a graph lives), and a
/// replay reads the weights it was captured against, so every graph is released before any training step runs:
/// <see cref="DiffusionModelBase{T}.Train"/> and the fused training session call <see cref="ReleaseAll"/>.
/// </summary>
internal static class InferenceGraphRegistry
{
    private static readonly object Gate = new();
    private static readonly List<WeakReference<IInferenceGraphOwner>> Owners = new();
    private static long _generation;

    /// <summary>Bumped by every <see cref="ReleaseAll"/>; part of the key a graph was captured under.</summary>
    internal static long Generation => Interlocked.Read(ref _generation);

    /// <summary>
    /// The process-level switch: on by default (a CUDA engine that supports capture still decides per call);
    /// <c>AIDOTNET_DIFFUSION_CUDA_GRAPH=0</c> opts out. Read live so a harness can flip it after startup.
    /// </summary>
    internal static bool Enabled => Environment.GetEnvironmentVariable("AIDOTNET_DIFFUSION_CUDA_GRAPH") != "0";

    internal static void Register(IInferenceGraphOwner owner)
    {
        lock (Gate)
        {
            foreach (var reference in Owners)
                if (reference.TryGetTarget(out var existing) && ReferenceEquals(existing, owner)) return;
            Owners.Add(new WeakReference<IInferenceGraphOwner>(owner));
        }
    }

    /// <summary>Releases every live graph (weights are about to change, or training is about to run).</summary>
    internal static void ReleaseAll()
    {
        Interlocked.Increment(ref _generation);
        List<IInferenceGraphOwner>? live = null;
        lock (Gate)
        {
            for (int i = Owners.Count - 1; i >= 0; i--)
            {
                if (Owners[i].TryGetTarget(out var owner)) (live ??= new List<IInferenceGraphOwner>()).Add(owner);
                else Owners.RemoveAt(i);
            }
        }
        if (live is null) return;
        foreach (var owner in live) owner.ReleaseInferenceGraph();
    }
}
