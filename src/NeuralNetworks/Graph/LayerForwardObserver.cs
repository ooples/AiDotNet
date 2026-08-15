using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Threading;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.NeuralNetworks.Graph;

/// <summary>
/// Records layer calls during one forward pass so a model's real dataflow can be recovered from it.
/// </summary>
/// <typeparam name="T">Numeric type.</typeparam>
/// <remarks>
/// <para>
/// THE POINT IS TO STOP ASKING MODELS TO DECLARE THEIR TOPOLOGY. A <c>List&lt;ILayer&gt;</c> cannot say
/// "these are two independent branches", so validation reading it as a chain pairs layers that never
/// meet — measured on every constructible model, that produced four reports and not one was a real
/// defect. The information was never missing though: every model's forward method states the topology
/// exactly, in imperative C#. It simply was not readable. Running the forward with an observer attached
/// makes it readable, and then no model has to repeat itself in a declaration that can drift.
/// </para>
/// <para>
/// EDGES COME FROM TENSOR IDENTITY, not from ordering or naming. If layer B's input is reference-equal to
/// layer A's output then A feeds B — that is a fact about what ran, not an inference from a list. It
/// follows branches, skips, and reuse without being told any of them exist.
/// </para>
/// <para>
/// A TRACE IS EVIDENCE, NOT PROOF, and the limit is the same one <c>ShapeRelationDiscovery</c> carries: a
/// trace describes the input it was taken with. A model whose forward branches on data can do something
/// else next time. That is why the recovered graph is checked against a model's declared one where both
/// exist, rather than silently replacing it.
/// </para>
/// </remarks>
internal sealed class LayerForwardObserver<T> : IDisposable
{
    private static readonly AsyncLocal<LayerForwardObserver<T>?> s_current = new();

    private readonly List<(ILayer<T> Layer, Tensor<T> Input, Tensor<T> Output)> _calls = new();
    private readonly LayerForwardObserver<T>? _previous;
    private bool _disposed;

    /// <summary>The observer recording on this async context, if any.</summary>
    /// <remarks>
    /// <c>AsyncLocal</c> rather than a plain static so concurrent traces on different threads cannot
    /// interleave into one another's recordings — the test suite runs fixtures in parallel by default.
    /// </remarks>
    public static LayerForwardObserver<T>? Current => s_active == 0 ? null : s_current.Value;

    /// <summary>
    /// How many observers exist process-wide. Read INSTEAD of the AsyncLocal on the common path.
    /// </summary>
    /// <remarks>
    /// This is a hot-path guard, not bookkeeping. <c>LayerBase.Forward</c> consults <see cref="Current"/>
    /// on EVERY layer call in the library, and <c>AsyncLocal&lt;T&gt;.Value</c> costs roughly 20-50ns
    /// against ~1ns for a static int - a cost paid per layer, per step, per model, in production as well
    /// as in tests. Measured consequence of getting this wrong: with the AsyncLocal read unconditional,
    /// a full-suite run went from a handful of timeouts to 315, purely from the added per-forward cost
    /// under load.
    ///
    /// A plain int is sufficient because it only ever gates whether the AsyncLocal is worth reading. When
    /// it is zero no observer exists anywhere, so the answer is null without further work; when it is
    /// non-zero the AsyncLocal decides whether one exists on THIS context. A stale read can only cost one
    /// unnecessary AsyncLocal lookup, never a missed recording, because the count is incremented before
    /// the observer becomes reachable.
    /// </remarks>
    private static int s_active;

    /// <summary>Begins recording, restoring whatever was recording before on dispose.</summary>
    public LayerForwardObserver()
    {
        Interlocked.Increment(ref s_active);
        _previous = s_current.Value;
        s_current.Value = this;
    }

    /// <summary>Every layer call seen, in execution order.</summary>
    public IReadOnlyList<(ILayer<T> Layer, Tensor<T> Input, Tensor<T> Output)> Calls => _calls;

    /// <summary>
    /// Validates the actual producer-consumer edge before the consumer kernel executes.
    /// Tensor identity makes this branch-safe: unrelated neighbors in a flattened layer registry
    /// are never compared.
    /// </summary>
    public void ValidateInput(ILayer<T> layer, Tensor<T> input, string? portName)
    {
        if (_disposed || layer is null || input is null) return;

        for (int i = _calls.Count - 1; i >= 0; i--)
        {
            if (!ReferenceEquals(_calls[i].Output, input)) continue;
            LayerContractValidator.EnsureValueDomainCompatible(_calls[i].Layer, layer, portName);
            return;
        }

        _previous?.ValidateInput(layer, input, portName);
    }

    /// <summary>Records one layer call. Called by <c>LayerBase.Forward</c>; not for direct use.</summary>
    /// <remarks>
    /// FORWARDS TO THE ENCLOSING OBSERVER TOO, so nesting composes instead of stealing. Only one observer
    /// can be Current, so an inner trace would otherwise silently blind an outer one - and the outer
    /// caller has no way to detect that its trace came back empty because someone nested inside it. This
    /// is not hypothetical: the base's own one-shot chain validation traces the first forward, which
    /// swallowed the trace of any caller who was already observing.
    /// </remarks>
    public void Record(ILayer<T> layer, Tensor<T> input, Tensor<T> output)
    {
        if (_disposed || layer is null || input is null || output is null) return;
        _calls.Add((layer, input, output));
        _previous?.Record(layer, input, output);
    }

    /// <summary>
    /// Recovers the dataflow edges: for each recorded call, which earlier call produced its input.
    /// </summary>
    /// <returns>
    /// One entry per recorded call, giving the index of the call that produced its input, or -1 when the
    /// input came from outside the traced layers (the model's own input, or a tensor some non-layer step
    /// such as a reshape or concat produced).
    /// </returns>
    /// <remarks>
    /// <para>
    /// Scans BACKWARDS for the most recent producer, which matters when a tensor is reused: the nearest
    /// preceding producer is the one that actually fed this call.
    /// </para>
    /// <para>
    /// A -1 is informative rather than a gap. It marks exactly where the dataflow left the layer list —
    /// a branch entry point, or an edge transform like ABCNet's permute+reshape between its last
    /// recognition convolution and its dense head. Those are the boundaries a chain validator must not
    /// read across, and this is how it learns where they are without being told.
    /// </para>
    /// </remarks>
    public IReadOnlyList<int> ResolveProducers()
    {
        var producers = new int[_calls.Count];
        for (int i = 0; i < _calls.Count; i++)
        {
            producers[i] = -1;
            for (int j = i - 1; j >= 0; j--)
            {
                if (ReferenceEquals(_calls[j].Output, _calls[i].Input))
                {
                    producers[i] = j;
                    break;
                }
            }
        }

        return producers;
    }

    /// <summary>
    /// The maximal runs of calls that are genuinely sequential — each consuming the previous one's output
    /// exclusively.
    /// </summary>
    /// <remarks>
    /// The traced counterpart of <c>LayerGraph.ContiguousRuns</c>, and what chain validation should walk.
    /// A run breaks wherever a call's input did not come from its immediate predecessor, which is exactly
    /// where two layers stop meeting — at a branch, at a rejoin, or across a transform that is not a layer.
    /// </remarks>
    public IReadOnlyList<IReadOnlyList<ILayer<T>>> ContiguousRuns()
    {
        var producers = ResolveProducers();
        var consumerCount = new int[_calls.Count];
        foreach (int p in producers)
        {
            if (p >= 0) consumerCount[p]++;
        }

        var runs = new List<IReadOnlyList<ILayer<T>>>();
        var current = new List<ILayer<T>>();

        for (int i = 0; i < _calls.Count; i++)
        {
            bool continues =
                current.Count > 0
                && producers[i] == i - 1
                && consumerCount[i - 1] == 1;

            if (!continues)
            {
                if (current.Count > 1) runs.Add(current);
                current = new List<ILayer<T>>();
            }

            current.Add(_calls[i].Layer);
        }

        if (current.Count > 1) runs.Add(current);
        return runs;
    }

    /// <summary>True when some tensor fed more than one call — the model branches.</summary>
    public bool HasFanOut
    {
        get
        {
            var producers = ResolveProducers();
            var seen = new HashSet<int>();
            foreach (int p in producers)
            {
                if (p >= 0 && !seen.Add(p)) return true;
            }

            return false;
        }
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        s_current.Value = _previous;
        Interlocked.Decrement(ref s_active);
    }
}
