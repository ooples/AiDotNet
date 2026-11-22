using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.DirectGpu;

namespace AiDotNet.Training.Memory;

/// <summary>
/// What a layer needs to know to stay correct inside a gradient-checkpointed segment, whose forward runs twice: once
/// without recording (its activations are discarded) and again during the backward pass, on the tape, to rebuild them.
/// </summary>
/// <remarks>
/// <para>
/// The recompute must reproduce the first forward exactly, or the gradients it produces belong to a different function.
/// A layer whose forward changes its own state (a routing bias nudged by the observed load, a step counter, a running
/// statistic) breaks that: the recompute sees the changed state, and applies the change a second time. PyTorch's
/// <c>torch.utils.checkpoint</c> has the same requirement and preserves only the random-number state for the user.
/// </para>
/// <para>
/// A stateful layer uses this class in two ways. It skips its state updates while <see cref="IsRecomputing"/> is true,
/// because the first forward already made them. And it reads the state its forward depends on through
/// <see cref="Preserve{T}"/> and <see cref="TryRecall{T}"/>: the first forward preserves the value it is about to use,
/// and the recompute recalls that value instead of the current one. Outside a checkpointed segment both calls do
/// nothing, so the plain training path is unchanged.
/// </para>
/// <para>
/// An auxiliary loss (an MoE load-balance term, say) needs nothing from the layer. The network collects whatever its
/// <c>ConsumeAuxiliaryTapeLoss</c> returns at the end of each segment as an extra output of that segment, and the
/// segment's backward recomputes it on the tape together with the segment output, so its gradient reaches the
/// segment's parameters as it does without checkpointing.
/// </para>
/// </remarks>
public static class GradientCheckpointingContext
{
    [ThreadStatic] private static SegmentState? t_segment;
    [ThreadStatic] private static bool t_recomputing;

    /// <summary>
    /// True while a checkpointed segment's first forward runs: its activations are discarded and the backward pass will
    /// run the same forward again (see <see cref="IsRecomputing"/>).
    /// </summary>
    public static bool IsSegmentForward => t_segment is not null && !t_recomputing;

    /// <summary>
    /// True while the backward pass re-runs a checkpointed segment's forward. The first forward already made every
    /// state update, so a layer must not make them again.
    /// </summary>
    public static bool IsRecomputing => t_segment is not null && t_recomputing;

    /// <summary>
    /// During a checkpointed segment's first forward, stores a copy of <paramref name="value"/> for
    /// <paramref name="owner"/>, which the segment's recompute returns from <see cref="TryRecall{T}"/>. Does nothing
    /// outside a first forward. Call it with the state the forward is about to read, before changing that state.
    /// </summary>
    /// <param name="owner">The object the state belongs to, compared by reference (usually the layer itself).</param>
    /// <param name="value">The state to keep. It is copied, so changing it in place afterwards does not affect the copy.</param>
    /// <remarks>
    /// The values an owner preserves in one forward are recalled in the same order, so a layer that runs more than
    /// once per segment (a shared block) gets each call's own value back.
    /// </remarks>
    public static void Preserve<T>(object owner, Tensor<T> value)
    {
        if (owner is null) throw new ArgumentNullException(nameof(owner));
        if (value is null) throw new ArgumentNullException(nameof(value));
        var segment = t_segment;
        if (segment is null || t_recomputing) return;
        segment.Preserved.Add(new PreservedValue(owner, CopyOutOfSegment(value)));
    }

    /// <summary>
    /// During a checkpointed segment's recompute, returns the next value <paramref name="owner"/> preserved in the
    /// segment's first forward. Returns false outside a recompute.
    /// </summary>
    /// <exception cref="InvalidOperationException">
    /// The recompute asked for more values than the first forward preserved, or for a different type: the two forwards
    /// diverged, and the gradients would not belong to the forward that produced the loss.
    /// </exception>
    public static bool TryRecall<T>(object owner, out Tensor<T> value)
    {
        if (owner is null) throw new ArgumentNullException(nameof(owner));
        var segment = t_segment;
        if (segment is null || !t_recomputing)
        {
            value = null!;
            return false;
        }

        var preserved = segment.Preserved;
        for (int i = 0; i < preserved.Count; i++)
        {
            var entry = preserved[i];
            if (entry.Recalled || !ReferenceEquals(entry.Owner, owner)) continue;
            if (entry.Value is not Tensor<T> typed)
                throw new InvalidOperationException(
                    $"{owner.GetType().Name} preserved a {entry.Value.GetType().Name} in the checkpointed forward but "
                    + $"recalled a Tensor<{typeof(T).Name}> in the recompute.");
            entry.Recalled = true;
            value = typed;
            return true;
        }

        throw new InvalidOperationException(
            $"{owner.GetType().Name} recalled more preserved values in a checkpoint recompute than it preserved in the "
            + "segment's first forward: the recompute took a different path, so its gradients would not match the loss.");
    }

    /// <summary>Marks the calling thread as running a segment's first forward until disposed.</summary>
    internal static Scope EnterSegmentForward(SegmentState segment) => new(segment, recomputing: false);

    /// <summary>Marks the calling thread as re-running a segment's forward until disposed.</summary>
    internal static Scope EnterRecompute(SegmentState segment) => new(segment, recomputing: true);

    // A segment's forward runs inside a TensorScope and (on the CPU) a TensorArena that both release their tensors when
    // the forward ends, so the copy is made outside both: device to device on the GPU, a fresh array on the CPU.
    private static Tensor<T> CopyOutOfSegment<T>(Tensor<T> value)
    {
        var engine = AiDotNetEngine.Current;
        if (engine is DirectGpuTensorEngine && value.TryGetGpuBuffer() is not null)
        {
            using (GradientTape<T>.NoGrad())
            {
                var copy = engine.StopGradient(value.IsContiguous ? value : value.Contiguous());
                return TensorScope.Detach(copy);
            }
        }
        var contiguous = value.IsContiguous ? value : value.Contiguous();
        return new Tensor<T>(contiguous.AsSpan().ToArray(), value.Shape.ToArray());
    }

    /// <summary>The values one checkpointed segment's first forward preserved for its recompute.</summary>
    internal sealed class SegmentState
    {
        internal List<PreservedValue> Preserved { get; } = new();

        /// <summary>Releases the preserved copies once the recompute that needed them is done.</summary>
        internal void Release()
        {
            foreach (var entry in Preserved)
                (entry.Value as IDisposable)?.Dispose();
            Preserved.Clear();
        }
    }

    internal sealed class PreservedValue
    {
        internal PreservedValue(object owner, object value)
        {
            Owner = owner;
            Value = value;
        }

        internal object Owner { get; }
        internal object Value { get; }
        internal bool Recalled { get; set; }
    }

    /// <summary>Restores the previous checkpoint state of the thread when disposed.</summary>
    internal readonly struct Scope : IDisposable
    {
        private readonly SegmentState? _previousSegment;
        private readonly bool _previousRecomputing;

        internal Scope(SegmentState segment, bool recomputing)
        {
            _previousSegment = t_segment;
            _previousRecomputing = t_recomputing;
            t_segment = segment;
            t_recomputing = recomputing;
        }

        public void Dispose()
        {
            t_segment = _previousSegment;
            t_recomputing = _previousRecomputing;
        }
    }
}
