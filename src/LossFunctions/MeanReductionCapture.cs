namespace AiDotNet.LossFunctions;

/// <summary>
/// The mean reduction a training objective's loss divided by, recorded while the objective was scored: the loss and
/// the target layout it scored. Distributed training weights each rank's gradient by the count this reduction divides
/// by, so the count must come from the loss that formed the gradient, not from a model's default loss.
/// </summary>
internal sealed class MeanReductionRecord<T>
{
    internal MeanReductionRecord(LossFunctionBase<T> loss, object layout)
    {
        Loss = loss;
        Layout = layout;
    }

    /// <summary>The loss that scored the objective.</summary>
    internal LossFunctionBase<T> Loss { get; }

    /// <summary>The layout it scored, as that loss records it (see <see cref="LossFunctionBase{T}.CountMeanReductionElements(Tensor{T}, object?)"/>).</summary>
    internal object Layout { get; }
}

/// <summary>
/// Records, for the duration of a scope on the current thread, the first mean reduction a loss reports while it scores
/// an objective (a loss whose mean leaves some targets out, such as cross-entropy's ignored rows). A compiled step's
/// trace runs inside a scope, so the plan keeps the reduction its objective traced, and a replay counts with it even
/// when the same loss instance has since scored something else.
/// </summary>
internal static class MeanReductionCapture<T>
{
    [System.ThreadStatic] private static Scope? t_current;

    /// <summary>Starts recording on this thread until the returned scope is disposed (scopes nest).</summary>
    internal static Scope Begin() => t_current = new Scope(t_current);

    /// <summary>Called by a loss as it scores a target: the innermost open scope keeps the first report.</summary>
    internal static void Record(LossFunctionBase<T> loss, object layout)
    {
        if (t_current is { Captured: null } scope)
            scope.Captured = new MeanReductionRecord<T>(loss, layout);
    }

    internal sealed class Scope : System.IDisposable
    {
        private readonly Scope? _outer;
        private bool _disposed;

        internal Scope(Scope? outer) => _outer = outer;

        /// <summary>The first reduction reported in this scope, or null when no loss that reports one scored.</summary>
        internal MeanReductionRecord<T>? Captured { get; set; }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            if (ReferenceEquals(t_current, this)) t_current = _outer;
        }
    }
}
