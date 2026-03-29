using System.Runtime.CompilerServices;
using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Autodiff;

/// <summary>
/// PyTorch-style gradient tape that records tensor operations during forward execution
/// and computes exact gradients via reverse-mode automatic differentiation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Operations performed on tensors while a tape is active are automatically recorded
/// with their backward functions. Calling <see cref="Gradient"/> performs reverse-mode
/// autodiff (backpropagation) through the recorded operations.
/// </para>
/// <para>
/// Uses <see cref="AsyncLocal{T}"/> for async pipeline safety — the active tape
/// propagates correctly across async/await boundaries.
/// </para>
/// <para><b>For Beginners:</b> This tracks math operations so gradients can be computed automatically.
///
/// <code>
/// using var tape = new GradientTape&lt;double&gt;();
/// var y = Engine.TensorMatMul(x, w);   // recorded
/// var loss = Engine.TensorMean(Engine.TensorMultiply(y, y)); // recorded
/// var grads = tape.Gradient(loss);      // backpropagation
/// var dw = grads[w];                    // gradient of loss w.r.t. w
/// </code>
/// </para>
/// </remarks>
public sealed class GradientTape<T> : IDisposable
{
    /// <summary>
    /// A single recorded operation on the tape.
    /// </summary>
    private readonly record struct TapeEntry(
        string OpName,
        Tensor<T>[] Inputs,
        Tensor<T> Output,
        Func<Tensor<T>, Tensor<T>[]> Backward);

    // ─── Thread/async-safe active tape ───────────────────────────────

    private static readonly AsyncLocal<Stack<GradientTape<T>>?> _tapeStack = new();

    /// <summary>
    /// Gets the currently active tape, or null if none is active.
    /// Uses AsyncLocal for correct propagation across async/await boundaries.
    /// </summary>
    public static GradientTape<T>? Current
    {
        get
        {
            var stack = _tapeStack.Value;
            return stack is { Count: > 0 } ? stack.Peek() : null;
        }
    }

    // ─── Instance state ──────────────────────────────────────────────

    private readonly List<TapeEntry> _ops = [];
    private readonly object _opsLock = new();
    private readonly HashSet<Tensor<T>> _watched = new(ReferenceEqualityComparer.Instance);
    private volatile bool _disposed;
    private volatile bool _used;

    /// <summary>
    /// If true, the tape can compute gradients multiple times.
    /// </summary>
    public bool Persistent { get; }

    /// <summary>
    /// Whether the tape is currently recording operations.
    /// </summary>
    public bool IsRecording { get; internal set; } = true;

    // ─── Construction / disposal ─────────────────────────────────────

    /// <summary>
    /// Creates a new gradient tape and pushes it onto the async-local tape stack.
    /// </summary>
    /// <param name="persistent">If true, the tape can compute gradients multiple times.</param>
    public GradientTape(bool persistent = false)
    {
        Persistent = persistent;
        _tapeStack.Value ??= new Stack<GradientTape<T>>();
        _tapeStack.Value.Push(this);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        IsRecording = false;

        var stack = _tapeStack.Value;
        if (stack is { Count: > 0 } && ReferenceEquals(stack.Peek(), this))
            stack.Pop();

        if (!Persistent)
        {
            lock (_opsLock)
            {
                _ops.Clear();
                _watched.Clear();
            }
        }
    }

    // ─── Public API ──────────────────────────────────────────────────

    /// <summary>
    /// Marks a tensor so its gradient will be computed by <see cref="Gradient"/>.
    /// Equivalent to PyTorch's <c>requires_grad=True</c>.
    /// </summary>
    public void Watch(Tensor<T> tensor)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(GradientTape<T>));
        lock (_opsLock) { _watched.Add(tensor); }
    }

    /// <summary>
    /// Marks multiple tensors for gradient computation.
    /// </summary>
    public void Watch(IEnumerable<Tensor<T>> tensors)
    {
        foreach (var t in tensors) Watch(t);
    }

    /// <summary>
    /// Backward-compatible overload: watches a ComputationNode's value tensor.
    /// </summary>
    public void Watch(ComputationNode<T> node)
    {
        node.RequiresGradient = true;
        Watch(node.Value);
        lock (_opsLock) { _watchedNodes_compat.Add(node); }
    }

    /// <summary>
    /// Backward-compatible overload: watches multiple ComputationNodes.
    /// </summary>
    public void Watch(IEnumerable<ComputationNode<T>> nodes)
    {
        foreach (var n in nodes) Watch(n);
    }

    /// <summary>
    /// Backward-compatible: records a ComputationNode as an operation.
    /// Bridge until ExportComputationGraph is removed (#1059).
    /// </summary>
    public void RecordOperation(ComputationNode<T> node)
    {
        // Old API — bridge from ComputationNode to tape.
        if (!IsRecording || NoGradScope<T>.IsSuppressed) return;
        lock (_opsLock) { _ops.Add(new TapeEntry(
            node.Name ?? "node",
            node.Parents.Select(p => p.Value).ToArray(),
            node.Value,
            grad =>
            {
                node.Gradient = grad;
                node.BackwardFunction?.Invoke(grad);
                return node.Parents.Select(p => p.Gradient ?? new Tensor<T>(p.Value.Shape.ToArray())).ToArray();
            })); }
    }

    /// <summary>
    /// Backward-compatible: computes gradients for ComputationNodes.
    /// </summary>
    public Dictionary<ComputationNode<T>, Tensor<T>> Gradient(
        ComputationNode<T> target,
        IEnumerable<ComputationNode<T>>? sources = null,
        bool createGraph = false)
    {
        if (createGraph)
        {
            throw new NotSupportedException(
                "createGraph is not supported for ComputationNode-based Gradient. " +
                "Use the Tensor<T> Gradient overload with the new tape API instead.");
        }

        // Use the ComputationNode's own backward (old path)
        target.Backward();

        var result = new Dictionary<ComputationNode<T>, Tensor<T>>();
        var sourceList = sources?.ToList() ?? _watchedNodes_compat;
        foreach (var src in sourceList)
        {
            if (src.Gradient is not null)
                result[src] = src.Gradient;
        }
        return result;
    }

    // Compat tracked nodes (only populated via Watch(ComputationNode))
    private readonly List<ComputationNode<T>> _watchedNodes_compat = [];

    /// <summary>
    /// Stops recording. Backward compat alias.
    /// </summary>
    public void StopRecording() => IsRecording = false;

    /// <summary>
    /// Resumes recording. Backward compat alias.
    /// </summary>
    public void ResumeRecording() => IsRecording = true;

    /// <summary>
    /// Records an operation with its backward function.
    /// Called by Engine ops when a tape is active.
    /// </summary>
    /// <param name="opName">Human-readable name for debugging (e.g., "MatMul", "Add").</param>
    /// <param name="inputs">Input tensors to the operation.</param>
    /// <param name="output">Output tensor produced by the operation.</param>
    /// <param name="backward">
    /// Given the gradient of the loss w.r.t. <paramref name="output"/>,
    /// returns gradients w.r.t. each element of <paramref name="inputs"/> (same order).
    /// </param>
    public void RecordOp(
        string opName,
        Tensor<T>[] inputs,
        Tensor<T> output,
        Func<Tensor<T>, Tensor<T>[]> backward)
    {
        if (!IsRecording || NoGradScope<T>.IsSuppressed) return;

        // Reject ops whose output aliases an input — the reverse pass keys
        // gradients by tensor identity, so aliasing would corrupt accumulation.
        foreach (var input in inputs)
        {
            if (ReferenceEquals(input, output))
            {
                throw new ArgumentException(
                    $"RecordOp '{opName}': output tensor must not be the same instance as an input. " +
                    "Clone the output or allocate a new tensor to avoid gradient aliasing.");
            }
        }

        lock (_opsLock)
        {
            _ops.Add(new TapeEntry(opName, inputs, output, backward));
        }
    }

    /// <summary>
    /// Computes gradients of <paramref name="loss"/> w.r.t. all watched tensors
    /// via reverse-mode automatic differentiation.
    /// </summary>
    /// <param name="loss">The scalar loss tensor to differentiate.</param>
    /// <returns>
    /// Dictionary mapping each watched tensor that contributed to the loss
    /// to its gradient tensor. Tensors that did not contribute are omitted.
    /// </returns>
    public Dictionary<Tensor<T>, Tensor<T>> Gradient(Tensor<T> loss)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(GradientTape<T>));

        if (!Persistent && _used)
            throw new InvalidOperationException(
                "Non-persistent GradientTape can only compute gradients once. " +
                "Set persistent: true in the constructor to reuse.");

        // Stop recording during backward to avoid recording gradient ops
        bool wasRecording = IsRecording;
        IsRecording = false;

        try
        {
            var result = ReverseAccumulate(loss);

            // Mark as used AFTER successful gradient computation (#5 fix).
            // If backward throws, the tape remains usable for retry.
            _used = true;

            // Release closure references for non-persistent tapes (#10 fix).
            // The backward closures capture forward tensors — release them immediately
            // instead of waiting for Dispose().
            if (!Persistent)
            {
                lock (_opsLock) { _ops.Clear(); }
            }

            return result;
        }
        finally
        {
            if (wasRecording && Persistent) IsRecording = true;
        }
    }

    /// <summary>
    /// Resets the tape for reuse. Only valid for persistent tapes.
    /// Clears all recorded operations and watched tensors, allowing
    /// fresh recording without creating a new tape instance.
    /// </summary>
    /// <exception cref="InvalidOperationException">Thrown if the tape is not persistent.</exception>
    public void Reset()
    {
        if (!Persistent)
        {
            throw new InvalidOperationException(
                "Reset() is only valid for persistent tapes. " +
                "Non-persistent tapes should be disposed and recreated.");
        }

        lock (_opsLock)
        {
            _ops.Clear();
            _watched.Clear();
        }
        _used = false;
        IsRecording = true;
    }

    /// <summary>
    /// Enables anomaly detection during backward pass. When true, each backward
    /// function's output is checked for NaN/Inf and an exception is thrown immediately.
    /// Like PyTorch's <c>torch.autograd.set_detect_anomaly(True)</c>.
    /// </summary>
    public bool DetectAnomaly { get; set; }

    // ─── Reverse-mode AD core ────────────────────────────────────────

    private Dictionary<Tensor<T>, Tensor<T>> ReverseAccumulate(Tensor<T> loss)
    {
        var numOps = MathHelper.GetNumericOperations<T>();

        // Gradient accumulator: maps tensor → accumulated gradient
        var grads = new Dictionary<Tensor<T>, Tensor<T>>(ReferenceEqualityComparer.Instance);

        // Seed: dLoss/dLoss = 1
        var seed = Tensor<T>.CreateDefault(loss.Shape.ToArray(), numOps.One);
        grads[loss] = seed;

        // Snapshot ops under lock (another thread may be recording concurrently)
        TapeEntry[] opsSnapshot;
        lock (_opsLock) { opsSnapshot = _ops.ToArray(); }

        // Walk tape in reverse (reverse topological order by construction)
        for (int i = opsSnapshot.Length - 1; i >= 0; i--)
        {
            var entry = opsSnapshot[i];

            // Skip if this op's output has no gradient (not on the path from loss)
            if (!grads.TryGetValue(entry.Output, out var outputGrad))
                continue;

            // Compute input gradients via the backward function
            Tensor<T>[] inputGrads;
            try
            {
                inputGrads = entry.Backward(outputGrad);
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException(
                    $"Backward function for op '{entry.OpName}' failed. " +
                    $"Output grad shape: [{string.Join(", ", outputGrad.Shape.ToArray())}].", ex);
            }

            if (inputGrads.Length != entry.Inputs.Length)
                throw new InvalidOperationException(
                    $"Op '{entry.OpName}' backward returned {inputGrads.Length} gradients " +
                    $"but has {entry.Inputs.Length} inputs.");

            // Anomaly detection: check for NaN/Inf in backward outputs
            if (DetectAnomaly)
            {
                for (int j = 0; j < inputGrads.Length; j++)
                {
                    if (inputGrads[j] is null) continue;
                    for (int k = 0; k < inputGrads[j].Length; k++)
                    {
                        double val = numOps.ToDouble(inputGrads[j][k]);
                        if (double.IsNaN(val) || double.IsInfinity(val))
                            throw new ArithmeticException(
                                $"Op '{entry.OpName}' backward produced {(double.IsNaN(val) ? "NaN" : "Inf")} " +
                                $"at input[{j}][{k}]. Check forward inputs for numerical issues.");
                    }
                }
            }

            // Accumulate gradients for each input (in-place when possible)
            for (int j = 0; j < entry.Inputs.Length; j++)
            {
                var input = entry.Inputs[j];
                var grad = inputGrads[j];
                if (grad is null) continue;

                if (grads.TryGetValue(input, out var existing))
                {
                    // In-place accumulation: avoids allocating a new tensor (#9 fix)
                    for (int k = 0; k < existing.Length && k < grad.Length; k++)
                        existing[k] = numOps.Add(existing[k], grad[k]);
                }
                else
                {
                    grads[input] = grad;
                }
            }
        }

        // Filter to only watched tensors (snapshot under lock for thread safety)
        Tensor<T>[] watchedSnapshot;
        lock (_opsLock) { watchedSnapshot = _watched.ToArray(); }
        var result = new Dictionary<Tensor<T>, Tensor<T>>(ReferenceEqualityComparer.Instance);
        foreach (var tensor in watchedSnapshot)
        {
            if (grads.TryGetValue(tensor, out var grad))
                result[tensor] = grad;
        }

        return result;
    }

}

/// <summary>
/// Disables gradient tape recording within its scope.
/// Equivalent to PyTorch's <c>torch.no_grad()</c>.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Use this during inference to skip gradient tracking and save memory.
///
/// <code>
/// using var _ = new NoGradScope&lt;double&gt;();
/// var prediction = model.Predict(input); // no tape overhead
/// </code>
/// </para>
/// </remarks>
internal sealed class NoGradScope<T> : IDisposable
{
    // AsyncLocal counter: each async flow has its own suppression depth.
    // This prevents one flow's NoGradScope from affecting sibling flows.
    private static readonly AsyncLocal<int> _suppressionDepth = new();

    /// <summary>Returns true if recording is currently suppressed in this async flow.</summary>
    internal static bool IsSuppressed => _suppressionDepth.Value > 0;

    public NoGradScope()
    {
        _suppressionDepth.Value++;
    }

    public void Dispose()
    {
        if (_suppressionDepth.Value > 0)
            _suppressionDepth.Value--;
    }
}

/// <summary>
/// Reference equality comparer for using tensors as dictionary keys by identity.
/// </summary>
file sealed class ReferenceEqualityComparer : IEqualityComparer<object>
{
    public static readonly ReferenceEqualityComparer Instance = new();
    public new bool Equals(object? x, object? y) => ReferenceEquals(x, y);
    public int GetHashCode(object obj) => RuntimeHelpers.GetHashCode(obj);
}
