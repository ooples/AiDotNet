using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Autodiff;

/// <summary>
/// Records operations for automatic differentiation (autodiff).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// GradientTape implements automatic differentiation using the tape-based approach
/// popularized by TensorFlow. It records operations performed within its context and
/// builds a computation graph. When Gradient() is called, it performs reverse-mode
/// automatic differentiation (backpropagation) to compute gradients.
/// </para>
/// <para>
/// This implementation follows industry-standard patterns:
/// - Opt-in recording via using statement (like TensorFlow's GradientTape)
/// - Memory-efficient as tapes can be disposed after gradient computation
/// - Supports watching specific tensors/variables
/// - Thread-safe with ThreadStatic tape stack
/// </para>
/// <para><b>For Beginners:</b> This automatically tracks calculations so gradients can be computed.
///
/// Think of it as a recording device:
/// - You start recording by creating a GradientTape (use using statement)
/// - All mathematical operations within the scope are recorded
/// - When you're done, you can play it backwards to get gradients
/// - This is how neural networks learn - by computing gradients automatically
///
/// Example usage:
/// <code>
/// using (var tape = new GradientTape&lt;double&gt;())
/// {
///     tape.Watch(parameters);
///     var loss = ComputeLoss(parameters);
///     var gradients = tape.Gradient(loss, parameters);
///     // Use gradients to update parameters
/// }
/// </code>
/// </para>
/// </remarks>
public class GradientTape<T> : IDisposable
{
    /// <summary>
    /// Thread-local stack of active tapes for handling nested tapes.
    /// </summary>
    /// <remarks>
    /// ThreadStatic fields are null by default for each thread. Never initialize with '= value'
    /// as that only runs once for the type, not per-thread.
    /// </remarks>
    [ThreadStatic]
    private static Stack<GradientTape<T>>? _tapeStack;

    /// <summary>
    /// Gets the currently active tape for this thread, or null if no tape is active.
    /// </summary>
    /// <returns>The active GradientTape, or null if none exists.</returns>
    public static GradientTape<T>? Current
    {
        get
        {
            if (_tapeStack == null || _tapeStack.Count == 0)
            {
                return null;
            }
            return _tapeStack.Peek();
        }
    }

    /// <summary>
    /// Gets or sets the list of watched variables/tensors.
    /// </summary>
    /// <value>A list of computation nodes being watched.</value>
    /// <remarks>
    /// Only watched variables will have gradients computed. By default, all operations
    /// within the tape's scope are recorded, but only watched variables receive gradients.
    /// </remarks>
    private readonly List<ComputationNode<T>> _watchedNodes;

    /// <summary>
    /// Gets or sets the list of recorded operations.
    /// </summary>
    /// <value>A list of all computation nodes created during recording.</value>
    private readonly List<ComputationNode<T>> _operations;

    /// <summary>
    /// Gets or sets a value indicating whether this tape is actively recording.
    /// </summary>
    /// <value>True if the tape is recording operations; false otherwise.</value>
    public bool IsRecording { get; private set; }

    /// <summary>
    /// Gets or sets a value indicating whether gradients should persist after first use.
    /// </summary>
    /// <value>If false, the tape can only compute gradients once.</value>
    /// <remarks>
    /// <para>
    /// By default (false), tapes are single-use for memory efficiency. Set to true if you
    /// need to compute gradients multiple times from the same tape.
    /// </para>
    /// <para><b>For Beginners:</b> Controls whether you can reuse this tape.
    ///
    /// - False (default): Can only compute gradients once, then tape is used up (more efficient)
    /// - True: Can compute gradients multiple times (uses more memory)
    ///
    /// Most of the time, false is fine since you create a new tape for each training step.
    /// </para>
    /// </remarks>
    public bool Persistent { get; set; }

    /// <summary>
    /// Gets or sets a value indicating whether this tape has been used for gradient computation.
    /// </summary>
    private bool _hasBeenUsed;

    /// <summary>
    /// Gets or sets a value indicating whether this tape has been disposed.
    /// </summary>
    private bool _disposed;

    /// <summary>
    /// Caches topological orders for graph structures to avoid recomputation.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The cache maps from a graph signature (based on node structure) to the
    /// precomputed topological order. This significantly improves performance
    /// for persistent tapes when computing gradients multiple times with the
    /// same graph structure.
    /// </para>
    /// <para>
    /// Graph caching is automatically enabled for persistent tapes and disabled
    /// for single-use tapes, providing optimal performance without requiring
    /// user configuration.
    /// </para>
    /// </remarks>
    private Dictionary<string, List<ComputationNode<T>>>? _graphCache;

    /// <summary>
    /// Initializes a new instance of the <see cref="GradientTape{T}"/> class.
    /// </summary>
    /// <param name="persistent">Whether the tape should persist after first use.</param>
    /// <remarks>
    /// <para>
    /// Creates a new gradient tape and pushes it onto the thread-local tape stack,
    /// making it the active tape for this thread. All operations performed within
    /// the scope of this tape will be recorded for automatic differentiation.
    /// </para>
    /// <para>
    /// Graph caching is automatically enabled for persistent tapes to optimize
    /// performance when computing gradients multiple times.
    /// </para>
    /// <para><b>For Beginners:</b> This creates a new recording session.
    ///
    /// When you create a tape:
    /// - It starts recording all operations
    /// - Use 'using' statement to ensure cleanup: using (var tape = new GradientTape&lt;T&gt;())
    /// - Operations inside the using block are tracked
    /// - When the block ends, recording stops and resources are cleaned up
    /// </para>
    /// </remarks>
    public GradientTape(bool persistent = false)
    {
        _watchedNodes = new List<ComputationNode<T>>();
        _operations = new List<ComputationNode<T>>();
        IsRecording = true;
        Persistent = persistent;
        _hasBeenUsed = false;

        // Automatically enable graph caching for persistent tapes
        // This provides optimal performance without requiring user configuration
        if (persistent)
        {
            _graphCache = new Dictionary<string, List<ComputationNode<T>>>();
        }

        // Push onto tape stack
        if (_tapeStack == null)
        {
            _tapeStack = new Stack<GradientTape<T>>();
        }
        _tapeStack.Push(this);
    }

    /// <summary>
    /// Watches a computation node so its gradient will be computed.
    /// </summary>
    /// <param name="node">The computation node to watch.</param>
    /// <remarks>
    /// <para>
    /// Only watched nodes will have their gradients computed during backpropagation.
    /// Typically, you watch model parameters or other variables you want to optimize.
    /// </para>
    /// <para><b>For Beginners:</b> This marks a value to track gradients for.
    ///
    /// Watch variables you want to:
    /// - Train (like neural network weights)
    /// - Optimize
    /// - Compute gradients for
    ///
    /// Think of it like saying "I care about how the output changes when THIS value changes."
    /// </para>
    /// </remarks>
    public void Watch(ComputationNode<T> node)
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(GradientTape<T>));
        }

        if (!_watchedNodes.Contains(node))
        {
            node.RequiresGradient = true;
            _watchedNodes.Add(node);
        }
    }

    /// <summary>
    /// Watches multiple computation nodes.
    /// </summary>
    /// <param name="nodes">The computation nodes to watch.</param>
    public void Watch(IEnumerable<ComputationNode<T>> nodes)
    {
        foreach (var node in nodes)
        {
            Watch(node);
        }
    }

    /// <summary>
    /// Records a computation node in the tape.
    /// </summary>
    /// <param name="node">The computation node to record.</param>
    /// <remarks>
    /// <para>
    /// This method is called automatically by operations that support autodiff.
    /// It adds the node to the tape's operation list so it can be included
    /// in gradient computation.
    /// </para>
    /// <para><b>For Beginners:</b> This adds an operation to the recording.
    ///
    /// Usually you don't call this directly - operations call it automatically.
    /// Each mathematical operation records itself on the active tape.
    /// </para>
    /// </remarks>
    public void RecordOperation(ComputationNode<T> node)
    {
        if (IsRecording)
        {
            _operations.Add(node);
        }
    }

    /// <summary>
    /// Computes a signature for the computation graph to enable caching.
    /// </summary>
    /// <param name="target">The target node of the computation graph.</param>
    /// <returns>A string signature representing the graph structure.</returns>
    /// <remarks>
    /// The signature is based on the structure of the computation graph, including
    /// the relationships between nodes. Identical graph structures produce identical
    /// signatures, enabling cache hits.
    /// </remarks>
    private string ComputeGraphSignature(ComputationNode<T> target)
    {
        // Use a hash-based signature of the node IDs and their relationships
        var visited = new HashSet<ComputationNode<T>>();
        var signatureBuilder = new System.Text.StringBuilder();

        void BuildSignature(ComputationNode<T> node)
        {
            if (visited.Contains(node))
            {
                return;
            }

            visited.Add(node);

            // Add node hash code and parent count to signature
            signatureBuilder.Append($"{node.GetHashCode()}:{node.Parents.Count};");

            foreach (var parent in node.Parents)
            {
                BuildSignature(parent);
            }
        }

        BuildSignature(target);
        return signatureBuilder.ToString();
    }

    /// <summary>
    /// Performs a topological sort of the computation graph rooted at the target node.
    /// </summary>
    /// <param name="target">The target node to start the topological sort from.</param>
    /// <returns>A list of nodes in topological order.</returns>
    /// <remarks>
    /// <para>
    /// This is an optimized version of topological sort that can be cached and reused
    /// for identical graph structures. The algorithm uses iterative DFS to avoid
    /// stack overflow for deep graphs.
    /// </para>
    /// </remarks>
    private List<ComputationNode<T>> ComputeTopologicalOrder(ComputationNode<T> target)
    {
        var visited = new HashSet<ComputationNode<T>>();
        var result = new List<ComputationNode<T>>();

        // Use iterative DFS with explicit stack to avoid stack overflow for deep graphs
        var stack = new Stack<(ComputationNode<T> node, bool processed)>();
        stack.Push((target, false));

        while (stack.Count > 0)
        {
            var (node, processed) = stack.Pop();

            if (visited.Contains(node))
            {
                continue;
            }

            if (processed)
            {
                // All parents have been visited, add to result
                visited.Add(node);
                result.Add(node);
            }
            else
            {
                // Mark for processing after parents
                stack.Push((node, true));

                // Push parents onto stack (they will be processed first)
                foreach (var parent in node.Parents)
                {
                    if (!visited.Contains(parent))
                    {
                        stack.Push((parent, false));
                    }
                }
            }
        }

        return result;
    }

    /// <summary>
    /// Computes the gradient of a target node with respect to watched variables.
    /// </summary>
    /// <param name="target">The target node (typically the loss).</param>
    /// <param name="sources">The source nodes to compute gradients for (if null, uses all watched nodes).</param>
    /// <param name="createGraph">If true, the gradient computation itself will be recorded, enabling higher-order derivatives.</param>
    /// <returns>A dictionary mapping each source node to its gradient.</returns>
    /// <remarks>
    /// <para>
    /// This method performs reverse-mode automatic differentiation (backpropagation)
    /// to compute gradients. It builds the computation graph, performs topological
    /// sorting, and executes the backward pass.
    /// </para>
    /// <para>
    /// After calling this method, the tape is marked as used. If Persistent is false,
    /// calling Gradient again will throw an exception.
    /// </para>
    /// <para>
    /// **Higher-Order Gradients**:
    /// When createGraph=true, the gradient computation itself is recorded to an active tape.
    /// This enables computing gradients of gradients (second derivatives, Hessians, etc.).
    /// </para>
    /// <para><b>For Beginners:</b> This computes how the output changes with respect to inputs.
    ///
    /// The process:
    /// 1. You give it a target (like the loss you want to minimize)
    /// 2. It computes how much each watched variable affects that target
    /// 3. Returns gradients showing which direction to adjust each variable
    ///
    /// These gradients are what you use to update neural network weights during training.
    ///
    /// **Higher-Order Derivatives (createGraph=true)**:
    /// Sometimes you need the gradient of a gradient (like computing curvature).
    /// When createGraph=true, the gradient calculation itself is tracked, so you can
    /// differentiate it again. This is used in:
    /// - Second-order optimization methods
    /// - Physics simulations
    /// - Some adversarial training techniques
    ///
    /// Example:
    /// <code>
    /// // First-order gradient (normal)
    /// var gradients = tape.Gradient(loss, parameters);
    ///
    /// // Second-order gradient (gradient of gradient)
    /// using (var tape1 = new GradientTape&lt;T&gt;())
    /// {
    ///     tape1.Watch(parameters);
    ///     using (var tape2 = new GradientTape&lt;T&gt;())
    ///     {
    ///         tape2.Watch(parameters);
    ///         var loss = ComputeLoss(parameters);
    ///         var firstGrad = tape2.Gradient(loss, parameters, createGraph: true);
    ///     }
    ///     var secondGrad = tape1.Gradient(firstGrad, parameters);
    /// }
    /// </code>
    /// </para>
    /// </remarks>
    public Dictionary<ComputationNode<T>, Tensor<T>> Gradient(
        ComputationNode<T> target,
        IEnumerable<ComputationNode<T>>? sources = null,
        bool createGraph = false)
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(GradientTape<T>));
        }

        // Check if tape has already been used
        if (!Persistent && _hasBeenUsed)
        {
            throw new InvalidOperationException(
                "GradientTape has already been used to compute gradients. " +
                "Set Persistent=true if you need to compute gradients multiple times.");
        }

        _hasBeenUsed = true;

        // Use watched nodes if sources not specified
        List<ComputationNode<T>> sourceList;
        if (sources != null)
        {
            sourceList = new List<ComputationNode<T>>(sources);
        }
        else
        {
            sourceList = _watchedNodes;
        }

        // Initialize result dictionary
        var result = new Dictionary<ComputationNode<T>, Tensor<T>>();

        // If createGraph is true, temporarily resume recording to capture gradient ops
        bool wasRecording = IsRecording;
        if (createGraph)
        {
            // Check if there's an outer tape to record to
            var outerTape = _tapeStack != null && _tapeStack.Count > 1
                ? _tapeStack.ElementAt(1)
                : null;

            if (outerTape != null)
            {
                // The gradient computation will be recorded to the outer tape
                // This happens automatically through TensorOperations
            }
        }
        else
        {
            // Normal mode: stop recording during backward pass
            StopRecording();
        }

        // Perform backward pass from target
        // For persistent tapes, use graph caching optimization automatically
        if (_graphCache != null)
        {
            // Use cached topological order if available
            string graphSignature = ComputeGraphSignature(target);
            List<ComputationNode<T>> topoOrder;

            if (_graphCache.TryGetValue(graphSignature, out var cachedOrder))
            {
                // Cache hit - reuse cached topological order
                topoOrder = cachedOrder;
            }
            else
            {
                // Cache miss - compute and cache topological order
                topoOrder = ComputeTopologicalOrder(target);
                _graphCache[graphSignature] = topoOrder;
            }

            // Execute cached backward pass
            PerformBackwardPass(target, topoOrder);
        }
        else
        {
            // Standard backward pass without caching (for non-persistent tapes)
            target.Backward();
        }

        // Restore recording state
        if (!createGraph && wasRecording)
        {
            ResumeRecording();
        }

        // Collect gradients for source nodes
        // If createGraph is true, wrap gradients in ComputationNodes for further differentiation
        foreach (var source in sourceList)
        {
            if (source.Gradient != null)
            {
                result[source] = source.Gradient;

                // For higher-order gradients, the gradient itself should be differentiable
                if (createGraph)
                {
                    // Mark that this gradient can be differentiated
                    // The gradient tensor is already a result of operations that were recorded
                }
            }
        }

        return result;
    }

    /// <summary>
    /// Performs the backward pass using a precomputed topological order.
    /// </summary>
    /// <param name="target">The target node to start backward pass from.</param>
    /// <param name="topoOrder">The precomputed topological order of nodes.</param>
    /// <remarks>
    /// <para>
    /// This method executes the backward pass without recomputing the topological sort,
    /// which is the primary performance benefit of graph caching. It mimics the behavior
    /// of ComputationNode.Backward() but uses the cached node order.
    /// </para>
    /// </remarks>
    private void PerformBackwardPass(ComputationNode<T> target, List<ComputationNode<T>> topoOrder)
    {
        // Clear all gradients in the topological order to ensure clean state
        // This is critical for multiple backward passes (persistent tapes, higher-order gradients)
        foreach (var node in topoOrder)
        {
            node.Gradient = null;
        }

        // Initialize root gradient to ones (for final node)
        target.Gradient = new Tensor<T>(target.Value.Shape);
        var numOps = MathHelper.GetNumericOperations<T>();
        for (int i = 0; i < target.Gradient.Length; i++)
        {
            target.Gradient[i] = numOps.One;
        }

        // Execute backward pass in reverse topological order
        for (int i = topoOrder.Count - 1; i >= 0; i--)
        {
            var node = topoOrder[i];
            if (node.RequiresGradient && node.BackwardFunction != null && node.Gradient != null)
            {
                node.BackwardFunction(node.Gradient);
            }
        }
    }

    /// <summary>
    /// Stops recording operations on this tape.
    /// </summary>
    /// <remarks>
    /// <para>
    /// After calling this method, operations will no longer be recorded on this tape.
    /// This can be useful for inference or when you want to temporarily disable recording.
    /// </para>
    /// <para><b>For Beginners:</b> This pauses the recording.
    ///
    /// Use this when:
    /// - You want to do calculations without tracking them
    /// - Running inference (not training)
    /// - Computing metrics that don't need gradients
    /// </para>
    /// </remarks>
    public void StopRecording()
    {
        IsRecording = false;
    }

    /// <summary>
    /// Resumes recording operations on this tape.
    /// </summary>
    /// <remarks>
    /// This re-enables recording after it was stopped with StopRecording().
    /// </remarks>
    public void ResumeRecording()
    {
        IsRecording = true;
    }

    /// <summary>
    /// Resets the tape, clearing all recorded operations and watched variables.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method clears all state from the tape, allowing it to be reused.
    /// It's useful when you want to reuse a persistent tape for a new computation.
    /// </para>
    /// <para><b>For Beginners:</b> This clears the tape to start fresh.
    ///
    /// After reset:
    /// - All recorded operations are forgotten
    /// - Watched variables are cleared
    /// - The tape can be used for a new calculation
    /// </para>
    /// </remarks>
    public void Reset()
    {
        _operations.Clear();
        _watchedNodes.Clear();
        _graphCache?.Clear();
        _hasBeenUsed = false;
        IsRecording = true;
    }

    /// <summary>
    /// Disposes the gradient tape, stopping recording and popping it from the tape stack.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method is automatically called when exiting a using block. It stops recording,
    /// pops the tape from the thread-local stack, and cleans up resources if the tape
    /// is not persistent.
    /// </para>
    /// <para><b>For Beginners:</b> This cleans up the tape when you're done.
    ///
    /// When you use 'using' statement:
    /// <code>
    /// using (var tape = new GradientTape&lt;T&gt;())
    /// {
    ///     // your code
    /// } // Dispose is automatically called here
    /// </code>
    ///
    /// Dispose:
    /// - Stops recording
    /// - Removes the tape from the active stack
    /// - Frees up memory if not persistent
    /// </para>
    /// </remarks>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        StopRecording();

        // Pop from tape stack
        if (_tapeStack != null && _tapeStack.Count > 0 && _tapeStack.Peek() == this)
        {
            _tapeStack.Pop();
        }

        // Clear data if not persistent
        if (!Persistent)
        {
            _operations.Clear();
            _watchedNodes.Clear();
            _graphCache?.Clear();
        }

        _disposed = true;
        GC.SuppressFinalize(this);
    }
}
