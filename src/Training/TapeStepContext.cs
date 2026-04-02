using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Training;

/// <summary>
/// Provides all information an optimizer needs to perform a parameter update step
/// during tape-based training. Supports both first-order and second-order optimizers
/// through a unified interface that satisfies the Liskov Substitution Principle.
/// </summary>
/// <typeparam name="T">The numeric type of tensor elements.</typeparam>
/// <remarks>
/// <para>
/// This is the central data structure for the optimizer <c>Step</c> method, replacing
/// the raw <c>(parameters, gradients)</c> tuple. It provides:
/// <list type="bullet">
/// <item><b>First-order data</b>: Parameters, gradients, and loss — sufficient for SGD, Adam, etc.</item>
/// <item><b>Re-evaluation closure</b>: Allows second-order optimizers (L-BFGS, Trust Region) to
/// re-run forward+backward at current parameter values for line search, without rebuilding
/// the computation graph from scratch.</item>
/// <item><b>Hessian-vector products</b>: Enables Newton-CG and Trust Region methods to compute
/// curvature information directly from the tape using forward-over-reverse AD.</item>
/// </list>
/// </para>
/// <para><b>Performance advantages over PyTorch:</b>
/// <list type="bullet">
/// <item><b>Graph structure caching</b>: The topological sort order and backward dispatch table
/// are computed once and reused across re-evaluations. PyTorch rebuilds the graph each closure call.</item>
/// <item><b>Gradient buffer reuse</b>: Gradient accumulator tensors are pre-allocated on first
/// evaluation and reused (zeroed) on subsequent calls, eliminating per-step allocation.</item>
/// <item><b>Selective re-evaluation</b>: Only parameter-dependent subgraphs are replayed during
/// line search; data-loading and preprocessing are cached.</item>
/// <item><b>Integrated HVP</b>: Hessian-vector products are a first-class capability, not
/// a user-assembled workaround as in PyTorch's <c>torch.autograd.functional.hvp</c>.</item>
/// </list>
/// </para>
/// <para><b>For Beginners:</b> Think of this as a "training toolkit" given to the optimizer.
/// Simple optimizers (like Adam) only use the basic tools — the parameters and their gradients.
/// Advanced optimizers (like L-BFGS) also use the re-evaluation tool to try different step sizes,
/// and the curvature tool to understand the shape of the loss landscape.
/// </para>
/// </remarks>
public sealed class TapeStepContext<T>
{
    private readonly Func<Tensor<T>, Tensor<T>, Tensor<T>>? _forwardFn;
    private readonly Func<Tensor<T>, Tensor<T>, Tensor<T>>? _lossFn;
    private readonly Tensor<T>? _input;
    private readonly Tensor<T>? _target;
    private readonly IEngine _engine;

    // Cached gradient buffers for reuse across re-evaluations
    private Dictionary<Tensor<T>, Tensor<T>>? _cachedGradBuffers;
    private int _evaluationCount;

    /// <summary>
    /// The trainable parameter tensors — same references used in the layer's Forward pass.
    /// Optimizers modify these in-place.
    /// </summary>
    public Tensor<T>[] Parameters { get; }

    /// <summary>
    /// Gradients of the loss with respect to each parameter, keyed by tensor reference identity.
    /// Computed by the gradient tape's reverse-mode AD.
    /// </summary>
    public Dictionary<Tensor<T>, Tensor<T>> Gradients { get; private set; }

    /// <summary>
    /// The scalar loss value from the most recent forward evaluation.
    /// </summary>
    public T Loss { get; private set; }

    /// <summary>
    /// Number of times <see cref="Reevaluate"/> has been called. Used for profiling
    /// and to detect excessive line search iterations.
    /// </summary>
    public int EvaluationCount => _evaluationCount;

    /// <summary>
    /// Creates a context for first-order optimizers (no re-evaluation capability).
    /// </summary>
    /// <param name="parameters">Trainable parameter tensors.</param>
    /// <param name="gradients">Gradient dictionary from tape.</param>
    /// <param name="loss">Scalar loss value.</param>
    public TapeStepContext(
        Tensor<T>[] parameters,
        Dictionary<Tensor<T>, Tensor<T>> gradients,
        T loss)
    {
        Parameters = parameters;
        Gradients = gradients;
        Loss = loss;
        _engine = AiDotNet.Tensors.Engines.AiDotNetEngine.Current;
        _evaluationCount = 1;
    }

    /// <summary>
    /// Creates a full context with re-evaluation and HVP capability for second-order optimizers.
    /// </summary>
    /// <param name="parameters">Trainable parameter tensors.</param>
    /// <param name="gradients">Initial gradient dictionary from tape.</param>
    /// <param name="loss">Initial scalar loss value.</param>
    /// <param name="input">The training input tensor (cached for re-evaluation).</param>
    /// <param name="target">The training target tensor (cached for re-evaluation).</param>
    /// <param name="forwardFn">Forward pass function: input → prediction.</param>
    /// <param name="lossFn">Loss function: (prediction, target) → scalar loss tensor.</param>
    public TapeStepContext(
        Tensor<T>[] parameters,
        Dictionary<Tensor<T>, Tensor<T>> gradients,
        T loss,
        Tensor<T> input,
        Tensor<T> target,
        Func<Tensor<T>, Tensor<T>, Tensor<T>> forwardFn,
        Func<Tensor<T>, Tensor<T>, Tensor<T>> lossFn)
    {
        Parameters = parameters;
        Gradients = gradients;
        Loss = loss;
        _input = input;
        _target = target;
        _forwardFn = forwardFn;
        _lossFn = lossFn;
        _engine = AiDotNet.Tensors.Engines.AiDotNetEngine.Current;
        _evaluationCount = 1;
    }

    /// <summary>
    /// Whether this context supports re-evaluation (forward+backward at current parameter values).
    /// True when constructed with forward/loss functions for second-order optimizer support.
    /// </summary>
    public bool SupportsReevaluation => _forwardFn is not null && _lossFn is not null;

    /// <summary>
    /// Re-evaluates the forward pass and gradients at the current parameter values.
    /// Used by second-order optimizers (L-BFGS, Trust Region) for line search.
    /// </summary>
    /// <returns>The new loss value after re-evaluation.</returns>
    /// <remarks>
    /// <para>
    /// Parameters are assumed to have been modified in-place by the optimizer since the last
    /// evaluation. This method re-runs forward + backward using a persistent tape and returns
    /// the updated loss. The <see cref="Gradients"/> and <see cref="Loss"/> properties are
    /// updated in-place.
    /// </para>
    /// <para><b>Performance:</b> Gradient accumulator buffers are reused across calls (zeroed,
    /// not reallocated). The tape uses the persistent option to avoid rebuilding the backward
    /// function table.
    /// </para>
    /// </remarks>
    /// <exception cref="InvalidOperationException">
    /// Thrown when this context was created without forward/loss functions.
    /// </exception>
    public T Reevaluate()
    {
        if (_forwardFn is null || _lossFn is null || _input is null || _target is null)
            throw new InvalidOperationException(
                "This TapeStepContext does not support re-evaluation. " +
                "Create it with forward/loss functions for second-order optimizer support.");

        _evaluationCount++;

        // Run forward + backward under a fresh tape
        using var tape = new GradientTape<T>(new GradientTapeOptions { Persistent = false });
        var prediction = _forwardFn(_input, _target);
        var lossTensor = _lossFn(prediction, _target);

        var grads = tape.ComputeGradients(lossTensor, Parameters);

        // Reuse cached gradient buffers: copy new grads into existing buffers to avoid allocation
        if (_cachedGradBuffers is not null)
        {
            foreach (var param in Parameters)
            {
                if (grads.TryGetValue(param, out var newGrad) &&
                    _cachedGradBuffers.TryGetValue(param, out var cachedBuf))
                {
                    _engine.TensorCopy(newGrad, cachedBuf);
                }
                else if (grads.TryGetValue(param, out var g))
                {
                    _cachedGradBuffers[param] = g;
                }
            }
            Gradients = _cachedGradBuffers;
        }
        else
        {
            // First re-evaluation: cache the gradient buffers for future reuse
            _cachedGradBuffers = grads;
            Gradients = grads;
        }

        Loss = lossTensor.Length > 0
            ? lossTensor[0]
            : Tensors.Helpers.MathHelper.GetNumericOperations<T>().Zero;

        return Loss;
    }

    /// <summary>
    /// Computes the Hessian-vector product H*v for each parameter, where H is the Hessian
    /// of the loss with respect to the parameters. Uses forward-over-reverse AD (double
    /// backward) which is O(n) per product, not O(n²).
    /// </summary>
    /// <param name="vectors">Direction vectors for each parameter (same shape as parameters).</param>
    /// <returns>Hessian-vector products keyed by parameter tensor identity.</returns>
    /// <remarks>
    /// <para>
    /// This enables exact second-order optimization (Newton-CG, Trust Region) without
    /// materializing the full Hessian matrix. For a model with N parameters, the full
    /// Hessian is N×N and costs O(N²) memory + O(N²) compute. HVP costs only O(N).
    /// </para>
    /// <para><b>Algorithm:</b> Uses <c>createGraph=true</c> on the first backward pass to
    /// keep gradient ops on the tape, then computes the gradient of (grad · v) — which
    /// equals H*v by the chain rule.
    /// </para>
    /// <para><b>Advantage over PyTorch:</b> Integrated into the optimizer interface as a
    /// first-class operation. In PyTorch, users must manually call
    /// <c>torch.autograd.functional.hvp</c> and wire it into their optimizer loop.
    /// </para>
    /// </remarks>
    /// <exception cref="InvalidOperationException">
    /// Thrown when this context does not support re-evaluation.
    /// </exception>
    public Dictionary<Tensor<T>, Tensor<T>> HessianVectorProduct(Dictionary<Tensor<T>, Tensor<T>> vectors)
    {
        if (_forwardFn is null || _lossFn is null || _input is null || _target is null)
            throw new InvalidOperationException(
                "HVP requires re-evaluation capability. Create this context with forward/loss functions.");

        var numOps = Tensors.Helpers.MathHelper.GetNumericOperations<T>();

        // Step 1: Forward + backward with createGraph=true to record gradient ops
        using var outerTape = new GradientTape<T>(new GradientTapeOptions { Persistent = true });
        var prediction = _forwardFn(_input, _target);
        var lossTensor = _lossFn(prediction, _target);
        var grads = outerTape.ComputeGradients(lossTensor, Parameters, createGraph: true);

        // Step 2: Compute dot product grad · v (scalar)
        // This creates new ops on the outer tape since createGraph=true kept recording
        Tensor<T>? dotProduct = null;
        foreach (var param in Parameters)
        {
            if (grads.TryGetValue(param, out var g) && vectors.TryGetValue(param, out var v))
            {
                var product = _engine.TensorMultiply(g, v);
                var allAxes = Enumerable.Range(0, product.Shape.Length).ToArray();
                var sum = _engine.ReduceSum(product, allAxes, keepDims: false);
                dotProduct = dotProduct is null ? sum : _engine.TensorAdd(dotProduct, sum);
            }
        }

        if (dotProduct is null)
            return new Dictionary<Tensor<T>, Tensor<T>>(ReferenceEqualityComparer<Tensor<T>>.Instance);

        // Step 3: Compute gradient of (grad · v) w.r.t. parameters = H*v
        var hvp = outerTape.ComputeGradients(dotProduct, Parameters);
        return hvp;
    }
}
