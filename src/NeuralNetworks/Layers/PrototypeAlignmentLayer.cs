using AiDotNet.Attributes;
using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// A learned prototype-alignment layer per Sun et al. 2024 "TEST: Text Prototype Aligned
/// Embedding to Activate LLM's Ability for Time Series". Maintains a bank of
/// <c>numPrototypes</c> learnable embeddings of dimension <c>embedDim</c>. For each input
/// token, computes cosine similarity to every prototype, softmax-normalizes, then
/// aggregates prototypes by weight — producing an aligned representation that lives in
/// the same prototype subspace as a frozen LLM's text-token embeddings.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Think of this as a "translator" that maps each input
/// patch to the closest matches in a small library of learned reference embeddings
/// (prototypes). The output is a weighted blend of those reference embeddings, so
/// downstream layers (often a frozen language model) see inputs in a vocabulary they
/// already understand.</para>
/// <para>
/// Forward: input <c>[B, N, D]</c> → cosine similarity with prototypes
/// <c>[K, D]</c> → softmax over K → weights <c>[B, N, K]</c> → aggregate
/// prototypes → output <c>[B, N, D]</c>.
/// </para>
/// <para>
/// The prototype bank is trainable and initialized with small random values. During
/// training, the prototypes learn to represent the "grammar" of time-series patches in a
/// way that is compatible with a downstream frozen LLM.
/// </para>
/// <para><b>Reference:</b> Sun, C. et al., "TEST: Text Prototype Aligned Embedding to
/// Activate LLM's Ability for Time Series", ICLR 2024.
/// <see href="https://openreview.net/forum?id=Tuh4nZVb0g"/>.</para>
/// </remarks>
/// <typeparam name="T">Numeric element type.</typeparam>
[LayerCategory(LayerCategory.Attention)]
[LayerCategory(LayerCategory.Structural)]
[LayerTask(LayerTask.FeatureExtraction)]
[LayerProperty(IsTrainable = true, Cost = ComputeCost.Medium, TestInputShape = "2, 4", TestConstructorArgs = "4, 3")]
public class PrototypeAlignmentLayer<T> : LayerBase<T>
{
    private readonly int _embedDim;
    private readonly int _numPrototypes;

    private Tensor<T> _prototypes; // [numPrototypes, embedDim]

    /// <inheritdoc/>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Initializes a new <see cref="PrototypeAlignmentLayer{T}"/>.
    /// </summary>
    /// <param name="embedDim">Dimension of each prototype and each input token.</param>
    /// <param name="numPrototypes">Number of learned prototypes.</param>
    public PrototypeAlignmentLayer(int embedDim, int numPrototypes)
        : base(new[] { embedDim }, new[] { embedDim })
    {
        if (embedDim < 1) throw new ArgumentOutOfRangeException(nameof(embedDim));
        if (numPrototypes < 1) throw new ArgumentOutOfRangeException(nameof(numPrototypes));

        _embedDim = embedDim;
        _numPrototypes = numPrototypes;

        // Initialize prototype bank with small random values.
        _prototypes = new Tensor<T>(new[] { numPrototypes, embedDim });
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(42);
        double scale = 1.0 / Math.Sqrt(embedDim);
        for (int i = 0; i < numPrototypes * embedDim; i++)
        {
            double u = rng.NextDouble() * 2.0 - 1.0;
            _prototypes.Data.Span[i] = NumOps.FromDouble(u * scale);
        }

        RegisterTrainableParameter(_prototypes, PersistentTensorRole.Weights);
    }

    /// <inheritdoc/>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        // Built entirely out of Engine ops so the gradient tape records the
        // cosine-similarity → softmax → aggregate chain. The old per-element
        // loop read input / _prototypes through .Data.Span and wrote results
        // into fresh Tensors, which detached gradients from the prototype
        // bank (registered as a trainable parameter) — so the parameter was
        // trained via tape-driven autodiff but the layer itself never
        // contributed gradient. Per-op build below keeps the whole chain
        // differentiable.

        if (input is null)
            throw new ArgumentNullException(nameof(input));
        if (input.Rank < 1)
            throw new ArgumentException(
                "PrototypeAlignmentLayer expects at least rank-1 input.", nameof(input));

        // Boundary-check the trailing dimension against the expected embedding
        // width so a bad caller shape surfaces as a clear argument error
        // instead of an opaque reshape/matmul failure deeper in the stack.
        int trailingDim = input.Shape[input.Rank - 1];
        if (trailingDim != _embedDim)
        {
            throw new ArgumentException(
                $"PrototypeAlignmentLayer expected trailing dimension {_embedDim}, "
                + $"but got shape [{string.Join(", ", input.Shape.ToArray())}].",
                nameof(input));
        }

        // Flatten leading dimensions → [N, embedDim].
        int rank = input.Rank;
        Tensor<T> input2D;
        int[]? origShape = null;
        if (rank == 1)
        {
            input2D = Engine.Reshape(input, new[] { 1, input.Shape[0] });
        }
        else if (rank == 2)
        {
            input2D = input;
        }
        else
        {
            origShape = input.Shape.ToArray();
            int leading = 1;
            for (int d = 0; d < rank - 1; d++) leading *= input.Shape[d];
            input2D = Engine.Reshape(input, new[] { leading, _embedDim });
        }

        var eps = NumOps.FromDouble(1e-8);

        // dotProducts = input @ prototypes^T  → [N, K]
        var protoT = Engine.TensorTranspose(_prototypes);                              // [D, K]
        var dotProducts = Engine.TensorMatMul(input2D, protoT);                        // [N, K]

        // Input norms: sqrt(sum(input^2, axis=1, keepDims=true)) → [N, 1]
        var inputSq = Engine.TensorSquare(input2D);                                    // [N, D]
        var inputNorm = Engine.TensorSqrt(
            Engine.TensorAddScalar(Engine.ReduceSum(inputSq, new[] { 1 }, keepDims: true), eps)); // [N, 1]

        // Prototype norms: sqrt(sum(prototypes^2, axis=1, keepDims=true)) → [K, 1],
        // then transpose to [1, K] so we can broadcast-divide against [N, K].
        var protoSq = Engine.TensorSquare(_prototypes);                                // [K, D]
        var protoNormCol = Engine.TensorSqrt(
            Engine.TensorAddScalar(Engine.ReduceSum(protoSq, new[] { 1 }, keepDims: true), eps)); // [K, 1]
        var protoNormRow = Engine.TensorTranspose(protoNormCol);                       // [1, K]

        // Cosine sims = dotProducts / (inputNorm * protoNormRow), with both broadcasts.
        var denom = Engine.TensorBroadcastMultiply(inputNorm, protoNormRow);           // [N, K]
        var sims = Engine.TensorBroadcastDivide(dotProducts, denom);                   // [N, K]

        // Softmax over the prototype axis so each row is a distribution over K.
        var weights = Engine.TensorSoftmax(sims, axis: 1);                             // [N, K]

        // Aggregate: output = weights @ prototypes  → [N, D]
        var output2D = Engine.TensorMatMul(weights, _prototypes);                      // [N, D]

        if (origShape is not null)
            return Engine.Reshape(output2D, origShape);
        if (rank == 1)
            return Engine.Reshape(output2D, new[] { _embedDim });
        return output2D;
    }

    /// <inheritdoc/>
    /// <remarks>
    /// No-op by design. Prototypes were registered with
    /// <see cref="LayerBase{T}.RegisterTrainableParameter"/>, so the tape-based
    /// <c>NeuralNetworkBase.TrainWithTape</c> path updates them through the optimizer's
    /// <c>Step(TapeStepContext)</c>. For non-tape training paths (legacy per-layer
    /// <c>UpdateParameters(learningRate)</c> flow), the parameter is still addressable
    /// via <see cref="GetParameters"/> / <see cref="SetParameters"/> so external
    /// drivers can apply updates explicitly — there is no internal gradient buffer to
    /// consume here.
    /// </remarks>
    public override void UpdateParameters(T learningRate)
    {
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameters()
    {
        var vec = new T[_numPrototypes * _embedDim];
        for (int i = 0; i < vec.Length; i++)
            vec[i] = _prototypes.Data.Span[i];
        return new Vector<T>(vec);
    }

    /// <inheritdoc/>
    public override void ResetState()
    {
        // No cached forward state.
    }
}
