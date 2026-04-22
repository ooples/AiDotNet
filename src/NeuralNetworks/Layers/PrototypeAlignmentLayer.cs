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
            origShape = (int[])input._shape.Clone();
            int leading = 1;
            for (int d = 0; d < rank - 1; d++) leading *= input.Shape[d];
            input2D = Engine.Reshape(input, new[] { leading, _embedDim });
        }

        int n = input2D.Shape[0];

        // Compute cosine similarities: [N, K].
        var sims = new Tensor<T>(new[] { n, _numPrototypes });
        for (int i = 0; i < n; i++)
        {
            double normI = 0;
            for (int d = 0; d < _embedDim; d++)
            {
                double v = NumOps.ToDouble(input2D.Data.Span[i * _embedDim + d]);
                normI += v * v;
            }
            normI = Math.Sqrt(normI) + 1e-8;

            for (int k = 0; k < _numPrototypes; k++)
            {
                double dot = 0, normK = 0;
                for (int d = 0; d < _embedDim; d++)
                {
                    double vi = NumOps.ToDouble(input2D.Data.Span[i * _embedDim + d]);
                    double vk = NumOps.ToDouble(_prototypes.Data.Span[k * _embedDim + d]);
                    dot += vi * vk;
                    normK += vk * vk;
                }
                normK = Math.Sqrt(normK) + 1e-8;
                sims.Data.Span[i * _numPrototypes + k] = NumOps.FromDouble(dot / (normI * normK));
            }
        }

        // Softmax over K per row of sims.
        var weights = new Tensor<T>(new[] { n, _numPrototypes });
        for (int i = 0; i < n; i++)
        {
            double max = double.NegativeInfinity;
            for (int k = 0; k < _numPrototypes; k++)
            {
                double s = NumOps.ToDouble(sims.Data.Span[i * _numPrototypes + k]);
                if (s > max) max = s;
            }
            double sum = 0;
            for (int k = 0; k < _numPrototypes; k++)
            {
                double e = Math.Exp(NumOps.ToDouble(sims.Data.Span[i * _numPrototypes + k]) - max);
                weights.Data.Span[i * _numPrototypes + k] = NumOps.FromDouble(e);
                sum += e;
            }
            if (sum > 1e-12)
            {
                for (int k = 0; k < _numPrototypes; k++)
                    weights.Data.Span[i * _numPrototypes + k] =
                        NumOps.FromDouble(NumOps.ToDouble(weights.Data.Span[i * _numPrototypes + k]) / sum);
            }
        }

        // Aggregate: output[i] = sum_k weights[i, k] * prototypes[k].
        var output2D = new Tensor<T>(new[] { n, _embedDim });
        for (int i = 0; i < n; i++)
        {
            for (int d = 0; d < _embedDim; d++)
            {
                double sum = 0;
                for (int k = 0; k < _numPrototypes; k++)
                {
                    double w = NumOps.ToDouble(weights.Data.Span[i * _numPrototypes + k]);
                    double p = NumOps.ToDouble(_prototypes.Data.Span[k * _embedDim + d]);
                    sum += w * p;
                }
                output2D.Data.Span[i * _embedDim + d] = NumOps.FromDouble(sum);
            }
        }

        if (origShape is not null)
            return Engine.Reshape(output2D, origShape);
        if (rank == 1)
            return Engine.Reshape(output2D, new[] { _embedDim });
        return output2D;
    }

    /// <inheritdoc/>
    public override void UpdateParameters(T learningRate)
    {
        // Prototypes are updated via the tape-based optimizer since they were registered
        // as a trainable parameter. This explicit per-layer UpdateParameters is a no-op
        // for tape-driven training; ParameterCount still reflects _prototypes.
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
