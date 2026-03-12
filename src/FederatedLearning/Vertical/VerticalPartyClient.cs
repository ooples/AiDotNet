using AiDotNet.FederatedLearning.Infrastructure;
using AiDotNet.Tensors;

namespace AiDotNet.FederatedLearning.Vertical;

/// <summary>
/// Represents a feature-holding party in vertical federated learning.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> This is one of the parties in a VFL setup that holds some features
/// (columns) of the dataset but NOT the labels (prediction targets). For example, a bank that
/// has income and credit score data but doesn't know patient outcomes.</para>
///
/// <para>The party runs a local "bottom model" (a small neural network) on its features to produce
/// an embedding (compressed representation) that is sent to the coordinator. During backpropagation,
/// the party receives gradients for its embedding and uses them to update its local model.</para>
///
/// <para><b>Privacy guarantee:</b> The raw features never leave the party. Only the embedding
/// (and its gradients) cross party boundaries. The embedding is a lossy compression that
/// makes it difficult to reconstruct the original features.</para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class VerticalPartyClient<T> : FederatedLearningComponentBase<T>, IVerticalParty<T>
{
    private readonly Tensor<T> _localData;
    private readonly IReadOnlyList<string> _entityIds;
    private readonly int _embeddingDim;

    // Simple bottom model: two-layer network (input -> hidden -> embedding)
    private Tensor<T> _weightsHidden = new Tensor<T>(new[] { 0 });
    private Tensor<T> _biasHidden = new Tensor<T>(new[] { 0 });
    private Tensor<T> _weightsEmbed = new Tensor<T>(new[] { 0 });
    private Tensor<T> _biasEmbed = new Tensor<T>(new[] { 0 });

    // Cached for backward pass
    private Tensor<T>? _lastInput;
    private Tensor<T>? _lastHidden;

    /// <inheritdoc/>
    public string PartyId { get; }

    /// <inheritdoc/>
    public int FeatureCount { get; }

    /// <inheritdoc/>
    public int EmbeddingDimension => _embeddingDim;

    /// <inheritdoc/>
    public bool IsLabelHolder => false;

    /// <summary>
    /// Initializes a new instance of <see cref="VerticalPartyClient{T}"/>.
    /// </summary>
    /// <param name="partyId">Unique identifier for this party.</param>
    /// <param name="localData">The local feature tensor with shape [numEntities, numFeatures].</param>
    /// <param name="entityIds">The entity IDs corresponding to each row of localData.</param>
    /// <param name="embeddingDimension">The output dimension of the bottom model.</param>
    /// <param name="hiddenDimension">The hidden layer dimension. Defaults to 2x embedding dimension.</param>
    /// <param name="seed">Random seed for weight initialization.</param>
    public VerticalPartyClient(
        string partyId,
        Tensor<T> localData,
        IReadOnlyList<string> entityIds,
        int embeddingDimension = 64,
        int hiddenDimension = 0,
        int? seed = null)
    {
        if (string.IsNullOrEmpty(partyId))
        {
            throw new ArgumentException("Party ID must not be empty.", nameof(partyId));
        }

        PartyId = partyId;
        _localData = localData ?? throw new ArgumentNullException(nameof(localData));
        _entityIds = entityIds ?? throw new ArgumentNullException(nameof(entityIds));
        _embeddingDim = embeddingDimension;
        FeatureCount = localData.Rank > 1 ? localData.Shape[1] : localData.Shape[0];

        int hiddenDim = hiddenDimension > 0 ? hiddenDimension : embeddingDimension * 2;
        InitializeWeights(FeatureCount, hiddenDim, embeddingDimension, seed);
    }

    /// <inheritdoc/>
    public IReadOnlyList<string> GetEntityIds()
    {
        return _entityIds;
    }

    /// <inheritdoc/>
    public Tensor<T> ComputeForward(IReadOnlyList<int> alignedIndices)
    {
        if (alignedIndices is null || alignedIndices.Count == 0)
        {
            throw new ArgumentException("Aligned indices must not be empty.", nameof(alignedIndices));
        }

        int batchSize = alignedIndices.Count;

        // Extract batch from local data
        var input = new Tensor<T>(new[] { batchSize, FeatureCount });
        for (int b = 0; b < batchSize; b++)
        {
            int rowIdx = alignedIndices[b];
            for (int f = 0; f < FeatureCount; f++)
            {
                input[b * FeatureCount + f] = _localData[rowIdx * FeatureCount + f];
            }
        }

        _lastInput = input;

        // Hidden layer: h = ReLU(input * W_h + b_h)
        int hiddenDim = _weightsHidden.Shape[1];
        var hidden = new Tensor<T>(new[] { batchSize, hiddenDim });
        MatMul(input, _weightsHidden, hidden, batchSize, FeatureCount, hiddenDim);
        AddBias(hidden, _biasHidden, batchSize, hiddenDim);
        ApplyRelu(hidden, batchSize * hiddenDim);

        _lastHidden = hidden;

        // Embedding layer: e = input * W_e + b_e (no activation, linear output)
        var embedding = new Tensor<T>(new[] { batchSize, _embeddingDim });
        MatMul(hidden, _weightsEmbed, embedding, batchSize, hiddenDim, _embeddingDim);
        AddBias(embedding, _biasEmbed, batchSize, _embeddingDim);

        return embedding;
    }

    /// <inheritdoc/>
    public void ApplyBackward(Tensor<T> gradients, double learningRate)
    {
        if (gradients is null)
        {
            throw new ArgumentNullException(nameof(gradients));
        }

        if (_lastInput is null || _lastHidden is null)
        {
            throw new InvalidOperationException("Forward pass must be called before backward pass.");
        }

        int batchSize = gradients.Shape[0];
        int hiddenDim = _weightsHidden.Shape[1];

        // Backward through embedding layer: dW_e = hidden^T * grad, db_e = sum(grad)
        var gradWeightsEmbed = new Tensor<T>(new[] { hiddenDim, _embeddingDim });
        MatMulTransposeA(_lastHidden, gradients, gradWeightsEmbed, batchSize, hiddenDim, _embeddingDim);
        var gradBiasEmbed = SumOverBatch(gradients, batchSize, _embeddingDim);

        // Gradient to hidden: dh = grad * W_e^T
        var gradHidden = new Tensor<T>(new[] { batchSize, hiddenDim });
        MatMulTransposeB(gradients, _weightsEmbed, gradHidden, batchSize, _embeddingDim, hiddenDim);

        // Apply ReLU gradient
        ApplyReluGradient(gradHidden, _lastHidden, batchSize * hiddenDim);

        // Backward through hidden layer
        var gradWeightsHidden = new Tensor<T>(new[] { FeatureCount, hiddenDim });
        MatMulTransposeA(_lastInput, gradHidden, gradWeightsHidden, batchSize, FeatureCount, hiddenDim);
        var gradBiasHidden = SumOverBatch(gradHidden, batchSize, hiddenDim);

        // Update parameters with SGD
        double lr = learningRate / batchSize;
        UpdateTensor(_weightsEmbed, gradWeightsEmbed, lr);
        UpdateTensor(_biasEmbed, gradBiasEmbed, lr);
        UpdateTensor(_weightsHidden, gradWeightsHidden, lr);
        UpdateTensor(_biasHidden, gradBiasHidden, lr);
    }

    /// <inheritdoc/>
    public IReadOnlyList<Tensor<T>> GetParameters()
    {
        return new[] { _weightsHidden, _biasHidden, _weightsEmbed, _biasEmbed };
    }

    /// <inheritdoc/>
    public void SetParameters(IReadOnlyList<Tensor<T>> parameters)
    {
        if (parameters is null || parameters.Count < 4)
        {
            throw new ArgumentException("Expected 4 parameter tensors.", nameof(parameters));
        }

        _weightsHidden = parameters[0];
        _biasHidden = parameters[1];
        _weightsEmbed = parameters[2];
        _biasEmbed = parameters[3];
    }

    private void InitializeWeights(int inputDim, int hiddenDim, int embDim, int? seed)
    {
        var random = seed.HasValue
            ? Tensors.Helpers.RandomHelper.CreateSeededRandom(seed.Value)
            : Tensors.Helpers.RandomHelper.CreateSecureRandom();

        // Xavier initialization
        _weightsHidden = InitXavier(inputDim, hiddenDim, random);
        _biasHidden = new Tensor<T>(new[] { hiddenDim });
        _weightsEmbed = InitXavier(hiddenDim, embDim, random);
        _biasEmbed = new Tensor<T>(new[] { embDim });
    }

    private Tensor<T> InitXavier(int fanIn, int fanOut, Random random)
    {
        double limit = Math.Sqrt(6.0 / (fanIn + fanOut));
        var tensor = new Tensor<T>(new[] { fanIn, fanOut });
        int total = fanIn * fanOut;
        for (int i = 0; i < total; i++)
        {
            tensor[i] = NumOps.FromDouble(random.NextDouble() * 2.0 * limit - limit);
        }

        return tensor;
    }

    private static void MatMul(Tensor<T> a, Tensor<T> b, Tensor<T> result, int m, int k, int n)
    {
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                double sum = 0.0;
                for (int p = 0; p < k; p++)
                {
                    sum += NumOps.ToDouble(a[i * k + p]) * NumOps.ToDouble(b[p * n + j]);
                }

                result[i * n + j] = NumOps.FromDouble(sum);
            }
        }
    }

    private static void MatMulTransposeA(Tensor<T> a, Tensor<T> b, Tensor<T> result, int m, int k, int n)
    {
        // a^T * b: a is [m, k], a^T is [k, m], result is [k, n]
        for (int i = 0; i < k; i++)
        {
            for (int j = 0; j < n; j++)
            {
                double sum = 0.0;
                for (int p = 0; p < m; p++)
                {
                    sum += NumOps.ToDouble(a[p * k + i]) * NumOps.ToDouble(b[p * n + j]);
                }

                result[i * n + j] = NumOps.FromDouble(sum);
            }
        }
    }

    private static void MatMulTransposeB(Tensor<T> a, Tensor<T> b, Tensor<T> result, int m, int k, int n)
    {
        // a * b^T: a is [m, k], b is [n, k] (stored as [k, n] transposed), result is [m, n]
        // Actually: b is [k, n], b^T is [n, k], so a[m,k] * b^T[k,n] but we want b original [k, n]
        // grad[m, embDim] * W_e^T[embDim, hiddenDim] -> result[m, hiddenDim]
        // W_e is [hiddenDim, embDim], so W_e^T is [embDim, hiddenDim]
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                double sum = 0.0;
                for (int p = 0; p < k; p++)
                {
                    // b^T[p, j] = b[j, p] where b has shape [n, k]
                    // But our b (_weightsEmbed) has shape [hiddenDim, embDim] = [n, k]
                    sum += NumOps.ToDouble(a[i * k + p]) * NumOps.ToDouble(b[j * k + p]);
                }

                result[i * n + j] = NumOps.FromDouble(sum);
            }
        }
    }

    private static void AddBias(Tensor<T> tensor, Tensor<T> bias, int batchSize, int dim)
    {
        for (int b = 0; b < batchSize; b++)
        {
            for (int d = 0; d < dim; d++)
            {
                double val = NumOps.ToDouble(tensor[b * dim + d]) + NumOps.ToDouble(bias[d]);
                tensor[b * dim + d] = NumOps.FromDouble(val);
            }
        }
    }

    private static void ApplyRelu(Tensor<T> tensor, int totalElements)
    {
        for (int i = 0; i < totalElements; i++)
        {
            double val = NumOps.ToDouble(tensor[i]);
            if (val < 0.0)
            {
                tensor[i] = NumOps.FromDouble(0.0);
            }
        }
    }

    private static void ApplyReluGradient(Tensor<T> grad, Tensor<T> activation, int totalElements)
    {
        for (int i = 0; i < totalElements; i++)
        {
            double act = NumOps.ToDouble(activation[i]);
            if (act <= 0.0)
            {
                grad[i] = NumOps.FromDouble(0.0);
            }
        }
    }

    private static Tensor<T> SumOverBatch(Tensor<T> tensor, int batchSize, int dim)
    {
        var result = new Tensor<T>(new[] { dim });
        for (int b = 0; b < batchSize; b++)
        {
            for (int d = 0; d < dim; d++)
            {
                double val = NumOps.ToDouble(result[d]) + NumOps.ToDouble(tensor[b * dim + d]);
                result[d] = NumOps.FromDouble(val);
            }
        }

        return result;
    }

    private static void UpdateTensor(Tensor<T> param, Tensor<T> grad, double lr)
    {
        int total = 1;
        for (int d = 0; d < param.Rank; d++)
        {
            total *= param.Shape[d];
        }

        for (int i = 0; i < total; i++)
        {
            double val = NumOps.ToDouble(param[i]) - lr * NumOps.ToDouble(grad[i]);
            param[i] = NumOps.FromDouble(val);
        }
    }
}
