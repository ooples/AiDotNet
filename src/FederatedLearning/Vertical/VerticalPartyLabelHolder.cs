using AiDotNet.FederatedLearning.Infrastructure;
using AiDotNet.Tensors;

namespace AiDotNet.FederatedLearning.Vertical;

/// <summary>
/// Represents the label-holding party in vertical federated learning.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> In VFL, one special party holds the labels (prediction targets).
/// For example, a hospital knows patient outcomes while partner banks only have financial data.
/// The label holder plays a unique role:</para>
/// <list type="bullet">
/// <item><description>It may also hold some features (like a regular party).</description></item>
/// <item><description>It computes the loss function using the top model's predictions and the true labels.</description></item>
/// <item><description>It initiates backpropagation by computing loss gradients.</description></item>
/// <item><description>Its gradients must be protected (via <see cref="ILabelProtector{T}"/>)
/// to prevent feature parties from inferring labels.</description></item>
/// </list>
///
/// <para><b>Security note:</b> The label holder's gradients can reveal label information.
/// For example, large gradient magnitudes suggest the model made a large error, which hints
/// at the true label. Always use label protection in production deployments.</para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class VerticalPartyLabelHolder<T> : FederatedLearningComponentBase<T>, IVerticalParty<T>
{
    private readonly Tensor<T> _localData;
    private readonly Tensor<T> _labels;
    private readonly IReadOnlyList<string> _entityIds;
    private readonly int _embeddingDim;

    // Bottom model weights (same structure as VerticalPartyClient)
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
    public bool IsLabelHolder => true;

    /// <summary>
    /// Initializes a new instance of <see cref="VerticalPartyLabelHolder{T}"/>.
    /// </summary>
    /// <param name="partyId">Unique identifier for this party.</param>
    /// <param name="localData">The local feature tensor with shape [numEntities, numFeatures].</param>
    /// <param name="labels">The label tensor with shape [numEntities] or [numEntities, numClasses].</param>
    /// <param name="entityIds">The entity IDs corresponding to each row.</param>
    /// <param name="embeddingDimension">The output dimension of the bottom model.</param>
    /// <param name="seed">Random seed for weight initialization.</param>
    public VerticalPartyLabelHolder(
        string partyId,
        Tensor<T> localData,
        Tensor<T> labels,
        IReadOnlyList<string> entityIds,
        int embeddingDimension = 64,
        int? seed = null)
    {
        if (string.IsNullOrEmpty(partyId))
        {
            throw new ArgumentException("Party ID must not be empty.", nameof(partyId));
        }

        PartyId = partyId;
        _localData = localData ?? throw new ArgumentNullException(nameof(localData));
        _labels = labels ?? throw new ArgumentNullException(nameof(labels));
        _entityIds = entityIds ?? throw new ArgumentNullException(nameof(entityIds));
        _embeddingDim = embeddingDimension;
        FeatureCount = localData.Rank > 1 ? localData.Shape[1] : localData.Shape[0];

        int hiddenDim = embeddingDimension * 2;
        InitializeWeights(FeatureCount, hiddenDim, embeddingDimension, seed);
    }

    /// <inheritdoc/>
    public IReadOnlyList<string> GetEntityIds()
    {
        return _entityIds;
    }

    /// <summary>
    /// Gets the labels for the specified aligned entity indices.
    /// </summary>
    /// <param name="alignedIndices">The local row indices to extract labels for.</param>
    /// <returns>A label tensor for the specified entities.</returns>
    public Tensor<T> GetLabels(IReadOnlyList<int> alignedIndices)
    {
        if (alignedIndices is null || alignedIndices.Count == 0)
        {
            throw new ArgumentException("Aligned indices must not be empty.", nameof(alignedIndices));
        }

        int batchSize = alignedIndices.Count;

        // Handle both 1D and 2D label tensors
        if (_labels.Rank == 1)
        {
            var batchLabels = new Tensor<T>(new[] { batchSize });
            for (int b = 0; b < batchSize; b++)
            {
                batchLabels[b] = _labels[alignedIndices[b]];
            }

            return batchLabels;
        }
        else
        {
            int numClasses = _labels.Shape[1];
            var batchLabels = new Tensor<T>(new[] { batchSize, numClasses });
            for (int b = 0; b < batchSize; b++)
            {
                for (int c = 0; c < numClasses; c++)
                {
                    batchLabels[b * numClasses + c] = _labels[alignedIndices[b] * numClasses + c];
                }
            }

            return batchLabels;
        }
    }

    /// <summary>
    /// Computes the mean squared error loss and its gradient.
    /// </summary>
    /// <param name="predictions">The top model predictions.</param>
    /// <param name="batchLabels">The true labels for this batch.</param>
    /// <returns>A tuple of (loss value, loss gradient with respect to predictions).</returns>
    public (double Loss, Tensor<T> Gradient) ComputeLoss(Tensor<T> predictions, Tensor<T> batchLabels)
    {
        if (predictions is null)
        {
            throw new ArgumentNullException(nameof(predictions));
        }

        if (batchLabels is null)
        {
            throw new ArgumentNullException(nameof(batchLabels));
        }

        int totalElements = 1;
        for (int d = 0; d < predictions.Rank; d++)
        {
            totalElements *= predictions.Shape[d];
        }

        // MSE loss: L = (1/n) * sum((pred - label)^2)
        double lossSum = 0.0;
        var gradient = new Tensor<T>(predictions.Shape);

        for (int i = 0; i < totalElements; i++)
        {
            double pred = NumOps.ToDouble(predictions[i]);
            double label = NumOps.ToDouble(batchLabels[i]);
            double diff = pred - label;
            lossSum += diff * diff;
            gradient[i] = NumOps.FromDouble(2.0 * diff / totalElements);
        }

        double loss = lossSum / totalElements;
        return (loss, gradient);
    }

    /// <inheritdoc/>
    public Tensor<T> ComputeForward(IReadOnlyList<int> alignedIndices)
    {
        if (alignedIndices is null || alignedIndices.Count == 0)
        {
            throw new ArgumentException("Aligned indices must not be empty.", nameof(alignedIndices));
        }

        int batchSize = alignedIndices.Count;
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

        int hiddenDim = _weightsHidden.Shape[1];
        var hidden = new Tensor<T>(new[] { batchSize, hiddenDim });
        MatMul(input, _weightsHidden, hidden, batchSize, FeatureCount, hiddenDim);
        AddBias(hidden, _biasHidden, batchSize, hiddenDim);
        ApplyRelu(hidden, batchSize * hiddenDim);

        _lastHidden = hidden;

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

        // Same backward logic as VerticalPartyClient
        var gradWeightsEmbed = new Tensor<T>(new[] { hiddenDim, _embeddingDim });
        MatMulTransposeA(_lastHidden, gradients, gradWeightsEmbed, batchSize, hiddenDim, _embeddingDim);
        var gradBiasEmbed = SumOverBatch(gradients, batchSize, _embeddingDim);

        var gradHidden = new Tensor<T>(new[] { batchSize, hiddenDim });
        MatMulTransposeB(gradients, _weightsEmbed, gradHidden, batchSize, _embeddingDim, hiddenDim);
        ApplyReluGradient(gradHidden, _lastHidden, batchSize * hiddenDim);

        var gradWeightsHidden = new Tensor<T>(new[] { FeatureCount, hiddenDim });
        MatMulTransposeA(_lastInput, gradHidden, gradWeightsHidden, batchSize, FeatureCount, hiddenDim);
        var gradBiasHidden = SumOverBatch(gradHidden, batchSize, hiddenDim);

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
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                double sum = 0.0;
                for (int p = 0; p < k; p++)
                {
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
            if (NumOps.ToDouble(tensor[i]) < 0.0)
            {
                tensor[i] = NumOps.FromDouble(0.0);
            }
        }
    }

    private static void ApplyReluGradient(Tensor<T> grad, Tensor<T> activation, int totalElements)
    {
        for (int i = 0; i < totalElements; i++)
        {
            if (NumOps.ToDouble(activation[i]) <= 0.0)
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
