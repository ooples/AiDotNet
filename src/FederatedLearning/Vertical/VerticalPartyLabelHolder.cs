using AiDotNet.Attributes;
using AiDotNet.Enums;
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
[ComponentType(ComponentType.FederatedAggregator)]
[PipelineStage(PipelineStage.Training)]
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
    [Scratch]
    private Tensor<T>? _lastInput;
    [Scratch]
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

        var indices = new Tensor<int>(alignedIndices.ToArray(), new[] { batchSize });

        // Handle both 1D and 2D label tensors
        if (_labels.Rank == 1)
        {
            return Engine.TensorIndexSelect(_labels, indices, 0);
        }
        else
        {
            int numClasses = _labels.Shape[1];
            var rows = VerticalEngineOps.Rows(_labels, _labels.Length / numClasses, numClasses);
            return Engine.TensorIndexSelect(rows, indices, 0);
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
        if (totalElements == 0)
        {
            return (double.NaN, new Tensor<T>(predictions._shape));
        }

        var diff = Engine.TensorSubtract(
            VerticalEngineOps.Rows(predictions, 1, totalElements),
            VerticalEngineOps.Rows(batchLabels, 1, totalElements));
        double lossSum = NumOps.ToDouble(Engine.TensorSum(Engine.TensorSquare(diff)));
        var gradient = Engine.TensorMultiplyScalar(diff, NumOps.FromDouble(2.0 / totalElements)).Reshape(predictions._shape);

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
        var rows = VerticalEngineOps.Rows(_localData, _localData.Length / FeatureCount, FeatureCount);
        var input = Engine.TensorIndexSelect(rows, new Tensor<int>(alignedIndices.ToArray(), new[] { batchSize }), 0);

        _lastInput = input;

        int hiddenDim = _weightsHidden.Shape[1];
        var hidden = VerticalEngineOps.MatMul(input, _weightsHidden, batchSize, FeatureCount, hiddenDim);
        hidden = VerticalEngineOps.Relu(VerticalEngineOps.AddBias(hidden, _biasHidden, batchSize, hiddenDim));

        _lastHidden = hidden;

        var embedding = VerticalEngineOps.MatMul(hidden, _weightsEmbed, batchSize, hiddenDim, _embeddingDim);
        return VerticalEngineOps.AddBias(embedding, _biasEmbed, batchSize, _embeddingDim);
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
        var gradWeightsEmbed = VerticalEngineOps.MatMulTransposeA(_lastHidden, gradients, batchSize, hiddenDim, _embeddingDim);
        var gradBiasEmbed = VerticalEngineOps.SumOverBatch(gradients, batchSize, _embeddingDim);

        var gradHidden = VerticalEngineOps.MatMulTransposeB(gradients, _weightsEmbed, batchSize, _embeddingDim, hiddenDim);
        gradHidden = VerticalEngineOps.ReluGradient(gradHidden, _lastHidden, zeroWhereActivationIsNaN: true);

        var gradWeightsHidden = VerticalEngineOps.MatMulTransposeA(_lastInput, gradHidden, batchSize, FeatureCount, hiddenDim);
        var gradBiasHidden = VerticalEngineOps.SumOverBatch(gradHidden, batchSize, hiddenDim);

        double lr = learningRate / batchSize;
        VerticalEngineOps.UpdateInPlace(_weightsEmbed, gradWeightsEmbed, lr);
        VerticalEngineOps.UpdateInPlace(_biasEmbed, gradBiasEmbed, lr);
        VerticalEngineOps.UpdateInPlace(_weightsHidden, gradWeightsHidden, lr);
        VerticalEngineOps.UpdateInPlace(_biasHidden, gradBiasHidden, lr);
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
}
