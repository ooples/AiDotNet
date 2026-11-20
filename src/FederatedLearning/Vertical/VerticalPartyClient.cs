using AiDotNet.Attributes;
using AiDotNet.Enums;
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
[ComponentType(ComponentType.FederatedAggregator)]
[PipelineStage(PipelineStage.Training)]
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
        var rows = VerticalEngineOps.Rows(_localData, _localData.Length / FeatureCount, FeatureCount);
        var input = Engine.TensorIndexSelect(rows, new Tensor<int>(alignedIndices.ToArray(), new[] { batchSize }), 0);

        _lastInput = input;

        // Hidden layer: h = ReLU(input * W_h + b_h)
        int hiddenDim = _weightsHidden.Shape[1];
        var hidden = VerticalEngineOps.MatMul(input, _weightsHidden, batchSize, FeatureCount, hiddenDim);
        hidden = VerticalEngineOps.Relu(VerticalEngineOps.AddBias(hidden, _biasHidden, batchSize, hiddenDim));

        _lastHidden = hidden;

        // Embedding layer: e = input * W_e + b_e (no activation, linear output)
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

        // Backward through embedding layer: dW_e = hidden^T * grad, db_e = sum(grad)
        var gradWeightsEmbed = VerticalEngineOps.MatMulTransposeA(_lastHidden, gradients, batchSize, hiddenDim, _embeddingDim);
        var gradBiasEmbed = VerticalEngineOps.SumOverBatch(gradients, batchSize, _embeddingDim);

        // Gradient to hidden: dh = grad * W_e^T
        var gradHidden = VerticalEngineOps.MatMulTransposeB(gradients, _weightsEmbed, batchSize, _embeddingDim, hiddenDim);

        // Apply ReLU gradient (zero where the activation is <= 0)
        gradHidden = VerticalEngineOps.ReluGradient(gradHidden, _lastHidden, zeroWhereActivationIsNaN: false);

        // Backward through hidden layer
        var gradWeightsHidden = VerticalEngineOps.MatMulTransposeA(_lastInput, gradHidden, batchSize, FeatureCount, hiddenDim);
        var gradBiasHidden = VerticalEngineOps.SumOverBatch(gradHidden, batchSize, hiddenDim);

        // Update parameters with SGD
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
}
