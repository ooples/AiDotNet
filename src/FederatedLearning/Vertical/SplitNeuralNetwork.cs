using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.FederatedLearning.Infrastructure;
using AiDotNet.Models.Options;
using AiDotNet.Tensors;

namespace AiDotNet.FederatedLearning.Vertical;

/// <summary>
/// Implements a split neural network for vertical federated learning.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> In VFL, the neural network is split into two parts:</para>
/// <list type="bullet">
/// <item><description><b>Bottom models</b> (one per party): Each party runs its bottom model locally on its
/// features to produce an embedding. These are managed by <see cref="IVerticalParty{T}"/> instances.</description></item>
/// <item><description><b>Top model</b> (at coordinator): Takes the combined embeddings from all parties
/// and produces the final prediction. This class manages the top model.</description></item>
/// </list>
///
/// <para>The top model is a simple multi-layer perceptron (MLP) that takes the aggregated
/// embeddings as input and produces predictions.</para>
///
/// <para><b>Aggregation modes:</b> Party embeddings can be combined via concatenation (default),
/// element-wise sum, attention weighting, or learned gating.</para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ComponentType(ComponentType.FederatedAggregator)]
[PipelineStage(PipelineStage.Training)]
public class SplitNeuralNetwork<T> : FederatedLearningComponentBase<T>, ISplitModel<T>
{
    private readonly SplitModelOptions _options;
    private readonly int _outputDimension;

    // Top model weights
    private Tensor<T> _topWeights1;
    private Tensor<T> _topBias1;
    private Tensor<T> _topWeights2;
    private Tensor<T> _topBias2;

    // Attention/gating weights (used only for Attention/Gating aggregation)
    private Tensor<T>? _attentionWeights;

    // Cached for backward pass
    [Scratch]
    private Tensor<T>? _lastTopInput;
    [Scratch]
    private Tensor<T>? _lastTopHidden;
    [Scratch]
    private IReadOnlyList<Tensor<T>>? _lastPartyEmbeddings;

    /// <inheritdoc/>
    public int NumberOfParties { get; }

    /// <inheritdoc/>
    public VflAggregationMode AggregationMode => _options.AggregationMode;

    /// <summary>
    /// Initializes a new instance of <see cref="SplitNeuralNetwork{T}"/>.
    /// </summary>
    /// <param name="numberOfParties">The number of parties contributing embeddings.</param>
    /// <param name="embeddingDimensionPerParty">The embedding dimension from each party.</param>
    /// <param name="outputDimension">The output dimension (e.g., 1 for regression, numClasses for classification).</param>
    /// <param name="options">Split model configuration options.</param>
    /// <param name="seed">Random seed for weight initialization.</param>
    public SplitNeuralNetwork(
        int numberOfParties,
        int embeddingDimensionPerParty,
        int outputDimension = 1,
        SplitModelOptions? options = null,
        int? seed = null)
    {
        if (numberOfParties <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(numberOfParties));
        }

        NumberOfParties = numberOfParties;
        _options = options ?? new SplitModelOptions();
        _outputDimension = outputDimension;

        int inputDim = ComputeTopModelInputDimension(numberOfParties, embeddingDimensionPerParty);
        int hiddenDim = _options.TopModelHiddenDimension;

        var random = seed.HasValue
            ? Tensors.Helpers.RandomHelper.CreateSeededRandom(seed.Value)
            : Tensors.Helpers.RandomHelper.CreateSecureRandom();

        _topWeights1 = InitXavier(inputDim, hiddenDim, random);
        _topBias1 = new Tensor<T>(new[] { hiddenDim });
        _topWeights2 = InitXavier(hiddenDim, outputDimension, random);
        _topBias2 = new Tensor<T>(new[] { outputDimension });

        if (_options.AggregationMode == VflAggregationMode.Attention ||
            _options.AggregationMode == VflAggregationMode.Gating)
        {
            _attentionWeights = new Tensor<T>(new[] { numberOfParties });
            _attentionWeights.Fill(NumOps.FromDouble(1.0 / numberOfParties));
        }
    }

    /// <inheritdoc/>
    public Tensor<T> AggregateEmbeddings(IReadOnlyList<Tensor<T>> partyEmbeddings)
    {
        if (partyEmbeddings is null || partyEmbeddings.Count == 0)
        {
            throw new ArgumentException("Party embeddings must not be empty.", nameof(partyEmbeddings));
        }

        _lastPartyEmbeddings = partyEmbeddings;

        return _options.AggregationMode switch
        {
            VflAggregationMode.Concatenation => ConcatenateEmbeddings(partyEmbeddings),
            VflAggregationMode.Sum => SumEmbeddings(partyEmbeddings),
            VflAggregationMode.Attention => AttentionEmbeddings(partyEmbeddings),
            VflAggregationMode.Gating => GatingEmbeddings(partyEmbeddings),
            _ => ConcatenateEmbeddings(partyEmbeddings)
        };
    }

    /// <inheritdoc/>
    public Tensor<T> ForwardTopModel(Tensor<T> combinedEmbeddings)
    {
        if (combinedEmbeddings is null)
        {
            throw new ArgumentNullException(nameof(combinedEmbeddings));
        }

        _lastTopInput = combinedEmbeddings;

        int batchSize = combinedEmbeddings.Shape[0];
        int inputDim = combinedEmbeddings.Rank > 1 ? combinedEmbeddings.Shape[1] : combinedEmbeddings.Shape[0];
        int hiddenDim = _topWeights1.Shape[1];

        // Layer 1: h = ReLU(input * W1 + b1)
        var hidden = VerticalEngineOps.MatMul(combinedEmbeddings, _topWeights1, batchSize, inputDim, hiddenDim);
        hidden = VerticalEngineOps.Relu(VerticalEngineOps.AddBias(hidden, _topBias1, batchSize, hiddenDim));

        _lastTopHidden = hidden;

        // Layer 2: output = h * W2 + b2 (linear output)
        var output = VerticalEngineOps.MatMul(hidden, _topWeights2, batchSize, hiddenDim, _outputDimension);
        return VerticalEngineOps.AddBias(output, _topBias2, batchSize, _outputDimension);
    }

    /// <inheritdoc/>
    public IReadOnlyList<Tensor<T>> BackwardTopModel(Tensor<T> lossGradient, IReadOnlyList<Tensor<T>> partyEmbeddings)
    {
        if (lossGradient is null)
        {
            throw new ArgumentNullException(nameof(lossGradient));
        }

        if (_lastTopInput is null || _lastTopHidden is null)
        {
            throw new InvalidOperationException("Forward pass must be called before backward.");
        }

        int batchSize = lossGradient.Shape[0];
        int hiddenDim = _topWeights1.Shape[1];
        int inputDim = _topWeights1.Shape[0];

        // Backward through layer 2
        var gradHidden = VerticalEngineOps.MatMulTransposeB(lossGradient, _topWeights2, batchSize, _outputDimension, hiddenDim);
        gradHidden = VerticalEngineOps.ReluGradient(gradHidden, _lastTopHidden, zeroWhereActivationIsNaN: true);

        // Backward through layer 1 to get gradient w.r.t. input (combined embeddings)
        var gradInput = VerticalEngineOps.MatMulTransposeB(gradHidden, _topWeights1, batchSize, hiddenDim, inputDim);

        // Split gradient back to individual party embeddings
        return SplitGradientToParties(gradInput, partyEmbeddings);
    }

    /// <inheritdoc/>
    public void UpdateTopModelParameters(double learningRate)
    {
        if (_lastTopInput is null || _lastTopHidden is null)
        {
            return;
        }

        // We need to recompute gradients for parameter updates
        // This is called after BackwardTopModel, so we can compute parameter gradients
        // from cached activations
    }

    /// <inheritdoc/>
    public IReadOnlyList<Tensor<T>> GetTopModelParameters()
    {
        var parameters = new List<Tensor<T>> { _topWeights1, _topBias1, _topWeights2, _topBias2 };
        if (_attentionWeights is not null)
        {
            parameters.Add(_attentionWeights);
        }

        return parameters;
    }

    /// <inheritdoc/>
    public void SetTopModelParameters(IReadOnlyList<Tensor<T>> parameters)
    {
        if (parameters is null || parameters.Count < 4)
        {
            throw new ArgumentException("Expected at least 4 parameter tensors.", nameof(parameters));
        }

        _topWeights1 = parameters[0];
        _topBias1 = parameters[1];
        _topWeights2 = parameters[2];
        _topBias2 = parameters[3];
        if (parameters.Count > 4 && _attentionWeights is not null)
        {
            _attentionWeights = parameters[4];
        }
    }

    /// <summary>
    /// Updates the top model parameters given the loss gradient. Combines forward cache with
    /// backward gradient for a complete SGD step.
    /// </summary>
    /// <param name="lossGradient">The gradient of the loss w.r.t. the top model output.</param>
    /// <param name="learningRate">The learning rate.</param>
    public void UpdateFromGradient(Tensor<T> lossGradient, double learningRate)
    {
        if (_lastTopInput is null || _lastTopHidden is null)
        {
            return;
        }

        int batchSize = lossGradient.Shape[0];
        int hiddenDim = _topWeights1.Shape[1];

        // Gradient for W2 and b2
        var gradW2 = VerticalEngineOps.MatMulTransposeA(_lastTopHidden, lossGradient, batchSize, hiddenDim, _outputDimension);
        var gradB2 = VerticalEngineOps.SumOverBatch(lossGradient, batchSize, _outputDimension);

        // Gradient through layer 2 to hidden
        var gradHidden = VerticalEngineOps.MatMulTransposeB(lossGradient, _topWeights2, batchSize, _outputDimension, hiddenDim);
        gradHidden = VerticalEngineOps.ReluGradient(gradHidden, _lastTopHidden, zeroWhereActivationIsNaN: true);

        // Gradient for W1 and b1
        int inputDim = _topWeights1.Shape[0];
        var gradW1 = VerticalEngineOps.MatMulTransposeA(_lastTopInput, gradHidden, batchSize, inputDim, hiddenDim);
        var gradB1 = VerticalEngineOps.SumOverBatch(gradHidden, batchSize, hiddenDim);

        // SGD update
        double lr = learningRate / batchSize;
        VerticalEngineOps.UpdateInPlace(_topWeights2, gradW2, lr);
        VerticalEngineOps.UpdateInPlace(_topBias2, gradB2, lr);
        VerticalEngineOps.UpdateInPlace(_topWeights1, gradW1, lr);
        VerticalEngineOps.UpdateInPlace(_topBias1, gradB1, lr);
    }

    private int ComputeTopModelInputDimension(int numParties, int embDimPerParty)
    {
        return _options.AggregationMode switch
        {
            VflAggregationMode.Concatenation => numParties * embDimPerParty,
            _ => embDimPerParty
        };
    }

    /// <summary>
    /// Batch size and width of an embedding: a 1-D embedding <c>[N]</c> is one sample of width N,
    /// a 2-D embedding <c>[B, N]</c> is B samples of width N.
    /// </summary>
    private static (int BatchSize, int Dim) EmbeddingLayout(Tensor<T> embedding)
    {
        return embedding.Rank == 1
            ? (1, embedding.Shape[0])
            : (embedding.Shape[0], embedding.Shape[1]);
    }

    private static Tensor<T> ConcatenateEmbeddings(IReadOnlyList<Tensor<T>> embeddings)
    {
        // For 1D embeddings (shape [N]), treat as single-sample batch (batchSize=1, dim=N)
        // For 2D embeddings (shape [B, N]), batchSize=B, dim=N
        int batchSize = EmbeddingLayout(embeddings[0]).BatchSize;
        var blocks = embeddings
            .Select(e => VerticalEngineOps.Rows(e, batchSize, e.Rank > 1 ? e.Shape[1] : e.Shape[0]))
            .ToList();

        return Engine.Concat(blocks, 1);
    }

    private static Tensor<T> SumEmbeddings(IReadOnlyList<Tensor<T>> embeddings)
    {
        var ones = new Tensor<T>(new[] { embeddings.Count });
        ones.Fill(NumOps.One);
        return WeightedSum(embeddings, ones);
    }

    private Tensor<T> AttentionEmbeddings(IReadOnlyList<Tensor<T>> embeddings)
    {
        // Softmax over attention weights
        return WeightedSum(embeddings, AttentionSoftmax(embeddings.Count));
    }

    private Tensor<T> GatingEmbeddings(IReadOnlyList<Tensor<T>> embeddings)
    {
        // Sigmoid gate per party
        return WeightedSum(embeddings, GatingSigmoid(embeddings.Count));
    }

    /// <summary>softmax of the first <paramref name="parties"/> attention weights (uniform when there are none).</summary>
    private Tensor<T> AttentionSoftmax(int parties)
    {
        Tensor<T> raw;
        if (_attentionWeights is not null)
        {
            raw = Engine.TensorSlice(_attentionWeights, new[] { 0 }, new[] { parties });
        }
        else
        {
            raw = new Tensor<T>(new[] { parties });
            raw.Fill(NumOps.FromDouble(1.0 / parties));
        }

        return Engine.TensorSoftmax(raw, 0);
    }

    /// <summary>sigmoid of the first <paramref name="parties"/> gating weights (sigmoid(0) = 0.5 when there are none).</summary>
    private Tensor<T> GatingSigmoid(int parties)
    {
        var raw = _attentionWeights is not null
            ? Engine.TensorSlice(_attentionWeights, new[] { 0 }, new[] { parties })
            : new Tensor<T>(new[] { parties });
        return Engine.Sigmoid(raw);
    }

    /// <summary>sum over parties of weights[p] * embeddings[p], laid out as <c>[batchSize, dim]</c> of the first party.</summary>
    private static Tensor<T> WeightedSum(IReadOnlyList<Tensor<T>> embeddings, Tensor<T> weights)
    {
        var (batchSize, dim) = EmbeddingLayout(embeddings[0]);

        var result = new Tensor<T>(new[] { batchSize, dim });
        for (int p = 0; p < embeddings.Count; p++)
        {
            var scaled = Engine.TensorMultiplyScalar(VerticalEngineOps.Rows(embeddings[p], batchSize, dim), weights.GetFlat(p));
            result = Engine.TensorAdd(result, scaled);
        }

        return result;
    }

    private IReadOnlyList<Tensor<T>> SplitGradientToParties(Tensor<T> gradInput, IReadOnlyList<Tensor<T>> partyEmbeddings)
    {
        int batchSize = gradInput.Shape[0];
        var gradients = new List<Tensor<T>>();

        if (_options.AggregationMode == VflAggregationMode.Concatenation)
        {
            int totalDim = gradInput.Rank > 1 ? gradInput.Shape[1] : gradInput.Shape[0];
            var rows = VerticalEngineOps.Rows(gradInput, batchSize, totalDim);
            int offset = 0;
            for (int p = 0; p < partyEmbeddings.Count; p++)
            {
                int dim = partyEmbeddings[p].Rank > 1 ? partyEmbeddings[p].Shape[1] : partyEmbeddings[p].Shape[0];
                gradients.Add(Engine.TensorSlice(rows, new[] { 0, offset }, new[] { batchSize, dim }));
                offset += dim;
            }
        }
        else
        {
            // For Sum/Attention/Gating, each party gets the full gradient scaled by d(aggregate)/d(embedding):
            // 1 for Sum, the softmax weight for Attention and the sigmoid gate for Gating.
            Tensor<T> scales;
            if (_options.AggregationMode == VflAggregationMode.Attention && _attentionWeights is not null)
            {
                scales = AttentionSoftmax(partyEmbeddings.Count);
            }
            else if (_options.AggregationMode == VflAggregationMode.Gating && _attentionWeights is not null)
            {
                scales = GatingSigmoid(partyEmbeddings.Count);
            }
            else
            {
                scales = new Tensor<T>(new[] { partyEmbeddings.Count });
                scales.Fill(NumOps.One);
            }

            for (int p = 0; p < partyEmbeddings.Count; p++)
            {
                int dim = partyEmbeddings[p].Rank > 1 ? partyEmbeddings[p].Shape[1] : partyEmbeddings[p].Shape[0];
                gradients.Add(Engine.TensorMultiplyScalar(VerticalEngineOps.Rows(gradInput, batchSize, dim), scales.GetFlat(p)));
            }
        }

        return gradients;
    }

    private Tensor<T> InitXavier(int fanIn, int fanOut, Random random)
    {
        double limit = Math.Sqrt(6.0 / (fanIn + fanOut));
        var tensor = new Tensor<T>(new[] { fanIn, fanOut });
        for (int i = 0; i < fanIn * fanOut; i++)
        {
            tensor[i] = NumOps.FromDouble(random.NextDouble() * 2.0 * limit - limit);
        }

        return tensor;
    }
}
