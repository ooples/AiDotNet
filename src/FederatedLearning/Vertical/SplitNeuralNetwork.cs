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
    private Tensor<T>? _lastTopInput;
    private Tensor<T>? _lastTopHidden;
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
            double initVal = 1.0 / numberOfParties;
            for (int i = 0; i < numberOfParties; i++)
            {
                _attentionWeights[i] = NumOps.FromDouble(initVal);
            }
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
        var hidden = new Tensor<T>(new[] { batchSize, hiddenDim });
        MatMul(combinedEmbeddings, _topWeights1, hidden, batchSize, inputDim, hiddenDim);
        AddBias(hidden, _topBias1, batchSize, hiddenDim);
        ApplyRelu(hidden, batchSize * hiddenDim);

        _lastTopHidden = hidden;

        // Layer 2: output = h * W2 + b2 (linear output)
        var output = new Tensor<T>(new[] { batchSize, _outputDimension });
        MatMul(hidden, _topWeights2, output, batchSize, hiddenDim, _outputDimension);
        AddBias(output, _topBias2, batchSize, _outputDimension);

        return output;
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
        var gradHidden = new Tensor<T>(new[] { batchSize, hiddenDim });
        MatMulTransposeB(lossGradient, _topWeights2, gradHidden, batchSize, _outputDimension, hiddenDim);
        ApplyReluGradient(gradHidden, _lastTopHidden, batchSize * hiddenDim);

        // Backward through layer 1 to get gradient w.r.t. input (combined embeddings)
        var gradInput = new Tensor<T>(new[] { batchSize, inputDim });
        MatMulTransposeB(gradHidden, _topWeights1, gradInput, batchSize, hiddenDim, inputDim);

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
        var gradW2 = new Tensor<T>(new[] { hiddenDim, _outputDimension });
        MatMulTransposeA(_lastTopHidden, lossGradient, gradW2, batchSize, hiddenDim, _outputDimension);
        var gradB2 = SumOverBatch(lossGradient, batchSize, _outputDimension);

        // Gradient through layer 2 to hidden
        var gradHidden = new Tensor<T>(new[] { batchSize, hiddenDim });
        MatMulTransposeB(lossGradient, _topWeights2, gradHidden, batchSize, _outputDimension, hiddenDim);
        ApplyReluGradient(gradHidden, _lastTopHidden, batchSize * hiddenDim);

        // Gradient for W1 and b1
        int inputDim = _topWeights1.Shape[0];
        var gradW1 = new Tensor<T>(new[] { inputDim, hiddenDim });
        MatMulTransposeA(_lastTopInput, gradHidden, gradW1, batchSize, inputDim, hiddenDim);
        var gradB1 = SumOverBatch(gradHidden, batchSize, hiddenDim);

        // SGD update
        double lr = learningRate / batchSize;
        UpdateTensor(_topWeights2, gradW2, lr);
        UpdateTensor(_topBias2, gradB2, lr);
        UpdateTensor(_topWeights1, gradW1, lr);
        UpdateTensor(_topBias1, gradB1, lr);
    }

    private int ComputeTopModelInputDimension(int numParties, int embDimPerParty)
    {
        return _options.AggregationMode switch
        {
            VflAggregationMode.Concatenation => numParties * embDimPerParty,
            _ => embDimPerParty
        };
    }

    private static Tensor<T> ConcatenateEmbeddings(IReadOnlyList<Tensor<T>> embeddings)
    {
        // For 1D embeddings (shape [N]), treat as single-sample batch (batchSize=1, dim=N)
        // For 2D embeddings (shape [B, N]), batchSize=B, dim=N
        bool is1D = embeddings[0].Rank == 1;
        int batchSize = is1D ? 1 : embeddings[0].Shape[0];
        int totalDim = 0;
        for (int p = 0; p < embeddings.Count; p++)
        {
            totalDim += embeddings[p].Rank > 1 ? embeddings[p].Shape[1] : embeddings[p].Shape[0];
        }

        var result = new Tensor<T>(new[] { batchSize, totalDim });
        int offset = 0;
        for (int p = 0; p < embeddings.Count; p++)
        {
            int dim = embeddings[p].Rank > 1 ? embeddings[p].Shape[1] : embeddings[p].Shape[0];
            for (int b = 0; b < batchSize; b++)
            {
                for (int d = 0; d < dim; d++)
                {
                    result[b * totalDim + offset + d] = embeddings[p][b * dim + d];
                }
            }

            offset += dim;
        }

        return result;
    }

    private static Tensor<T> SumEmbeddings(IReadOnlyList<Tensor<T>> embeddings)
    {
        int batchSize = embeddings[0].Shape[0];
        int dim = embeddings[0].Rank > 1 ? embeddings[0].Shape[1] : embeddings[0].Shape[0];

        var result = new Tensor<T>(new[] { batchSize, dim });
        for (int p = 0; p < embeddings.Count; p++)
        {
            for (int i = 0; i < batchSize * dim; i++)
            {
                double val = NumOps.ToDouble(result[i]) + NumOps.ToDouble(embeddings[p][i]);
                result[i] = NumOps.FromDouble(val);
            }
        }

        return result;
    }

    private Tensor<T> AttentionEmbeddings(IReadOnlyList<Tensor<T>> embeddings)
    {
        int batchSize = embeddings[0].Shape[0];
        int dim = embeddings[0].Rank > 1 ? embeddings[0].Shape[1] : embeddings[0].Shape[0];

        // Softmax over attention weights
        var weights = new double[embeddings.Count];
        double maxWeight = double.MinValue;
        for (int p = 0; p < embeddings.Count; p++)
        {
            weights[p] = _attentionWeights is not null
                ? NumOps.ToDouble(_attentionWeights[p])
                : 1.0 / embeddings.Count;
            if (weights[p] > maxWeight) maxWeight = weights[p];
        }

        double sumExp = 0.0;
        for (int p = 0; p < embeddings.Count; p++)
        {
            weights[p] = Math.Exp(weights[p] - maxWeight);
            sumExp += weights[p];
        }

        for (int p = 0; p < embeddings.Count; p++)
        {
            weights[p] /= sumExp;
        }

        // Weighted sum
        var result = new Tensor<T>(new[] { batchSize, dim });
        for (int p = 0; p < embeddings.Count; p++)
        {
            for (int i = 0; i < batchSize * dim; i++)
            {
                double val = NumOps.ToDouble(result[i]) + weights[p] * NumOps.ToDouble(embeddings[p][i]);
                result[i] = NumOps.FromDouble(val);
            }
        }

        return result;
    }

    private Tensor<T> GatingEmbeddings(IReadOnlyList<Tensor<T>> embeddings)
    {
        int batchSize = embeddings[0].Shape[0];
        int dim = embeddings[0].Rank > 1 ? embeddings[0].Shape[1] : embeddings[0].Shape[0];

        var result = new Tensor<T>(new[] { batchSize, dim });
        for (int p = 0; p < embeddings.Count; p++)
        {
            double rawWeight = _attentionWeights is not null
                ? NumOps.ToDouble(_attentionWeights[p])
                : 0.0;
            // Sigmoid gate
            double gate = 1.0 / (1.0 + Math.Exp(-rawWeight));

            for (int i = 0; i < batchSize * dim; i++)
            {
                double val = NumOps.ToDouble(result[i]) + gate * NumOps.ToDouble(embeddings[p][i]);
                result[i] = NumOps.FromDouble(val);
            }
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
            int offset = 0;
            for (int p = 0; p < partyEmbeddings.Count; p++)
            {
                int dim = partyEmbeddings[p].Rank > 1 ? partyEmbeddings[p].Shape[1] : partyEmbeddings[p].Shape[0];
                var grad = new Tensor<T>(new[] { batchSize, dim });
                for (int b = 0; b < batchSize; b++)
                {
                    for (int d = 0; d < dim; d++)
                    {
                        grad[b * dim + d] = gradInput[b * totalDim + offset + d];
                    }
                }

                gradients.Add(grad);
                offset += dim;
            }
        }
        else
        {
            // For Sum/Attention/Gating, each party gets the full gradient
            // (scaled by its weight for Attention/Gating)
            for (int p = 0; p < partyEmbeddings.Count; p++)
            {
                int dim = partyEmbeddings[p].Rank > 1 ? partyEmbeddings[p].Shape[1] : partyEmbeddings[p].Shape[0];
                var grad = new Tensor<T>(new[] { batchSize, dim });

                double scale = 1.0;
                if (_options.AggregationMode == VflAggregationMode.Attention && _attentionWeights is not null)
                {
                    scale = NumOps.ToDouble(_attentionWeights[p]);
                }
                else if (_options.AggregationMode == VflAggregationMode.Gating && _attentionWeights is not null)
                {
                    double rawWeight = NumOps.ToDouble(_attentionWeights[p]);
                    scale = 1.0 / (1.0 + Math.Exp(-rawWeight));
                }

                for (int i = 0; i < batchSize * dim; i++)
                {
                    grad[i] = NumOps.FromDouble(NumOps.ToDouble(gradInput[i]) * scale);
                }

                gradients.Add(grad);
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
                result[d] = NumOps.FromDouble(NumOps.ToDouble(result[d]) + NumOps.ToDouble(tensor[b * dim + d]));
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
            param[i] = NumOps.FromDouble(NumOps.ToDouble(param[i]) - lr * NumOps.ToDouble(grad[i]));
        }
    }
}
