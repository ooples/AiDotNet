using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.FederatedLearning.Infrastructure;
using AiDotNet.Tensors;

namespace AiDotNet.FederatedLearning.Graph;

/// <summary>
/// Generates pseudo-node features for missing cross-client neighbors using a learned model.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> When a GNN on Client A needs features from a node on Client B (a
/// cross-client neighbor), it can't directly access them. This generator learns to produce realistic
/// pseudo-node features based on the local graph structure:</para>
///
/// <list type="number">
/// <item><description><b>Input:</b> Features of the border node and its known neighbors.</description></item>
/// <item><description><b>Model:</b> A small MLP that maps local structure to predicted neighbor features.</description></item>
/// <item><description><b>Output:</b> Synthetic feature vector for the missing neighbor.</description></item>
/// </list>
///
/// <para><b>Training:</b> The generator is trained locally on each client using known edges —
/// it learns to predict a node's features from its neighbors' features. Then it's used to predict
/// features for missing (cross-client) neighbors.</para>
///
/// <para><b>Homophily assumption:</b> Works best on homophilic graphs where connected nodes tend to
/// have similar features (social networks, citation networks). Less effective on heterophilic graphs.</para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ComponentType(ComponentType.FederatedAggregator)]
[PipelineStage(PipelineStage.Training)]
public partial class GraphNodeGenerator<T> : FederatedLearningComponentBase<T>
{
    private readonly int _inputDim;
    private readonly int _hiddenDim;
    private readonly int _outputDim;
    private readonly double _learningRate;

    // Simple two-layer MLP: input -> hidden -> output
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T> _weightsHidden;
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T> _biasHidden;
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T> _weightsOutput;
    [AiDotNet.Attributes.TrainableParameter]
    private Tensor<T> _biasOutput;
    private bool _trained;

    /// <summary>
    /// Initializes a new instance of <see cref="GraphNodeGenerator{T}"/>.
    /// </summary>
    /// <param name="featureDim">Dimensionality of node features.</param>
    /// <param name="hiddenDim">Hidden layer dimension. Default 64.</param>
    /// <param name="learningRate">Learning rate for generator training. Default 0.01.</param>
    public GraphNodeGenerator(int featureDim, int hiddenDim = 64, double learningRate = 0.01)
    {
        if (featureDim <= 0) throw new ArgumentOutOfRangeException(nameof(featureDim));

        _inputDim = featureDim;
        _hiddenDim = hiddenDim;
        _outputDim = featureDim; // Output same dimension as input features
        _learningRate = learningRate;

        // Xavier initialization
        var rng = AiDotNet.Tensors.Helpers.RandomHelper.CreateSecureRandom();
        _weightsHidden = InitializeWeights(rng, _inputDim, _hiddenDim);
        _biasHidden = new Tensor<T>(new[] { _hiddenDim });
        _weightsOutput = InitializeWeights(rng, _hiddenDim, _outputDim);
        _biasOutput = new Tensor<T>(new[] { _outputDim });
    }

    /// <summary>
    /// Gets a value indicating whether the generator has been trained.
    /// </summary>
    public bool IsTrained => _trained;

    /// <summary>
    /// Trains the generator on known node-neighbor pairs from the local subgraph.
    /// </summary>
    /// <param name="adjacency">Local adjacency matrix (flattened NxN).</param>
    /// <param name="nodeFeatures">Node features (flattened NxF).</param>
    /// <param name="featureDim">Feature dimensionality.</param>
    /// <param name="epochs">Training epochs. Default 10.</param>
    public void Train(Tensor<T> adjacency, Tensor<T> nodeFeatures, int featureDim, int epochs = 10)
    {
        if (adjacency is null) throw new ArgumentNullException(nameof(adjacency));
        if (nodeFeatures is null) throw new ArgumentNullException(nameof(nodeFeatures));
        if (featureDim <= 0) throw new ArgumentOutOfRangeException(nameof(featureDim), "Feature dimension must be positive.");

        int numNodes = (int)Math.Sqrt(adjacency.Shape[0]);

        // Training pairs: (mean features of a node's neighbors, the node's own features) for every node with
        // at least one neighbor.
        var (inputs, targets, targetMask) = BuildTrainingPairs(adjacency, nodeFeatures, numNodes, featureDim);
        if (inputs is not null && targets is not null && targetMask is not null)
        {
            int batch = inputs.Shape[0];

            // MSE per node is (1/F) Σ_d (pred_d - target_d)^2; full-batch gradient descent on its mean.
            var gradScale = NumOps.FromDouble(2.0 / ((double)featureDim * batch));
            var learningRate = NumOps.FromDouble(_learningRate);
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                // Forward pass: predict each node's features from its neighbors
                var (hidden, predicted) = ForwardBatch(inputs);

                // Backward pass and update
                var gradOutput = Engine.TensorMultiplyScalar(
                    Engine.TensorMultiply(Engine.TensorSubtract(predicted, targets), targetMask), gradScale);
                var weightsOutput = _weightsOutput.Reshape(new[] { _hiddenDim, _outputDim });
                var gradWeightsOutput = Engine.TensorMatMul(Engine.TensorTranspose(hidden), gradOutput);
                var gradBiasOutput = Engine.ReduceSum(gradOutput, new[] { 0 }, false);

                var gradHidden = Engine.TensorMultiply(
                    Engine.TensorMatMul(gradOutput, Engine.TensorTranspose(weightsOutput)),
                    Engine.TensorGreaterThan(hidden, NumOps.Zero));
                var gradWeightsHidden = Engine.TensorMatMul(Engine.TensorTranspose(inputs), gradHidden);
                var gradBiasHidden = Engine.ReduceSum(gradHidden, new[] { 0 }, false);

                _weightsOutput = Step(_weightsOutput, gradWeightsOutput, learningRate);
                _biasOutput = Step(_biasOutput, gradBiasOutput, learningRate);
                _weightsHidden = Step(_weightsHidden, gradWeightsHidden, learningRate);
                _biasHidden = Step(_biasHidden, gradBiasHidden, learningRate);
            }
        }

        _trained = true;
    }

    /// <summary>
    /// Generates a pseudo-node feature vector for a missing cross-client neighbor.
    /// </summary>
    /// <param name="borderNodeFeatures">Features of the border node and its known local neighbors
    /// (mean-aggregated, dimension = featureDim).</param>
    /// <returns>Generated pseudo-node feature vector.</returns>
    public Tensor<T> Generate(Tensor<T> borderNodeFeatures)
    {
        if (borderNodeFeatures is null) throw new ArgumentNullException(nameof(borderNodeFeatures));
        return Forward(borderNodeFeatures);
    }

    private Tensor<T> Forward(Tensor<T> input)
    {
        var row = FitWidth(input.Reshape(new[] { 1, input.Length }), _inputDim);
        return ForwardBatch(row).Output.Reshape(new[] { _outputDim });
    }

    /// <summary>
    /// Hidden layer ReLU(X W_h + b_h) and output layer H W_o + b_o for a batch of rows.
    /// </summary>
    private (Tensor<T> Hidden, Tensor<T> Output) ForwardBatch(Tensor<T> inputs)
    {
        int batch = inputs.Shape[0];
        var hidden = Engine.TensorReLU(Engine.TensorAdd(
            Engine.TensorMatMul(inputs, _weightsHidden.Reshape(new[] { _inputDim, _hiddenDim })),
            Engine.TensorBroadcastTo(_biasHidden.Reshape(new[] { 1, _hiddenDim }), new[] { batch, _hiddenDim })));
        var output = Engine.TensorAdd(
            Engine.TensorMatMul(hidden, _weightsOutput.Reshape(new[] { _hiddenDim, _outputDim })),
            Engine.TensorBroadcastTo(_biasOutput.Reshape(new[] { 1, _outputDim }), new[] { batch, _outputDim }));
        return (hidden, output);
    }

    private Tensor<T> Step(Tensor<T> parameter, Tensor<T> gradient, T learningRate)
    {
        return Engine.TensorSubtract(parameter,
            Engine.TensorMultiplyScalar(gradient.Reshape(parameter.Shape.ToArray()), learningRate));
    }

    /// <summary>
    /// Truncates or zero-pads the columns of a [rows, width] tensor to <paramref name="width"/>.
    /// </summary>
    private Tensor<T> FitWidth(Tensor<T> rows, int width)
    {
        int count = rows.Shape[0];
        int current = rows.Shape[1];
        if (current == width)
        {
            return rows;
        }

        if (current > width)
        {
            return Engine.TensorSlice(rows, new[] { 0, 0 }, new[] { count, width });
        }

        return Engine.TensorConcatenate(new[] { rows, new Tensor<T>(new[] { count, width - current }) }, 1);
    }

    private (Tensor<T>? Inputs, Tensor<T>? Targets, Tensor<T>? TargetMask) BuildTrainingPairs(
        Tensor<T> adjacency, Tensor<T> nodeFeatures, int numNodes, int featureDim)
    {
        if (numNodes == 0)
        {
            return (null, null, null);
        }

        // Feature matrix [N, F]; entries past the end of the feature tensor are zero and masked out as targets.
        int featureCells = numNodes * featureDim;
        int available = Math.Min(featureCells, nodeFeatures.Shape[0]);
        var flatFeatures = nodeFeatures.Reshape(new[] { nodeFeatures.Length });
        var features = available == featureCells
            ? Engine.TensorSlice(flatFeatures, new[] { 0 }, new[] { featureCells })
            : Engine.TensorConcatenate(new[]
            {
                Engine.TensorSlice(flatFeatures, new[] { 0 }, new[] { available }),
                new Tensor<T>(new[] { featureCells - available }),
            }, 0);
        features = features.Reshape(new[] { numNodes, featureDim });

        // Neighbor indicator without self loops; mean neighbor features = (A F) / degree.
        var offDiagonal = new int[numNodes * numNodes];
        for (int i = 0; i < numNodes; i++)
        {
            for (int j = 0; j < numNodes; j++)
            {
                offDiagonal[i * numNodes + j] = i == j ? 0 : 1;
            }
        }

        var adjacencyCells = Engine.TensorSlice(adjacency.Reshape(new[] { adjacency.Length }), new[] { 0 }, new[] { numNodes * numNodes })
            .Reshape(new[] { numNodes, numNodes });
        var neighborMask = Engine.TensorMultiply(
            Engine.TensorGreaterThan(adjacencyCells, NumOps.Zero),
            new Tensor<int>(offDiagonal, new[] { numNodes, numNodes }).Cast<T>());
        var degrees = Engine.ReduceSum(neighborMask, new[] { 1 }, true); // [N, 1]

        var degreeCounts = new int[numNodes];
        var degreeValues = degrees.Cast<double>().ToArray();
        var withNeighbors = new List<int>();
        for (int i = 0; i < numNodes; i++)
        {
            degreeCounts[i] = (int)Math.Round(degreeValues[i]);
            if (degreeCounts[i] > 0)
            {
                withNeighbors.Add(i);
            }
        }

        if (withNeighbors.Count == 0)
        {
            return (null, null, null);
        }

        var selected = new Tensor<int>(withNeighbors.ToArray(), new[] { withNeighbors.Count });
        var sums = Engine.TensorGather(Engine.TensorMatMul(neighborMask, features), selected, 0);
        var counts = new double[withNeighbors.Count];
        for (int k = 0; k < counts.Length; k++)
        {
            counts[k] = degreeCounts[withNeighbors[k]];
        }

        var countTensor = new Tensor<double>(counts, new[] { counts.Length, 1 }).Cast<T>();
        var means = Engine.TensorDivide(sums, Engine.TensorBroadcastTo(countTensor, new[] { counts.Length, featureDim }));

        // Targets: the node's own first min(F, outputDim) features, masked where the feature tensor ended.
        int targetWidth = Math.Min(featureDim, _outputDim);
        var cellValid = new int[withNeighbors.Count * _outputDim];
        for (int k = 0; k < withNeighbors.Count; k++)
        {
            for (int d = 0; d < targetWidth; d++)
            {
                cellValid[k * _outputDim + d] = withNeighbors[k] * featureDim + d < nodeFeatures.Shape[0] ? 1 : 0;
            }
        }

        var targets = FitWidth(Engine.TensorGather(features, selected, 0), _outputDim);
        var targetMask = new Tensor<int>(cellValid, new[] { withNeighbors.Count, _outputDim }).Cast<T>();
        return (FitWidth(means, _inputDim), targets, targetMask);
    }

    private Tensor<T> InitializeWeights(Random rng, int inputSize, int outputSize)
    {
        int size = inputSize * outputSize;
        double scale = Math.Sqrt(2.0 / (inputSize + outputSize));

        var values = new double[size];
        for (int i = 0; i < size; i++)
        {
            values[i] = (rng.NextDouble() * 2 - 1) * scale;
        }

        return new Tensor<double>(values, new[] { size }).Cast<T>();
    }
}
