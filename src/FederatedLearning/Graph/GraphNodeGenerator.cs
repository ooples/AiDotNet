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
public class GraphNodeGenerator<T> : FederatedLearningComponentBase<T>
{
    private readonly int _inputDim;
    private readonly int _hiddenDim;
    private readonly int _outputDim;
    private readonly double _learningRate;

    // Simple two-layer MLP: input -> hidden -> output
    private Tensor<T> _weightsHidden;
    private Tensor<T> _biasHidden;
    private Tensor<T> _weightsOutput;
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

        int numNodes = (int)Math.Sqrt(adjacency.Shape[0]);

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            double totalLoss = 0;
            int sampleCount = 0;

            for (int i = 0; i < numNodes; i++)
            {
                // Compute mean neighbor features as input
                var meanNeighborFeatures = ComputeMeanNeighborFeatures(
                    adjacency, nodeFeatures, i, numNodes, featureDim);

                if (meanNeighborFeatures is null) continue;

                // Forward pass: predict node i's features from its neighbors
                var predicted = Forward(meanNeighborFeatures);

                // Compute MSE loss
                double loss = 0;
                var gradOutput = new double[_outputDim];

                for (int d = 0; d < featureDim && d < _outputDim; d++)
                {
                    int targetIdx = i * featureDim + d;
                    if (targetIdx >= nodeFeatures.Shape[0]) break;

                    double target = NumOps.ToDouble(nodeFeatures[targetIdx]);
                    double pred = NumOps.ToDouble(predicted[d]);
                    double diff = pred - target;
                    loss += diff * diff;
                    gradOutput[d] = 2.0 * diff / featureDim;
                }

                totalLoss += loss / featureDim;
                sampleCount++;

                // Backward pass and update
                Backward(meanNeighborFeatures, gradOutput);
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
        // Hidden layer: ReLU(input * W_h + b_h)
        var hidden = new Tensor<T>(new[] { _hiddenDim });
        for (int h = 0; h < _hiddenDim; h++)
        {
            double sum = NumOps.ToDouble(_biasHidden[h]);
            for (int i = 0; i < _inputDim && i < input.Shape[0]; i++)
            {
                int wIdx = i * _hiddenDim + h;
                if (wIdx < _weightsHidden.Shape[0])
                {
                    sum += NumOps.ToDouble(input[i]) * NumOps.ToDouble(_weightsHidden[wIdx]);
                }
            }

            hidden[h] = NumOps.FromDouble(Math.Max(0, sum)); // ReLU
        }

        // Output layer: hidden * W_o + b_o
        var output = new Tensor<T>(new[] { _outputDim });
        for (int o = 0; o < _outputDim; o++)
        {
            double sum = NumOps.ToDouble(_biasOutput[o]);
            for (int h = 0; h < _hiddenDim; h++)
            {
                int wIdx = h * _outputDim + o;
                if (wIdx < _weightsOutput.Shape[0])
                {
                    sum += NumOps.ToDouble(hidden[h]) * NumOps.ToDouble(_weightsOutput[wIdx]);
                }
            }

            output[o] = NumOps.FromDouble(sum);
        }

        return output;
    }

    private void Backward(Tensor<T> input, double[] gradOutput)
    {
        // Recompute hidden activations (needed for backward)
        var hiddenPre = new double[_hiddenDim];
        var hiddenPost = new double[_hiddenDim];

        for (int h = 0; h < _hiddenDim; h++)
        {
            double sum = NumOps.ToDouble(_biasHidden[h]);
            for (int i = 0; i < _inputDim && i < input.Shape[0]; i++)
            {
                int wIdx = i * _hiddenDim + h;
                if (wIdx < _weightsHidden.Shape[0])
                {
                    sum += NumOps.ToDouble(input[i]) * NumOps.ToDouble(_weightsHidden[wIdx]);
                }
            }

            hiddenPre[h] = sum;
            hiddenPost[h] = Math.Max(0, sum);
        }

        // Gradient through output layer
        var gradHidden = new double[_hiddenDim];

        for (int o = 0; o < _outputDim; o++)
        {
            double go = gradOutput[o];

            // Update output weights and bias
            double biasVal = NumOps.ToDouble(_biasOutput[o]);
            _biasOutput[o] = NumOps.FromDouble(biasVal - _learningRate * go);

            for (int h = 0; h < _hiddenDim; h++)
            {
                int wIdx = h * _outputDim + o;
                if (wIdx < _weightsOutput.Shape[0])
                {
                    double w = NumOps.ToDouble(_weightsOutput[wIdx]);
                    gradHidden[h] += go * w;
                    _weightsOutput[wIdx] = NumOps.FromDouble(w - _learningRate * go * hiddenPost[h]);
                }
            }
        }

        // Gradient through ReLU
        for (int h = 0; h < _hiddenDim; h++)
        {
            if (hiddenPre[h] <= 0) gradHidden[h] = 0; // ReLU derivative
        }

        // Update hidden weights and bias
        for (int h = 0; h < _hiddenDim; h++)
        {
            double gh = gradHidden[h];
            double biasVal = NumOps.ToDouble(_biasHidden[h]);
            _biasHidden[h] = NumOps.FromDouble(biasVal - _learningRate * gh);

            for (int i = 0; i < _inputDim && i < input.Shape[0]; i++)
            {
                int wIdx = i * _hiddenDim + h;
                if (wIdx < _weightsHidden.Shape[0])
                {
                    double w = NumOps.ToDouble(_weightsHidden[wIdx]);
                    double inputVal = NumOps.ToDouble(input[i]);
                    _weightsHidden[wIdx] = NumOps.FromDouble(w - _learningRate * gh * inputVal);
                }
            }
        }
    }

    private Tensor<T>? ComputeMeanNeighborFeatures(
        Tensor<T> adjacency, Tensor<T> nodeFeatures,
        int nodeIdx, int numNodes, int featureDim)
    {
        var mean = new double[featureDim];
        int neighborCount = 0;

        for (int j = 0; j < numNodes; j++)
        {
            if (j == nodeIdx) continue;

            int adjIdx = nodeIdx * numNodes + j;
            if (adjIdx < adjacency.Shape[0] && NumOps.ToDouble(adjacency[adjIdx]) > 0)
            {
                neighborCount++;
                for (int d = 0; d < featureDim; d++)
                {
                    int fIdx = j * featureDim + d;
                    if (fIdx < nodeFeatures.Shape[0])
                    {
                        mean[d] += NumOps.ToDouble(nodeFeatures[fIdx]);
                    }
                }
            }
        }

        if (neighborCount == 0) return null;

        for (int d = 0; d < featureDim; d++)
        {
            mean[d] /= neighborCount;
        }

        var result = new Tensor<T>(new[] { featureDim });
        for (int d = 0; d < featureDim; d++)
        {
            result[d] = NumOps.FromDouble(mean[d]);
        }

        return result;
    }

    private Tensor<T> InitializeWeights(Random rng, int inputSize, int outputSize)
    {
        int size = inputSize * outputSize;
        var weights = new Tensor<T>(new[] { size });
        double scale = Math.Sqrt(2.0 / (inputSize + outputSize));

        for (int i = 0; i < size; i++)
        {
            weights[i] = NumOps.FromDouble((rng.NextDouble() * 2 - 1) * scale);
        }

        return weights;
    }
}
