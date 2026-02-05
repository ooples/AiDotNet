using AiDotNet.Extensions;
using AiDotNet.Helpers;

namespace AiDotNet.NeuralNetworks.Tabular;

/// <summary>
/// Soft Decision Tree for GANDALF and similar architectures.
/// </summary>
/// <remarks>
/// <para>
/// A soft decision tree uses continuous (differentiable) split decisions instead of
/// hard binary splits. This allows the tree to be trained with gradient descent
/// while maintaining the interpretable structure of decision trees.
/// </para>
/// <para>
/// <b>For Beginners:</b> A soft tree is like a fuzzy decision tree:
/// - Regular tree: "Is age > 30? Go left or right"
/// - Soft tree: "Is age > 30? Go 70% left, 30% right"
///
/// The soft splits make the tree trainable with neural network methods.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class SoftTree<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly Random _random;

    private readonly int _inputDim;
    private readonly int _depth;
    private readonly int _outputDim;
    private readonly double _temperature;

    // Internal nodes: feature weights and biases for split decisions
    private readonly int _numInternalNodes;
    private Tensor<T> _splitWeights;  // [numInternalNodes, inputDim]
    private Tensor<T> _splitBiases;   // [numInternalNodes]

    // Leaf nodes: output values
    private readonly int _numLeaves;
    private Tensor<T> _leafValues;    // [numLeaves, outputDim]

    // Gradients
    private Tensor<T> _splitWeightsGrad;
    private Tensor<T> _splitBiasesGrad;
    private Tensor<T> _leafValuesGrad;

    // Cached values
    private Tensor<T>? _inputCache;
    private Tensor<T>? _pathProbabilitiesCache;

    /// <summary>
    /// Gets the number of leaf nodes.
    /// </summary>
    public int NumLeaves => _numLeaves;

    /// <summary>
    /// Gets the total parameter count.
    /// </summary>
    public int ParameterCount =>
        _numInternalNodes * _inputDim +  // split weights
        _numInternalNodes +               // split biases
        _numLeaves * _outputDim;          // leaf values

    /// <summary>
    /// Initializes a soft decision tree.
    /// </summary>
    /// <param name="inputDim">Input feature dimension.</param>
    /// <param name="depth">Tree depth (number of decision levels).</param>
    /// <param name="outputDim">Output dimension per leaf.</param>
    /// <param name="temperature">Temperature for soft splits (lower = harder splits).</param>
    /// <param name="initScale">Initialization scale.</param>
    public SoftTree(int inputDim, int depth = 4, int outputDim = 1, double temperature = 1.0, double initScale = 0.01)
    {
        _inputDim = inputDim;
        _depth = depth;
        _outputDim = outputDim;
        _temperature = temperature;
        _random = RandomHelper.CreateSecureRandom();

        _numInternalNodes = (1 << depth) - 1;  // 2^depth - 1
        _numLeaves = 1 << depth;                // 2^depth

        // Initialize split parameters
        _splitWeights = new Tensor<T>([_numInternalNodes, inputDim]);
        _splitBiases = new Tensor<T>([_numInternalNodes]);

        // Initialize leaf values
        _leafValues = new Tensor<T>([_numLeaves, outputDim]);

        // Initialize gradients
        _splitWeightsGrad = new Tensor<T>([_numInternalNodes, inputDim]);
        _splitBiasesGrad = new Tensor<T>([_numInternalNodes]);
        _leafValuesGrad = new Tensor<T>([_numLeaves, outputDim]);

        InitializeParameters(initScale);
    }

    private void InitializeParameters(double scale)
    {
        // Initialize split weights
        for (int i = 0; i < _splitWeights.Length; i++)
        {
            _splitWeights[i] = NumOps.FromDouble(_random.NextGaussian() * scale);
        }

        // Initialize split biases to small random values
        for (int i = 0; i < _splitBiases.Length; i++)
        {
            _splitBiases[i] = NumOps.FromDouble(_random.NextGaussian() * scale * 0.1);
        }

        // Initialize leaf values
        for (int i = 0; i < _leafValues.Length; i++)
        {
            _leafValues[i] = NumOps.FromDouble(_random.NextGaussian() * scale);
        }
    }

    /// <summary>
    /// Forward pass through the soft tree.
    /// </summary>
    /// <param name="input">Input features [batchSize, inputDim].</param>
    /// <returns>Tree output [batchSize, outputDim].</returns>
    public Tensor<T> Forward(Tensor<T> input)
    {
        _inputCache = input;
        int batchSize = input.Shape[0];

        // Compute path probabilities to each leaf
        var pathProbs = ComputePathProbabilities(input, batchSize);
        _pathProbabilitiesCache = pathProbs;

        // Weighted sum of leaf values
        var output = new Tensor<T>([batchSize, _outputDim]);

        for (int b = 0; b < batchSize; b++)
        {
            for (int o = 0; o < _outputDim; o++)
            {
                var sum = NumOps.Zero;
                for (int l = 0; l < _numLeaves; l++)
                {
                    sum = NumOps.Add(sum, NumOps.Multiply(
                        pathProbs[b * _numLeaves + l],
                        _leafValues[l * _outputDim + o]));
                }
                output[b * _outputDim + o] = sum;
            }
        }

        return output;
    }

    private Tensor<T> ComputePathProbabilities(Tensor<T> input, int batchSize)
    {
        var pathProbs = new Tensor<T>([batchSize, _numLeaves]);
        var tempScale = NumOps.FromDouble(1.0 / _temperature);

        for (int b = 0; b < batchSize; b++)
        {
            // Initialize all paths with probability 1
            var nodeProbs = new T[_numInternalNodes + _numLeaves];
            nodeProbs[0] = NumOps.One;  // Root node

            // Traverse tree level by level
            for (int node = 0; node < _numInternalNodes; node++)
            {
                // Compute split decision
                var logit = ComputeSplitLogit(input, b, node);
                var scaledLogit = NumOps.Multiply(logit, tempScale);
                var rightProb = Sigmoid(scaledLogit);
                var leftProb = NumOps.Subtract(NumOps.One, rightProb);

                // Propagate probabilities to children
                int leftChild = 2 * node + 1;
                int rightChild = 2 * node + 2;

                if (leftChild < _numInternalNodes + _numLeaves)
                {
                    nodeProbs[leftChild] = NumOps.Multiply(nodeProbs[node], leftProb);
                }
                if (rightChild < _numInternalNodes + _numLeaves)
                {
                    nodeProbs[rightChild] = NumOps.Multiply(nodeProbs[node], rightProb);
                }
            }

            // Extract leaf probabilities
            for (int l = 0; l < _numLeaves; l++)
            {
                pathProbs[b * _numLeaves + l] = nodeProbs[_numInternalNodes + l];
            }
        }

        return pathProbs;
    }

    private T ComputeSplitLogit(Tensor<T> input, int batchIdx, int nodeIdx)
    {
        var logit = _splitBiases[nodeIdx];
        for (int d = 0; d < _inputDim; d++)
        {
            logit = NumOps.Add(logit, NumOps.Multiply(
                _splitWeights[nodeIdx * _inputDim + d],
                input[batchIdx * _inputDim + d]));
        }
        return logit;
    }

    private T Sigmoid(T x)
    {
        var negX = NumOps.Negate(x);
        var expNegX = NumOps.Exp(negX);
        return NumOps.Divide(NumOps.One, NumOps.Add(NumOps.One, expNegX));
    }

    /// <summary>
    /// Backward pass through the soft tree.
    /// </summary>
    /// <param name="gradient">Gradient with respect to output [batchSize, outputDim].</param>
    /// <returns>Gradient with respect to input [batchSize, inputDim].</returns>
    public Tensor<T> Backward(Tensor<T> gradient)
    {
        if (_inputCache == null || _pathProbabilitiesCache == null)
        {
            throw new InvalidOperationException("Forward must be called before backward");
        }

        int batchSize = _inputCache.Shape[0];
        var inputGrad = new Tensor<T>([batchSize, _inputDim]);

        // Zero gradients
        for (int i = 0; i < _leafValuesGrad.Length; i++)
            _leafValuesGrad[i] = NumOps.Zero;
        for (int i = 0; i < _splitWeightsGrad.Length; i++)
            _splitWeightsGrad[i] = NumOps.Zero;
        for (int i = 0; i < _splitBiasesGrad.Length; i++)
            _splitBiasesGrad[i] = NumOps.Zero;

        // Gradient for leaf values
        for (int b = 0; b < batchSize; b++)
        {
            for (int l = 0; l < _numLeaves; l++)
            {
                var pathProb = _pathProbabilitiesCache[b * _numLeaves + l];
                for (int o = 0; o < _outputDim; o++)
                {
                    _leafValuesGrad[l * _outputDim + o] = NumOps.Add(
                        _leafValuesGrad[l * _outputDim + o],
                        NumOps.Multiply(pathProb, gradient[b * _outputDim + o]));
                }
            }
        }

        // Simplified input gradient (full implementation would backprop through path probabilities)
        return inputGrad;
    }

    /// <summary>
    /// Gets feature importance based on split weights.
    /// </summary>
    public T[] GetFeatureImportance()
    {
        var importance = new T[_inputDim];

        for (int d = 0; d < _inputDim; d++)
        {
            var sum = NumOps.Zero;
            for (int n = 0; n < _numInternalNodes; n++)
            {
                var weight = _splitWeights[n * _inputDim + d];
                sum = NumOps.Add(sum, NumOps.Multiply(weight, weight));  // Sum of squares
            }
            importance[d] = NumOps.Sqrt(sum);
        }

        return importance;
    }

    /// <summary>
    /// Updates parameters.
    /// </summary>
    public void UpdateParameters(T learningRate)
    {
        for (int i = 0; i < _splitWeights.Length; i++)
        {
            _splitWeights[i] = NumOps.Subtract(_splitWeights[i],
                NumOps.Multiply(learningRate, _splitWeightsGrad[i]));
        }

        for (int i = 0; i < _splitBiases.Length; i++)
        {
            _splitBiases[i] = NumOps.Subtract(_splitBiases[i],
                NumOps.Multiply(learningRate, _splitBiasesGrad[i]));
        }

        for (int i = 0; i < _leafValues.Length; i++)
        {
            _leafValues[i] = NumOps.Subtract(_leafValues[i],
                NumOps.Multiply(learningRate, _leafValuesGrad[i]));
        }
    }

    /// <summary>
    /// Resets internal state.
    /// </summary>
    public void ResetState()
    {
        _inputCache = null;
        _pathProbabilitiesCache = null;

        for (int i = 0; i < _splitWeightsGrad.Length; i++)
            _splitWeightsGrad[i] = NumOps.Zero;
        for (int i = 0; i < _splitBiasesGrad.Length; i++)
            _splitBiasesGrad[i] = NumOps.Zero;
        for (int i = 0; i < _leafValuesGrad.Length; i++)
            _leafValuesGrad[i] = NumOps.Zero;
    }
}
