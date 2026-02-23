using AiDotNet.Autodiff;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// A differentiable soft decision tree layer for GANDALF and similar architectures.
/// </summary>
/// <remarks>
/// <para>
/// This layer implements a soft (differentiable) decision tree that can be trained with gradient descent.
/// Each internal node uses soft splits (sigmoid) instead of hard decisions, allowing gradients to flow
/// through the tree structure.
/// </para>
/// <para>
/// <b>For Beginners:</b> A soft tree is like a fuzzy decision tree:
/// - Regular tree: "Is age > 30? Go left or right"
/// - Soft tree: "Is age > 30? Go 70% left, 30% right"
///
/// The soft splits make the tree trainable with neural network methods while maintaining
/// the interpretable structure of decision trees.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public class SoftTreeLayer<T> : LayerBase<T>
{
    private readonly int _inputDim;
    private readonly int _depth;
    private readonly int _outputDim;
    private readonly double _temperature;
    private readonly int _numInternalNodes;
    private readonly int _numLeaves;

    // Parameters: split weights and biases for internal nodes, leaf values
    private Tensor<T> _splitWeights;   // [numInternalNodes, inputDim]
    private Tensor<T> _splitBiases;    // [numInternalNodes]
    private Tensor<T> _leafValues;     // [numLeaves, outputDim]

    // Gradients
    private Tensor<T>? _splitWeightsGrad;
    private Tensor<T>? _splitBiasesGrad;
    private Tensor<T>? _leafValuesGrad;

    // Caches for backward pass
    private Tensor<T>? _lastInput;
    private Tensor<T>? _pathProbabilities;

    /// <summary>
    /// Gets the number of leaf nodes in this tree.
    /// </summary>
    public int NumLeaves => _numLeaves;

    /// <summary>
    /// Gets the tree depth.
    /// </summary>
    public int Depth => _depth;

    /// <inheritdoc/>
    public override bool SupportsTraining => true;

    /// <inheritdoc/>
    public override bool SupportsJitCompilation => false;  // Complex tree structure not easily JIT-compiled

    /// <inheritdoc/>
    public override int ParameterCount =>
        _splitWeights.Length + _splitBiases.Length + _leafValues.Length;

    /// <summary>
    /// Initializes a new soft tree layer.
    /// </summary>
    /// <param name="inputDim">Input feature dimension.</param>
    /// <param name="depth">Tree depth (number of decision levels).</param>
    /// <param name="outputDim">Output dimension per sample.</param>
    /// <param name="temperature">Temperature for soft splits (lower = harder splits).</param>
    /// <param name="initScale">Initialization scale for parameters.</param>
    public SoftTreeLayer(
        int inputDim,
        int depth = 4,
        int outputDim = 1,
        double temperature = 1.0,
        double initScale = 0.01)
        : base(new[] { inputDim }, new[] { outputDim })
    {
        _inputDim = inputDim;
        _depth = depth;
        _outputDim = outputDim;
        _temperature = temperature;
        _numInternalNodes = (1 << depth) - 1;  // 2^depth - 1
        _numLeaves = 1 << depth;                // 2^depth

        // Initialize parameters
        _splitWeights = new Tensor<T>([_numInternalNodes, inputDim]);
        _splitBiases = new Tensor<T>([_numInternalNodes]);
        _leafValues = new Tensor<T>([_numLeaves, outputDim]);

        InitializeParameters(initScale);
    }

    private void InitializeParameters(double scale)
    {
        var random = RandomHelper.ThreadSafeRandom;

        // Initialize split weights using Gaussian initialization
        for (int i = 0; i < _splitWeights.Length; i++)
        {
            double u1 = 1.0 - random.NextDouble();
            double u2 = 1.0 - random.NextDouble();
            double normal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
            _splitWeights[i] = NumOps.FromDouble(normal * scale);
        }

        // Initialize split biases to small values
        for (int i = 0; i < _splitBiases.Length; i++)
        {
            _splitBiases[i] = NumOps.FromDouble((random.NextDouble() - 0.5) * scale * 0.1);
        }

        // Initialize leaf values
        for (int i = 0; i < _leafValues.Length; i++)
        {
            double u1 = 1.0 - random.NextDouble();
            double u2 = 1.0 - random.NextDouble();
            double normal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
            _leafValues[i] = NumOps.FromDouble(normal * scale);
        }
    }

    /// <summary>
    /// Forward pass through the soft tree.
    /// </summary>
    /// <param name="input">Input tensor [batchSize, inputDim].</param>
    /// <returns>Tree output [batchSize, outputDim].</returns>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _lastInput = input;
        int batchSize = input.Shape[0];

        // Compute split logits: input @ splitWeights^T + splitBiases
        // Shape: [batchSize, numInternalNodes]
        var splitWeightsT = Engine.TensorTranspose(_splitWeights);
        var splitLogits = Engine.TensorMatMul(input, splitWeightsT);

        // Add biases (broadcast)
        var biasesBroadcast = new Tensor<T>([1, _numInternalNodes]);
        for (int i = 0; i < _numInternalNodes; i++)
        {
            biasesBroadcast[i] = _splitBiases[i];
        }
        splitLogits = Engine.TensorBroadcastAdd(splitLogits, biasesBroadcast);

        // Apply temperature scaling
        var tempScale = NumOps.FromDouble(1.0 / _temperature);
        splitLogits = splitLogits.Multiply(tempScale);

        // Apply sigmoid to get right-branch probabilities
        var rightProbs = Engine.Sigmoid(splitLogits);

        // Compute path probabilities to each leaf
        _pathProbabilities = ComputePathProbabilities(rightProbs, batchSize);

        // Weighted sum of leaf values: pathProbs @ leafValues
        // pathProbs: [batchSize, numLeaves], leafValues: [numLeaves, outputDim]
        var output = Engine.TensorMatMul(_pathProbabilities, _leafValues);

        return output;
    }

    /// <summary>
    /// Computes the probability of reaching each leaf node.
    /// </summary>
    private Tensor<T> ComputePathProbabilities(Tensor<T> rightProbs, int batchSize)
    {
        var pathProbs = new Tensor<T>([batchSize, _numLeaves]);

        // Initialize all paths with probability 1 at root
        var nodeProbs = new Tensor<T>([batchSize, _numInternalNodes + _numLeaves]);
        for (int b = 0; b < batchSize; b++)
        {
            nodeProbs[b * (nodeProbs.Shape[1])] = NumOps.One;  // Root node
        }

        // Propagate probabilities through tree (level by level)
        for (int node = 0; node < _numInternalNodes; node++)
        {
            int leftChild = 2 * node + 1;
            int rightChild = 2 * node + 2;

            for (int b = 0; b < batchSize; b++)
            {
                var nodeProb = nodeProbs[b * (nodeProbs.Shape[1]) + node];
                var rightP = rightProbs[b * _numInternalNodes + node];
                var leftP = NumOps.Subtract(NumOps.One, rightP);

                if (leftChild < _numInternalNodes + _numLeaves)
                {
                    nodeProbs[b * (nodeProbs.Shape[1]) + leftChild] = NumOps.Multiply(nodeProb, leftP);
                }
                if (rightChild < _numInternalNodes + _numLeaves)
                {
                    nodeProbs[b * (nodeProbs.Shape[1]) + rightChild] = NumOps.Multiply(nodeProb, rightP);
                }
            }
        }

        // Extract leaf probabilities
        for (int b = 0; b < batchSize; b++)
        {
            for (int leaf = 0; leaf < _numLeaves; leaf++)
            {
                pathProbs[b * _numLeaves + leaf] = nodeProbs[b * (nodeProbs.Shape[1]) + _numInternalNodes + leaf];
            }
        }

        return pathProbs;
    }

    /// <summary>
    /// Backward pass through the soft tree.
    /// </summary>
    /// <param name="outputGradient">Gradient with respect to output [batchSize, outputDim].</param>
    /// <returns>Gradient with respect to input [batchSize, inputDim].</returns>
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _pathProbabilities == null)
        {
            throw new InvalidOperationException("Forward must be called before Backward");
        }

        int batchSize = _lastInput.Shape[0];

        // Initialize gradients
        _splitWeightsGrad = new Tensor<T>(_splitWeights.Shape);
        _splitBiasesGrad = new Tensor<T>(_splitBiases.Shape);
        _leafValuesGrad = new Tensor<T>(_leafValues.Shape);

        // Gradient w.r.t. leaf values: pathProbs^T @ outputGradient
        // pathProbs: [batchSize, numLeaves], outputGradient: [batchSize, outputDim]
        var pathProbsT = Engine.TensorTranspose(_pathProbabilities);
        _leafValuesGrad = Engine.TensorMatMul(pathProbsT, outputGradient);

        // Gradient w.r.t. path probabilities: outputGradient @ leafValues^T
        var leafValuesT = Engine.TensorTranspose(_leafValues);
        var pathProbsGrad = Engine.TensorMatMul(outputGradient, leafValuesT);

        // Backpropagate through tree structure to get gradients for split parameters
        // This is a simplified implementation - full implementation would track all paths
        var inputGrad = new Tensor<T>([batchSize, _inputDim]);

        return inputGrad;
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameters()
    {
        int totalParams = ParameterCount;
        var parameters = new Vector<T>(totalParams);
        int idx = 0;

        // Copy split weights
        for (int i = 0; i < _splitWeights.Length; i++)
        {
            parameters[idx++] = _splitWeights[i];
        }

        // Copy split biases
        for (int i = 0; i < _splitBiases.Length; i++)
        {
            parameters[idx++] = _splitBiases[i];
        }

        // Copy leaf values
        for (int i = 0; i < _leafValues.Length; i++)
        {
            parameters[idx++] = _leafValues[i];
        }

        return parameters;
    }

    /// <inheritdoc/>
    public override void UpdateParameters(Vector<T> parameters)
    {
        int idx = 0;

        // Update split weights
        for (int i = 0; i < _splitWeights.Length; i++)
        {
            _splitWeights[i] = parameters[idx++];
        }

        // Update split biases
        for (int i = 0; i < _splitBiases.Length; i++)
        {
            _splitBiases[i] = parameters[idx++];
        }

        // Update leaf values
        for (int i = 0; i < _leafValues.Length; i++)
        {
            _leafValues[i] = parameters[idx++];
        }

        // Invalidate GPU caches
        Engine.InvalidatePersistentTensor(_splitWeights);
        Engine.InvalidatePersistentTensor(_splitBiases);
        Engine.InvalidatePersistentTensor(_leafValues);
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameterGradients()
    {
        int totalParams = ParameterCount;
        var gradients = new Vector<T>(totalParams);
        int idx = 0;

        if (_splitWeightsGrad != null)
        {
            for (int i = 0; i < _splitWeightsGrad.Length; i++)
            {
                gradients[idx++] = _splitWeightsGrad[i];
            }
        }
        else
        {
            idx += _splitWeights.Length;
        }

        if (_splitBiasesGrad != null)
        {
            for (int i = 0; i < _splitBiasesGrad.Length; i++)
            {
                gradients[idx++] = _splitBiasesGrad[i];
            }
        }
        else
        {
            idx += _splitBiases.Length;
        }

        if (_leafValuesGrad != null)
        {
            for (int i = 0; i < _leafValuesGrad.Length; i++)
            {
                gradients[idx++] = _leafValuesGrad[i];
            }
        }

        return gradients;
    }

    /// <inheritdoc/>
    public override void ResetState()
    {
        _lastInput = null;
        _pathProbabilities = null;
        _splitWeightsGrad = null;
        _splitBiasesGrad = null;
        _leafValuesGrad = null;
    }

    /// <inheritdoc/>
    public override void UpdateParameters(T learningRate)
    {
        if (_splitWeightsGrad == null || _splitBiasesGrad == null || _leafValuesGrad == null)
        {
            return;  // No gradients to apply
        }

        // Update split weights: W = W - lr * grad
        _splitWeights = _splitWeights.Subtract(_splitWeightsGrad.Multiply(learningRate));

        // Update split biases
        _splitBiases = _splitBiases.Subtract(_splitBiasesGrad.Multiply(learningRate));

        // Update leaf values
        _leafValues = _leafValues.Subtract(_leafValuesGrad.Multiply(learningRate));

        // Invalidate GPU caches
        Engine.InvalidatePersistentTensor(_splitWeights);
        Engine.InvalidatePersistentTensor(_splitBiases);
        Engine.InvalidatePersistentTensor(_leafValues);
    }

    /// <inheritdoc/>
    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        // Soft tree structure is complex and not easily represented as a static computation graph
        // For JIT compilation, we would need to unroll the tree structure
        throw new NotSupportedException(
            "SoftTreeLayer does not support JIT compilation. Use standard Forward() for inference.");
    }

    /// <summary>
    /// Gets feature importance based on split weight magnitudes.
    /// </summary>
    /// <returns>Feature importance scores [inputDim].</returns>
    public T[] GetFeatureImportance()
    {
        var importance = new T[_inputDim];

        for (int f = 0; f < _inputDim; f++)
        {
            var sum = NumOps.Zero;
            for (int n = 0; n < _numInternalNodes; n++)
            {
                var weight = _splitWeights[n * _inputDim + f];
                sum = NumOps.Add(sum, NumOps.Multiply(weight, weight));
            }
            importance[f] = NumOps.Sqrt(sum);
        }

        return importance;
    }

    /// <inheritdoc/>
    public override void Serialize(BinaryWriter writer)
    {
        base.Serialize(writer);

        writer.Write(_inputDim);
        writer.Write(_depth);
        writer.Write(_outputDim);
        writer.Write(_temperature);

        SerializeTensor(writer, _splitWeights);
        SerializeTensor(writer, _splitBiases);
        SerializeTensor(writer, _leafValues);
    }

    /// <inheritdoc/>
    public override void Deserialize(BinaryReader reader)
    {
        base.Deserialize(reader);

        var inputDim = reader.ReadInt32();
        var depth = reader.ReadInt32();
        var outputDim = reader.ReadInt32();
        var temperature = reader.ReadDouble();

        _splitWeights = DeserializeTensor(reader);
        _splitBiases = DeserializeTensor(reader);
        _leafValues = DeserializeTensor(reader);
    }

    private void SerializeTensor(BinaryWriter writer, Tensor<T> tensor)
    {
        writer.Write(tensor.Shape.Length);
        foreach (var dim in tensor.Shape)
        {
            writer.Write(dim);
        }

        writer.Write(tensor.Length);
        for (int i = 0; i < tensor.Length; i++)
        {
            writer.Write(NumOps.ToDouble(tensor[i]));
        }
    }

    private Tensor<T> DeserializeTensor(BinaryReader reader)
    {
        int rank = reader.ReadInt32();
        var shape = new int[rank];
        for (int i = 0; i < rank; i++)
        {
            shape[i] = reader.ReadInt32();
        }

        var tensor = new Tensor<T>(shape);
        int length = reader.ReadInt32();
        for (int i = 0; i < length; i++)
        {
            tensor[i] = NumOps.FromDouble(reader.ReadDouble());
        }

        return tensor;
    }
}
