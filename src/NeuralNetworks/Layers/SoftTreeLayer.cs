using AiDotNet.Helpers;
using AiDotNet.Attributes;
using AiDotNet.Autodiff;
using AiDotNet.Interfaces;
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
[LayerCategory(LayerCategory.Other)]
[LayerTask(LayerTask.FeatureExtraction)]
[LayerProperty(IsTrainable = true, ChangesShape = true, TestInputShape = "1, 4", TestConstructorArgs = "4, 8, 2")]
// FEATURE-LAST, like a dense projection: ForwardTraced flattens everything ahead of the trailing axis
// into a batch ("input.Length / features, features"), runs the tree, then restores the caller's leading
// dimensions with only the last axis replaced by outputDim. Its own <returns> says so: "[outputDim] for
// rank-1 input, [batchSize, outputDim] for rank-2, and [d0, ..., outputDim] for higher rank". The tree
// structure - depth, internal nodes, leaves - is entirely interior; it never reaches the output shape.
[TensorLayout(TensorAxis.Features, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Features, Direction = TensorLayoutDirection.Output)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Features, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Features, Direction = TensorLayoutDirection.Output)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features,
    Direction = TensorLayoutDirection.Input,
    Note = "Per-position tree evaluation: leading axes are flattened into the batch and restored.")]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features,
    Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class SoftTreeLayer<T> : LayerBase<T>, IShapeContract
{
    /// <inheritdoc />
    /// <remarks>
    /// <para>
    /// Rank-polymorphic for the same reason <c>DenseLayer</c> is: the layer fixes the TRAILING axis and
    /// carries every leading axis through untouched. <c>ForwardTraced</c> reshapes to
    /// <c>[input.Length / features, features]</c>, matmuls, and then rebuilds <c>outShape</c> by copying
    /// <c>input.Shape[i]</c> for every leading axis and setting only the last to <c>outputDim</c>.
    /// </para>
    /// <para>
    /// <c>Fixed(_outputDim)</c> is the constructor argument, not an observed number - it is the width of
    /// <c>_leafValues</c> (<c>[numLeaves, outputDim]</c>), the right operand of the final matmul, so it
    /// is a size the layer's parameters genuinely impose rather than one inherited from the input.
    /// </para>
    /// <para>
    /// <c>_numLeaves</c> and <c>_numInternalNodes</c> are deliberately absent. They size the path
    /// probabilities, an intermediate that the leaf matmul contracts away; naming a tree of depth 4 as a
    /// 16-wide axis would describe a tensor the caller never receives.
    /// </para>
    /// </remarks>
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
    {
        if (_outputDim <= 0 || inputRank < 1) return null;

        var features = new OutputAxisContract(TensorAxis.Features, AxisRelation.Fixed(_outputDim));
        OutputAxisContract Pass(TensorAxis a) => new(a, AxisRelation.Same(a));

        // Enumerated rather than looped: each leading axis needs a DISTINCT role, since a relation
        // refers to its input by role and two anonymous placeholders could not be told apart.
        return inputRank switch
        {
            1 => new[] { features },
            2 => new[] { Pass(TensorAxis.Batch), features },
            3 => new[] { Pass(TensorAxis.Batch), Pass(TensorAxis.Time), features },
            _ => null,
        };
    }

    private readonly int _inputDim;
    private readonly int _depth;
    private readonly int _outputDim;
    private readonly double _temperature;
    private readonly int _numInternalNodes;
    private readonly int _numLeaves;

    // Parameters: split weights and biases for internal nodes, leaf values
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _splitWeights;   // [numInternalNodes, inputDim]
    [TrainableParameter(Role = PersistentTensorRole.Biases)]

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
    private Tensor<T>? _cachedRightProbs;
    private Tensor<T>? _cachedNodeProbs;
    private Tensor<T>? _cachedSplitLogits;

    /// <summary>Construction state: the 'initScale' the layer was built with.</summary>
    private readonly double _initScale;

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
        _initScale = initScale;
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

        // Register after initialization so tensor references are final
        RegisterTrainableParameter(_splitWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_splitBiases, PersistentTensorRole.Biases);
        RegisterTrainableParameter(_leafValues, PersistentTensorRole.Weights);
    }

    /// <summary>
    /// Forward pass through the soft tree.
    /// </summary>
    /// <param name="input">
    /// Input tensor whose last dimension is the feature dimension. A rank-2
    /// <c>[batchSize, inputDim]</c> tensor is the canonical shape; a rank-1 <c>[inputDim]</c>
    /// sample and higher-rank <c>[d0, ..., inputDim]</c> tensors are also accepted â€” the leading
    /// dimensions are flattened into the batch for the internal matmuls and restored on the output.
    /// </param>
    /// <returns>
    /// Tree output with the input's leading dimensions preserved and the last dimension replaced by
    /// <c>outputDim</c>: <c>[outputDim]</c> for rank-1 input, <c>[batchSize, outputDim]</c> for
    /// rank-2, and <c>[d0, ..., outputDim]</c> for higher rank.
    /// </returns>
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        // The split-logit and leaf-value steps are matmuls and require a rank-2 [batch, features]
        // input. Flatten a rank-1 ([features]) or higher-rank input to 2D so an unbatched single
        // sample doesn't fault the matmul (TensorMatMul requires rank >= 2); restore the caller's
        // original leading dimensions on the way out. The reshape is tape-recorded, so gradients
        // still flow back to the original input.
        int features = input.Shape[input.Rank - 1];
        bool wasRank1 = input.Rank == 1;
        bool wasHigherRank = input.Rank > 2;
        var x = input.Rank == 2 ? input : Engine.Reshape(input, new[] { input.Length / features, features });
        _lastInput = x;
        int batchSize = x.Shape[0];

        // Compute split logits: input @ splitWeights^T + splitBiases
        // Shape: [batchSize, numInternalNodes]
        var splitWeightsT = Engine.TensorTranspose(_splitWeights);
        var splitLogits = Engine.TensorMatMul(x, splitWeightsT);

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

        // Cache split logits for backward
        _cachedSplitLogits = splitLogits;

        // Apply sigmoid to get right-branch probabilities
        var rightProbs = Engine.Sigmoid(splitLogits);
        _cachedRightProbs = rightProbs;

        // Compute path probabilities to each leaf (also caches nodeProbs)
        _pathProbabilities = ComputePathProbabilities(rightProbs, batchSize);

        // Weighted sum of leaf values: pathProbs @ leafValues
        // pathProbs: [batchSize, numLeaves], leafValues: [numLeaves, outputDim]
        var output = Engine.TensorMatMul(_pathProbabilities, _leafValues);
        int outputDim = _leafValues.Shape[1];
        if (wasRank1)
        {
            // Restore the caller's unbatched rank-1 shape: [outputDim].
            output = Engine.Reshape(output, new[] { outputDim });
        }
        else if (wasHigherRank)
        {
            // Restore the caller's leading dimensions: [d0, ..., d_{k-2}, outputDim].
            var outShape = new int[input.Rank];
            for (int i = 0; i < input.Rank - 1; i++) outShape[i] = input.Shape[i];
            outShape[input.Rank - 1] = outputDim;
            output = Engine.Reshape(output, outShape);
        }

        return output;
    }

    /// <summary>
    /// Computes the probability of reaching each leaf node.
    /// </summary>
    private Tensor<T> ComputePathProbabilities(Tensor<T> rightProbs, int batchSize)
    {
        var pathProbs = TensorAllocator.Rent<T>([batchSize, _numLeaves]);

        // Initialize all paths with probability 1 at root
        var nodeProbs = TensorAllocator.Rent<T>([batchSize, _numInternalNodes + _numLeaves]);
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

        _cachedNodeProbs = nodeProbs;
        return pathProbs;
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
        int totalParams = ParameterCountHelper.ToFlatVectorSize(ParameterCount);
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

    public override void ClearGradients()
    {
        base.ClearGradients();
        _splitWeightsGrad = null; _splitBiasesGrad = null; _leafValuesGrad = null;
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
        foreach (var dim in tensor._shape)
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
