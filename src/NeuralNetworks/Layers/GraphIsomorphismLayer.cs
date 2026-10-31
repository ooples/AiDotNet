using AiDotNet.Attributes;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Implements Graph Isomorphism Network (GIN) layer for powerful graph representation learning.
/// </summary>
/// <remarks>
/// <para>
/// Graph Isomorphism Networks (GIN), introduced by Xu et al., are provably as powerful as the
/// Weisfeiler-Lehman (WL) graph isomorphism test for distinguishing graph structures. GIN uses
/// a sum aggregation with a learnable epsilon parameter and applies a multi-layer perceptron (MLP)
/// to the aggregated features.
/// </para>
/// <para>
/// The layer computes: h_v^(k) = MLP^(k)((1 + epsilon^(k)) * h_v^(k-1) + sum_{u in N(v)} h_u^(k-1))
/// where h_v is the representation of node v, N(v) is the neighborhood of v,
/// epsilon is a learnable parameter (or fixed), and MLP is a multi-layer perceptron.
/// </para>
/// <para>
/// <b>Production-Ready Features:</b>
/// <list type="bullet">
/// <item>Fully vectorized operations using IEngine for GPU acceleration</item>
/// <item>Tensor-based weights for all parameters</item>
/// <item>Dual backward pass: BackwardManual() for efficiency, BackwardViaAutodiff() for accuracy</item>
/// <item>Full gradient computation through MLP and aggregation paths</item>
/// <item>Complete GetParameters()/SetParameters() for model persistence</item>
/// </list>
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
[LayerCategory(LayerCategory.Graph)]
[LayerTask(LayerTask.GraphProcessing)]
[LayerTask(LayerTask.FeatureExtraction)]
[LayerProperty(ApiShape = LayerApiShape.GraphWithSetup, IsTrainable = true, ChangesShape = true, TestInputShape = "4, 8", TestConstructorArgs = "8, 4, initializationStrategy: new global::AiDotNet.Initialization.EagerInitializationStrategy<double>(new System.Random(42))", TestSetupCode = "var adj = new AiDotNet.Tensors.LinearAlgebra.Tensor<double>(new[] { 4, 4 }); for (int i = 0; i < 4; i++) { adj[i, i] = 1.0; if (i > 0) adj[i, i-1] = 1.0; if (i < 3) adj[i, i+1] = 1.0; } var m = layer.GetType().GetMethod(\"SetAdjacencyMatrix\"); if (m != null) m.Invoke(layer, new object[] { adj });")]
// Rank 2 - [numNodes, inputFeatures] - is what the forward's own restore branch names ("Original was 2D
// [N, F] -> return [N, outputFeatures]") and the only form [LayerProperty(TestInputShape = "4, 8")]
// exercises. Other ranks ARE handled, but their node count must agree with the separately-installed
// adjacency matrix, so a declaration there would be a claim about a tensor this layer never sees alone.
//
// The node axis is TensorAxis.Other: graph nodes are neither a sequence nor a spatial extent, and naming
// them Time or Length would licence a downstream causal or convolutional layer to read an ordering into
// them that does not exist.
[TensorLayout(TensorAxis.Other, TensorAxis.Features,
    Direction = TensorLayoutDirection.Input,
    Note = "Node features: the leading axis is the graph's nodes, sized by the installed adjacency matrix.")]
[TensorLayout(TensorAxis.Other, TensorAxis.Features, Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class GraphIsomorphismLayer<T> : LayerBase<T>, IGraphConvolutionLayer<T>, IShapeContract
{
    /// <inheritdoc />
    /// <remarks>
    /// <para>
    /// HAND-WRITTEN because the emitted width is set by the MLP's SECOND layer, which is configuration:
    /// the aggregation runs at <c>_mlpHiddenDim</c> and only the final projection
    /// (<c>_mlpWeights2 = new Tensor&lt;T&gt;([_mlpHiddenDim, _outputFeatures])</c>) fixes the width the
    /// layer actually emits. A relation derived from the hidden dimension would be right only where the
    /// two happen to coincide.
    /// </para>
    /// <para>
    /// The node axis is Same: GIN sums each node's neighbourhood into that node, so nodes are neither
    /// created nor removed.
    /// </para>
    /// </remarks>
    public IReadOnlyList<OutputAxisContract>? OutputAxesFor(int inputRank)
    {
        if (inputRank != 2 || _outputFeatures <= 0) return null;

        return new[]
        {
            new OutputAxisContract(TensorAxis.Other, AxisRelation.Same(TensorAxis.Other)),
            new OutputAxisContract(TensorAxis.Features, AxisRelation.Fixed(_outputFeatures)),
        };
    }

    private readonly int _inputFeatures;
    private readonly int _outputFeatures;
    private readonly int _mlpHiddenDim;
    private readonly bool _learnEpsilon;

    /// <summary>
    /// Epsilon parameter for weighting self vs neighbor features.
    /// </summary>
    private T _epsilon;

    /// <summary>
    /// First layer of the MLP: [inputFeatures, mlpHiddenDim].
    /// </summary>
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _mlpWeights1;

    /// <summary>
    /// Second layer of the MLP: [mlpHiddenDim, outputFeatures].
    /// </summary>
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _mlpWeights2;

    /// <summary>
    /// Bias for first MLP layer: [mlpHiddenDim].
    /// </summary>
    [TrainableParameter(Role = PersistentTensorRole.Biases)]

    private Tensor<T> _mlpBias1;

    /// <summary>
    /// Bias for second MLP layer: [outputFeatures].
    /// </summary>
    [TrainableParameter(Role = PersistentTensorRole.Biases)]

    private Tensor<T> _mlpBias2;

    /// <summary>
    /// The adjacency matrix defining graph structure.
    /// </summary>
    [AiDotNet.Attributes.FittedParameter(InputSized = true)]
    private Tensor<T>? _adjacencyMatrix;

    /// <summary>
    /// Stores the original input shape for any-rank tensor support.
    /// </summary>
    private int[]? _originalInputShape;

    /// <summary>
    /// Cached aggregated features (before MLP).
    /// </summary>
    [Scratch]
    private Tensor<T>? _lastAggregated;

    /// <summary>
    /// Cached pre-ReLU hidden layer output from MLP.
    /// </summary>
    [Scratch]
    private Tensor<T>? _lastMlpHiddenPreRelu;

    /// <summary>
    /// Cached hidden layer output from MLP (after ReLU).
    /// </summary>
    [Scratch]
    private Tensor<T>? _lastMlpHidden;

    /// <summary>
    /// Gradients for epsilon.
    /// </summary>
    private T _epsilonGradient;

    /// <summary>
    /// Gradients for MLP weights.
    /// </summary>
    [Scratch]
    private Tensor<T>? _mlpWeights1Gradient;
    [Scratch]
    private Tensor<T>? _mlpWeights2Gradient;
    [Scratch]
    private Tensor<T>? _mlpBias1Gradient;
    [Scratch]
    private Tensor<T>? _mlpBias2Gradient;

    /// <inheritdoc/>
    public override bool SupportsTraining => true;

    /// <inheritdoc/>
    public int InputFeatures => _inputFeatures;

    /// <inheritdoc/>
    public int OutputFeatures => _outputFeatures;

    /// <summary>
    /// Initializes a new instance of the <see cref="GraphIsomorphismLayer{T}"/> class.
    /// </summary>
    public GraphIsomorphismLayer(
        [LayerState] int inputFeatures,
        [LayerState] int outputFeatures,
        [LayerState] int mlpHiddenDim = -1,
        [LayerState] bool learnEpsilon = true,
        [LayerState] double epsilon = 0.0,
        IActivationFunction<T>? activationFunction = null,
        IInitializationStrategy<T>? initializationStrategy = null)
        : base([inputFeatures], [outputFeatures], activationFunction ?? new IdentityActivation<T>())
    {
        InitializationStrategy = initializationStrategy ?? InitializationStrategies<T>.Eager;

        _inputFeatures = inputFeatures;
        _outputFeatures = outputFeatures;
        _mlpHiddenDim = mlpHiddenDim > 0 ? mlpHiddenDim : outputFeatures;
        _learnEpsilon = learnEpsilon;
        _epsilon = NumOps.FromDouble(epsilon);
        _epsilonGradient = NumOps.Zero;

        // Initialize weights as Tensors for GPU acceleration
        _mlpWeights1 = new Tensor<T>([_inputFeatures, _mlpHiddenDim]);
        _mlpWeights2 = new Tensor<T>([_mlpHiddenDim, _outputFeatures]);
        _mlpBias1 = new Tensor<T>([_mlpHiddenDim]);
        _mlpBias2 = new Tensor<T>([_outputFeatures]);

        InitializeParameters();

        RegisterTrainableParameter(_mlpWeights1, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_mlpWeights2, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_mlpBias1, PersistentTensorRole.Biases);
        RegisterTrainableParameter(_mlpBias2, PersistentTensorRole.Biases);
    }

    /// <summary>
    /// Initializes layer parameters using Xavier initialization with Engine operations.
    /// </summary>
    private void InitializeParameters()
    {
        // Xavier/Glorot initialization using Engine operations
        InitializeTensor(_mlpWeights1, _inputFeatures, _mlpHiddenDim);
        InitializeTensor(_mlpWeights2, _mlpHiddenDim, _outputFeatures);

        // Initialize biases to zero
        _mlpBias1.Fill(NumOps.Zero);
        _mlpBias2.Fill(NumOps.Zero);
    }

    private void InitializeTensor(Tensor<T> tensor, int fanIn, int fanOut)
    {
        InitializeLayerWeights(tensor, fanIn, fanOut);
    }

    /// <inheritdoc/>
    public void SetAdjacencyMatrix(Tensor<T> adjacencyMatrix)
    {
        _adjacencyMatrix = adjacencyMatrix;
    }

    /// <inheritdoc/>
    public Tensor<T>? GetAdjacencyMatrix()
    {
        return _adjacencyMatrix;
    }

    /// <inheritdoc/>
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        if (_adjacencyMatrix == null)
        {
            throw new InvalidOperationException(
                "Adjacency matrix must be set using SetAdjacencyMatrix before calling Forward.");
        }

        // Store original shape for any-rank tensor support
        _originalInputShape = input._shape;
        int rank = input.Shape.Length;

        // Handle tensor shapes for graph processing:
        // - 2D input [N, F] = single graph with N nodes and F features -> reshape to [1, N, F]
        // - 3D input [B, N, F] = batch of B graphs with N nodes and F features
        Tensor<T> processInput;
        int batchSize;
        int numNodes;

        if (rank == 1)
        {
            // 1D: treat as single node with F features -> [1, 1, F]
            batchSize = 1;
            numNodes = 1;
            processInput = Engine.Reshape(input, [1, 1, input.Shape[0]]);
        }
        else if (rank == 2)
        {
            // 2D: [numNodes, features] -> reshape to [1, numNodes, features]
            batchSize = 1;
            numNodes = input.Shape[0];
            processInput = Engine.Reshape(input, [1, input.Shape[0], input.Shape[1]]);
        }
        else
        {
            // 3D: [batch, numNodes, features]
            batchSize = input.Shape[0];
            numNodes = input.Shape[1];
            processInput = input;
        }

        // Reshape adjacency matrix to match batch dimension if needed
        Tensor<T> adjForBatch;
        if (_adjacencyMatrix.Shape.Length == 2 && batchSize == 1)
        {
            adjForBatch = Engine.Reshape(_adjacencyMatrix, [1, _adjacencyMatrix.Shape[0], _adjacencyMatrix.Shape[1]]);
        }
        else if (_adjacencyMatrix.Shape.Length == 2 && batchSize > 1)
        {
            // Tile adjacency matrix for batch
            int adjN = _adjacencyMatrix.Shape[0];
            adjForBatch = TensorAllocator.Rent<T>([batchSize, adjN, adjN]);
            for (int b = 0; b < batchSize; b++)
            {
                for (int i = 0; i < adjN; i++)
                {
                    for (int j = 0; j < adjN; j++)
                    {
                        adjForBatch[new int[] { b, i, j }] = _adjacencyMatrix[new int[] { i, j }];
                    }
                }
            }
        }
        else
        {
            adjForBatch = _adjacencyMatrix;
        }

        // Step 1: Aggregate neighbor features using sum (batched matmul: [batch, nodes, nodes] @ [batch, nodes, features])
        var neighborSum = Engine.BatchMatMul(adjForBatch, processInput);

        // Step 2: Combine with self features: (1 + epsilon) * h_v + sum(h_u)
        T onePlusEpsilon = NumOps.Add(NumOps.One, _epsilon);
        var scaledSelf = Engine.TensorMultiplyScalar(processInput, onePlusEpsilon);
        var aggregated = Engine.TensorAdd(scaledSelf, neighborSum);
        SaveForBackward(ref _lastAggregated, aggregated);

        // Step 3: Apply MLP - First layer (3D @ 2D requires reshape pattern)
        var hidden1 = BatchedMatMul3Dx2D(aggregated, _mlpWeights1, batchSize, numNodes, _inputFeatures, _mlpHiddenDim);
        var bias1Broadcast = BroadcastBias(_mlpBias1, batchSize, numNodes);
        var lastMlpHiddenPreRelu = Engine.TensorAdd(hidden1, bias1Broadcast);
        SaveForBackward(ref _lastMlpHiddenPreRelu, lastMlpHiddenPreRelu);

        // Apply ReLU activation using Engine operations
        var zeroTensor = new Tensor<T>(lastMlpHiddenPreRelu._shape);
        zeroTensor.Fill(NumOps.Zero);
        var mlpHidden = Engine.TensorMax(lastMlpHiddenPreRelu, zeroTensor);
        SaveForBackward(ref _lastMlpHidden, mlpHidden);

        // Step 4: Apply MLP - Second layer (3D @ 2D requires reshape pattern)
        var hidden2 = BatchedMatMul3Dx2D(mlpHidden, _mlpWeights2, batchSize, numNodes, _mlpHiddenDim, _outputFeatures);
        var bias2Broadcast = BroadcastBias(_mlpBias2, batchSize, numNodes);
        var mlpOutput = Engine.TensorAdd(hidden2, bias2Broadcast);

        var result = ApplyActivation(mlpOutput);

        // Reshape output to match original input shape (except for feature dimension)
        if (_originalInputShape != null && _originalInputShape.Length == 2)
        {
            // Original was 2D [N, F] -> return [N, outputFeatures]
            return Engine.Reshape(result, [numNodes, _outputFeatures]);
        }
        else if (_originalInputShape != null && _originalInputShape.Length == 1)
        {
            // Original was 1D [F] -> return [outputFeatures]
            return Engine.Reshape(result, [_outputFeatures]);
        }

        return result;
    }

    /// <summary>
    /// Performs batched matrix multiplication between a 3D tensor and a 2D weight matrix.
    /// Input: [batch, rows, cols] @ weights: [cols, output_cols] -> [batch, rows, output_cols]
    /// </summary>
    private Tensor<T> BatchedMatMul3Dx2D(Tensor<T> input3D, Tensor<T> weights2D, int batch, int rows, int cols, int outputCols)
    {
        // Flatten batch dimension: [batch, rows, cols] -> [batch * rows, cols]
        // Use Engine.Reshape (NOT Tensor.Reshape): the gradient tape only records Engine ops, so a
        // plain Tensor.Reshape here severs the tape edge around the MLP matmul. The downstream
        // gradient then never reaches Engine.TensorMatMul, so _mlpWeights1/_mlpWeights2 received
        // ZERO gradient and never updated — the GIN Training_ShouldChangeParameters / GradientFlow /
        // LossStrictlyDecreases failures. The rest of Forward already uses Engine.Reshape; this
        // helper was the lone non-recording reshape on the gradient path.
        var flattened = Engine.Reshape(input3D, [batch * rows, cols]);
        // Standard 2D matmul: [batch * rows, cols] @ [cols, output_cols] -> [batch * rows, output_cols]
        var result = Engine.TensorMatMul(flattened, weights2D);
        // Unflatten: [batch * rows, output_cols] -> [batch, rows, output_cols]
        return Engine.Reshape(result, [batch, rows, outputCols]);
    }

    /// <summary>
    /// Broadcasts a bias tensor across batch and node dimensions using Engine operations.
    /// </summary>
    private Tensor<T> BroadcastBias(Tensor<T> bias, int batchSize, int numNodes)
    {
        int features = bias.Length;
        var biasReshaped = bias.Reshape([1, 1, features]);
        return Engine.TensorTile(biasReshaped, [batchSize, numNodes, 1]);
    }

    /// <summary>
    /// Computes gradients using fully vectorized Engine operations.
    /// </summary>
    private void ComputeGradientsViaEngine(Tensor<T> activationGradient, int batchSize, int numNodes)
    {
        // Gradient through MLP Layer 2 bias: sum over batch and nodes
        _mlpBias2Gradient = Engine.ReduceSum(activationGradient, [0, 1], keepDims: false);

        // Gradient through MLP Layer 2 weights: hidden^T @ grad (batched then summed)
        var hiddenBatchedT = Engine.TensorPermute((_lastMlpHidden ?? throw new InvalidOperationException("_lastMlpHidden is null where it is required.")), [0, 2, 1]);
        var weights2GradBatched = Engine.BatchMatMul(hiddenBatchedT, activationGradient);
        _mlpWeights2Gradient = Engine.ReduceSum(weights2GradBatched, [0], keepDims: false);

        // Gradient to hidden layer: grad @ weights2^T
        var weights2T = Engine.TensorTranspose(_mlpWeights2);
        var hiddenGradPre = Engine.TensorMatMul(activationGradient, weights2T);

        // Gradient through ReLU: fully vectorized element-wise operations
        var lastMlpHiddenPreRelu = _lastMlpHiddenPreRelu ?? throw new InvalidOperationException("_lastMlpHiddenPreRelu has not been initialized.");
        var zeroTensor = new Tensor<T>(lastMlpHiddenPreRelu._shape);
        Engine.TensorFill(zeroTensor, NumOps.Zero);
        var reluMask = Engine.TensorGreaterThan(lastMlpHiddenPreRelu, zeroTensor);
        var oneTensor = new Tensor<T>(lastMlpHiddenPreRelu._shape);
        Engine.TensorFill(oneTensor, NumOps.One);
        var reluDeriv = Engine.TensorWhere(reluMask, oneTensor, zeroTensor);
        var hiddenGrad = Engine.TensorMultiply(hiddenGradPre, reluDeriv);

        // Gradient through MLP Layer 1 bias: sum over batch and nodes
        _mlpBias1Gradient = Engine.ReduceSum(hiddenGrad, [0, 1], keepDims: false);

        // Gradient through MLP Layer 1 weights: aggregated^T @ hiddenGrad (batched then summed)
        var aggBatchedT = Engine.TensorPermute((_lastAggregated ?? throw new InvalidOperationException("_lastAggregated is null where it is required.")), [0, 2, 1]);
        var weights1GradBatched = Engine.BatchMatMul(aggBatchedT, hiddenGrad);
        _mlpWeights1Gradient = Engine.ReduceSum(weights1GradBatched, [0], keepDims: false);
    }

    /// <inheritdoc/>
    public override void UpdateParameters(T learningRate)
    {
        if (_mlpWeights1Gradient == null || _mlpWeights2Gradient == null ||
            _mlpBias1Gradient == null || _mlpBias2Gradient == null)
        {
            throw new InvalidOperationException("Backward must be called before UpdateParameters.");
        }

        // Update using vectorized Engine operations
        Engine.TensorSubtractInPlace(_mlpWeights1,
            Engine.TensorMultiplyScalar(_mlpWeights1Gradient, learningRate));
        Engine.TensorSubtractInPlace(_mlpWeights2,
            Engine.TensorMultiplyScalar(_mlpWeights2Gradient, learningRate));
        Engine.TensorSubtractInPlace(_mlpBias1,
            Engine.TensorMultiplyScalar(_mlpBias1Gradient, learningRate));
        Engine.TensorSubtractInPlace(_mlpBias2,
            Engine.TensorMultiplyScalar(_mlpBias2Gradient, learningRate));

        // Update epsilon if learnable
        if (_learnEpsilon)
        {
            _epsilon = NumOps.Subtract(_epsilon, NumOps.Multiply(learningRate, _epsilonGradient));
        }
    }

    /// <summary>
    /// Returns layer-specific metadata for serialization.
    /// </summary>
    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["MlpHiddenDim"] = _mlpHiddenDim.ToString();
        metadata["LearnEpsilon"] = _learnEpsilon.ToString();
        metadata["InitialEpsilon"] = NumOps.ToDouble(_epsilon).ToString(System.Globalization.CultureInfo.InvariantCulture);
        return metadata;
    }

    public override void ClearGradients()
    {
        base.ClearGradients();
        _epsilonGradient = NumOps.Zero;
        _mlpWeights1Gradient = null; _mlpWeights2Gradient = null;
        _mlpBias1Gradient = null; _mlpBias2Gradient = null;
    }

    /// <inheritdoc/>
    public override void ResetState()
    {
        _lastAggregated = null;
        _lastMlpHiddenPreRelu = null;
        _lastMlpHidden = null;
        _mlpWeights1Gradient = null;
        _mlpWeights2Gradient = null;
        _mlpBias1Gradient = null;
        _mlpBias2Gradient = null;
        _epsilonGradient = NumOps.Zero;
    }

    #region GPU Helper Methods

    /// <summary>
    /// CPU fallback for copying data to an offset in a destination buffer.
    /// </summary>
    private static void CopyToOffsetCpu(IDirectGpuBackend backend, IGpuBuffer sourceBuffer, IGpuBuffer destBuffer, int destOffset, int size)
    {
        float[] source = backend.DownloadBuffer(sourceBuffer);
        float[] dest = backend.DownloadBuffer(destBuffer);
        Array.Copy(source, 0, dest, destOffset, size);
        using var tempBuffer = backend.AllocateBuffer(dest);
        backend.Copy(tempBuffer, destBuffer, dest.Length);
    }

    /// <summary>
    /// CPU fallback for applying activation function.
    /// </summary>
    private static void ApplyActivationCpu(IDirectGpuBackend backend, IGpuBuffer buffer, int size, FusedActivationType activationType)
    {
        switch (activationType)
        {
            case FusedActivationType.ReLU:
                backend.Relu(buffer, buffer, size);
                break;
            case FusedActivationType.Sigmoid:
                backend.Sigmoid(buffer, buffer, size);
                break;
            case FusedActivationType.Tanh:
                backend.Tanh(buffer, buffer, size);
                break;
            case FusedActivationType.GELU:
                backend.Gelu(buffer, buffer, size);
                break;
                // None/Identity does nothing
        }
    }

    #endregion
}
