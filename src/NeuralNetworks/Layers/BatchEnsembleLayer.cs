using AiDotNet.Autodiff;
using AiDotNet.Attributes;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// Implements a BatchEnsemble layer that provides parameter-efficient ensembling.
/// </summary>
/// <remarks>
/// <para>
/// BatchEnsemble creates multiple ensemble members that share base weights but have
/// their own small rank-1 matrices. For a weight matrix W, each member i computes:
/// W_i = W ⊙ (r_i ⊗ s_i)
/// where r_i and s_i are per-member rank vectors, ⊙ is element-wise multiplication,
/// and ⊗ is outer product.
/// </para>
/// <para>
/// <b>For Beginners:</b> BatchEnsemble is a clever way to create multiple models
/// (ensemble members) that share most of their weights.
///
/// Traditional ensemble: Train N separate models with N×parameters
/// BatchEnsemble: Train 1 base model + N small vectors = ~1×parameters + small overhead
///
/// How it works:
/// 1. A single shared weight matrix W captures the main learned patterns
/// 2. Each ensemble member has two small vectors (r and s)
/// 3. Member i's effective weights = W × (r_i outer-product s_i)
/// 4. This modulates the shared weights to create diversity
///
/// Benefits:
/// - Ensemble predictions (averaging multiple members) are usually more accurate
/// - Parameter cost is only slightly more than a single model
/// - Can process all members in parallel using batch operations
/// - Easy to implement and train
///
/// Example with 256-dim hidden layer and 4 members:
/// - Shared weights: 256 × 256 = 65,536 parameters
/// - Per-member vectors: 4 × (256 + 256) = 2,048 parameters
/// - Total overhead: ~3% more parameters for 4× ensemble benefit
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public partial class BatchEnsembleLayer<T> : LayerBase<T>
{
    private readonly int _inputDim;
    private readonly int _outputDim;
    private readonly int _numMembers;
    private readonly bool _useBias;

    // Shared base weights
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _weights;      // Shape: [inputDim, outputDim]
    [TrainableParameter(Role = PersistentTensorRole.Biases)]
    private Tensor<T>? _bias;        // Shape: [outputDim] (shared across members)

    // Per-member rank vectors
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _rVectors;     // Shape: [numMembers, inputDim]
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _sVectors;     // Shape: [numMembers, outputDim]

    // Gradients
    private Tensor<T>? _weightsGrad;
    private Tensor<T>? _biasGrad;
    private Tensor<T>? _rVectorsGrad;
    private Tensor<T>? _sVectorsGrad;

    // Cache for backward pass
    private Tensor<T>? _inputCache;     // Shape: [batchSize * numMembers, inputDim]
    private Tensor<T>? _scaledInputCache;  // Input after r-vector scaling

    /// <summary>
    /// Gets the input dimension.
    /// </summary>
    public int InputDim => _inputDim;

    /// <summary>
    /// Gets the output dimension.
    /// </summary>
    public int OutputDim => _outputDim;

    /// <summary>
    /// Gets the number of ensemble members.
    /// </summary>
    public int NumMembers => _numMembers;

    /// <inheritdoc/>
    public override bool SupportsTraining => true;

    /// <inheritdoc/>
    public override int ParameterCount
    {
        get
        {
            int count = _weights.Length + _rVectors.Length + _sVectors.Length;
            if (_bias != null) count += _bias.Length;
            return count;
        }
    }

    /// <summary>
    /// Initializes a new instance of the BatchEnsembleLayer class.
    /// </summary>
    /// <param name="inputDim">Input dimension.</param>
    /// <param name="outputDim">Output dimension.</param>
    /// <param name="numMembers">Number of ensemble members.</param>
    /// <param name="useBias">Whether to use bias.</param>
    /// <param name="rankInitScale">Initialization scale for rank vectors.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Creating a BatchEnsemble layer:
    /// - inputDim: Size of the input features
    /// - outputDim: Size of the output
    /// - numMembers: Number of ensemble members (typically 2-8)
    /// - useBias: Usually true, adds a learnable offset
    /// - rankInitScale: Controls initial diversity (0.3-0.7 typical)
    /// </para>
    /// </remarks>
    public BatchEnsembleLayer(
        int inputDim,
        int outputDim,
        int numMembers,
        bool useBias = true,
        double rankInitScale = 0.5)
        : base([inputDim], [outputDim])
    {
        _inputDim = inputDim;
        _outputDim = outputDim;
        _numMembers = numMembers;
        _useBias = useBias;

        // Initialize shared weights with Xavier/Glorot initialization
        _weights = new Tensor<T>([inputDim, outputDim]);
        InitializeXavier(_weights);

        // Initialize bias to zeros
        if (useBias)
        {
            _bias = new Tensor<T>([outputDim]);
            _bias.Fill(NumOps.Zero);
        }

        // Initialize r vectors (input modulation)
        _rVectors = new Tensor<T>([numMembers, inputDim]);
        InitializeRankVectors(_rVectors, rankInitScale);

        // Initialize s vectors (output modulation)
        _sVectors = new Tensor<T>([numMembers, outputDim]);
        InitializeRankVectors(_sVectors, rankInitScale);
    }

    /// <summary>
    /// Initializes a tensor with Xavier/Glorot initialization.
    /// </summary>
    private void InitializeXavier(Tensor<T> tensor)
    {
        var random = RandomHelper.CreateSecureRandom();
        int fanIn = tensor.Shape[0];
        int fanOut = tensor.Shape.Length > 1 ? tensor.Shape[1] : 1;
        double stdDev = Math.Sqrt(2.0 / (fanIn + fanOut));

        for (int i = 0; i < tensor.Length; i++)
        {
            double u1 = 1.0 - random.NextDouble();
            double u2 = 1.0 - random.NextDouble();
            double normal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
            tensor[i] = NumOps.FromDouble(normal * stdDev);
        }
    }

    /// <summary>
    /// Initializes rank vectors centered around 1 with some variation.
    /// </summary>
    private void InitializeRankVectors(Tensor<T> tensor, double scale)
    {
        var random = RandomHelper.CreateSecureRandom();

        for (int i = 0; i < tensor.Length; i++)
        {
            // Initialize around 1.0 with uniform noise in [-scale, +scale]
            double value = 1.0 + (random.NextDouble() * 2.0 - 1.0) * scale;
            tensor[i] = NumOps.FromDouble(value);
        }
    }

    /// <summary>
    /// Performs the forward pass through the BatchEnsemble layer.
    /// </summary>
    /// <param name="input">Input tensor [batchSize, inputDim].</param>
    /// <returns>Output tensor [batchSize * numMembers, outputDim].</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The forward pass processes input through all ensemble members:
    ///
    /// 1. Each input sample is replicated for each ensemble member
    /// 2. Input is scaled by each member's r vector (input modulation)
    /// 3. Scaled input is multiplied by shared weights
    /// 4. Output is scaled by each member's s vector (output modulation)
    /// 5. Bias is added (shared across members)
    ///
    /// The output has batchSize × numMembers rows, with consecutive numMembers
    /// rows belonging to the same input sample.
    /// </para>
    /// </remarks>
    public override Tensor<T> Forward(Tensor<T> input)
    {
        int batchSize = input.Shape[0];
        int expandedBatchSize = batchSize * _numMembers;

        // Build expandedInput [B*M, inputDim] = input tiled M times along a
        // synthetic member axis. Layout matches the original [b*M + m, i]
        // indexing because the reshape collapses the (b, m) dims in row-major
        // order. Replaces the per-(b, m, i) scalar copy loop.
        var input3D = Engine.Reshape(input, [batchSize, 1, _inputDim]);
        var inputTiled = Engine.TensorTile(input3D, [1, _numMembers, 1]);
        var expandedInput = Engine.Reshape(inputTiled, [expandedBatchSize, _inputDim]);

        // Build the per-row r-vector tensor by tiling _rVectors [M, inputDim]
        // to [B, M, inputDim] then collapsing to [B*M, inputDim]. Each row's
        // r-vector matches the row's member index.
        var rVecs3D = Engine.Reshape(_rVectors, [1, _numMembers, _inputDim]);
        var rVecsTiled = Engine.TensorTile(rVecs3D, [batchSize, 1, 1]);
        var rVecsFlat = Engine.Reshape(rVecsTiled, [expandedBatchSize, _inputDim]);

        // scaledInput = expandedInput * rVecsFlat (elementwise, vectorized)
        var scaledInput = Engine.TensorMultiply(expandedInput, rVecsFlat);

        _inputCache = expandedInput;
        _scaledInputCache = scaledInput;

        // Compute output via batched matmul: [B*M, inputDim] @ [inputDim, outputDim]
        // = [B*M, outputDim]. Replaces the O(B*M*outputDim*inputDim) scalar
        // accumulation chain with one Engine.TensorMatMul.
        var weights2D = Engine.Reshape(_weights, [_inputDim, _outputDim]);
        var matmul = Engine.TensorMatMul(scaledInput, weights2D);

        // Apply s-vector scaling per member by broadcasting [M, outputDim] →
        // [B, M, outputDim] → [B*M, outputDim]. Same tiling pattern used for
        // r-vectors above.
        var sVecs3D = Engine.Reshape(_sVectors, [1, _numMembers, _outputDim]);
        var sVecsTiled = Engine.TensorTile(sVecs3D, [batchSize, 1, 1]);
        var sVecsFlat = Engine.Reshape(sVecsTiled, [expandedBatchSize, _outputDim]);
        var output = Engine.TensorMultiply(matmul, sVecsFlat);

        // Add shared bias [outputDim] broadcast across all rows.
        if (_bias != null)
        {
            var biasBcast = Engine.Reshape(_bias, [1, _outputDim]);
            output = Engine.TensorBroadcastAdd(output, biasBcast);
        }

        return output;
    }

    /// <summary>
    /// Averages the outputs across ensemble members.
    /// </summary>
    /// <param name="output">Output tensor [batchSize * numMembers, outputDim].</param>
    /// <returns>Averaged output [batchSize, outputDim].</returns>
    public Tensor<T> AverageMembers(Tensor<T> output)
    {
        int expandedBatchSize = output.Shape[0];
        int batchSize = expandedBatchSize / _numMembers;

        // The output is laid out as [batchSize*numMembers, outputDim] with the
        // members for one batch item stored in consecutive rows — so reshape to
        // [batchSize, numMembers, outputDim] and reduce-mean over the member axis
        // to collapse the ensemble. One Engine call replaces
        // batchSize × outputDim × numMembers scalar NumOps.Add dispatches.
        var reshaped = Engine.Reshape(output, [batchSize, _numMembers, _outputDim]);
        return Engine.ReduceMean(reshaped, new[] { 1 }, keepDims: false);
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameters()
    {
        var paramsList = new List<T>();

        for (int i = 0; i < _weights.Length; i++)
            paramsList.Add(_weights[i]);

        if (_bias != null)
        {
            for (int i = 0; i < _bias.Length; i++)
                paramsList.Add(_bias[i]);
        }

        for (int i = 0; i < _rVectors.Length; i++)
            paramsList.Add(_rVectors[i]);

        for (int i = 0; i < _sVectors.Length; i++)
            paramsList.Add(_sVectors[i]);

        return new Vector<T>([.. paramsList]);
    }

    /// <summary>
    /// Sets the trainable parameters from a vector.
    /// </summary>
    public override void SetParameters(Vector<T> parameters)
    {
        int idx = 0;

        for (int i = 0; i < _weights.Length; i++)
            _weights[i] = parameters[idx++];

        if (_bias != null)
        {
            for (int i = 0; i < _bias.Length; i++)
                _bias[i] = parameters[idx++];
        }

        for (int i = 0; i < _rVectors.Length; i++)
            _rVectors[i] = parameters[idx++];

        for (int i = 0; i < _sVectors.Length; i++)
            _sVectors[i] = parameters[idx++];
    }

    /// <summary>
    /// Gets the parameter gradients as a single vector.
    /// </summary>
    public override Vector<T> GetParameterGradients()
    {
        var gradsList = new List<T>();

        if (_weightsGrad != null)
        {
            for (int i = 0; i < _weightsGrad.Length; i++)
                gradsList.Add(_weightsGrad[i]);
        }
        else
        {
            for (int i = 0; i < _weights.Length; i++)
                gradsList.Add(NumOps.Zero);
        }

        if (_bias != null)
        {
            if (_biasGrad != null)
            {
                for (int i = 0; i < _biasGrad.Length; i++)
                    gradsList.Add(_biasGrad[i]);
            }
            else
            {
                for (int i = 0; i < _bias.Length; i++)
                    gradsList.Add(NumOps.Zero);
            }
        }

        if (_rVectorsGrad != null)
        {
            for (int i = 0; i < _rVectorsGrad.Length; i++)
                gradsList.Add(_rVectorsGrad[i]);
        }
        else
        {
            for (int i = 0; i < _rVectors.Length; i++)
                gradsList.Add(NumOps.Zero);
        }

        if (_sVectorsGrad != null)
        {
            for (int i = 0; i < _sVectorsGrad.Length; i++)
                gradsList.Add(_sVectorsGrad[i]);
        }
        else
        {
            for (int i = 0; i < _sVectors.Length; i++)
                gradsList.Add(NumOps.Zero);
        }

        return new Vector<T>([.. gradsList]);
    }

    /// <inheritdoc/>
    public override void UpdateParameters(T learningRate)
    {
        if (_weightsGrad != null)
        {
            _weights = Engine.TensorSubtract(_weights, Engine.TensorMultiplyScalar(_weightsGrad, learningRate));
        }

        if (_bias != null && _biasGrad != null)
        {
            _bias = Engine.TensorSubtract(_bias, Engine.TensorMultiplyScalar(_biasGrad, learningRate));
        }

        if (_rVectorsGrad != null)
        {
            _rVectors = Engine.TensorSubtract(_rVectors, Engine.TensorMultiplyScalar(_rVectorsGrad, learningRate));
        }

        if (_sVectorsGrad != null)
        {
            _sVectors = Engine.TensorSubtract(_sVectors, Engine.TensorMultiplyScalar(_sVectorsGrad, learningRate));
        }

        // Register trainable parameters for tape-based autodiff
        RegisterTrainableParameter(_weights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_rVectors, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_sVectors, PersistentTensorRole.Weights);

    }

    /// <summary>
    /// Resets all gradients.
    /// </summary>
    public void ResetGradients()
    {
        _weightsGrad = null;
        _biasGrad = null;
        _rVectorsGrad = null;
        _sVectorsGrad = null;
    }

    /// <summary>
    /// Gets the shared weights tensor.
    /// </summary>
    public override Tensor<T>? GetWeights() => _weights;

    /// <summary>
    /// Gets the r vectors tensor.
    /// </summary>
    public Tensor<T> GetRVectors() => _rVectors;

    /// <summary>
    /// Gets the s vectors tensor.
    /// </summary>
    public Tensor<T> GetSVectors() => _sVectors;

    /// <inheritdoc/>
    public override void ResetState()
    {
        _inputCache = null;
        _scaledInputCache = null;
        _weightsGrad = null;
        _biasGrad = null;
        _rVectorsGrad = null;
        _sVectorsGrad = null;
    }
}
