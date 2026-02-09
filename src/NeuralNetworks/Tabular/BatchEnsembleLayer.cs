namespace AiDotNet.NeuralNetworks.Tabular;

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
public class BatchEnsembleLayer<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly int _inputDim;
    private readonly int _outputDim;
    private readonly int _numMembers;
    private readonly bool _useBias;

    // Shared base weights
    private Tensor<T> _weights;      // Shape: [inputDim, outputDim]
    private Tensor<T>? _bias;        // Shape: [outputDim] (shared across members)

    // Per-member rank vectors
    private Tensor<T> _rVectors;     // Shape: [numMembers, inputDim]
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

    /// <summary>
    /// Gets the total number of trainable parameters.
    /// </summary>
    public int ParameterCount
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
    {
        _numOps = MathHelper.GetNumericOperations<T>();
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
            _bias.Fill(_numOps.Zero);
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
            tensor[i] = _numOps.FromDouble(normal * stdDev);
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
            tensor[i] = _numOps.FromDouble(value);
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
    public Tensor<T> Forward(Tensor<T> input)
    {
        int batchSize = input.Shape[0];
        int expandedBatchSize = batchSize * _numMembers;

        // Expand input for each ensemble member
        // Input: [batch, input_dim]
        // Expanded: [batch * members, input_dim]
        var expandedInput = new Tensor<T>([expandedBatchSize, _inputDim]);

        // Apply r-vector scaling and expand
        var scaledInput = new Tensor<T>([expandedBatchSize, _inputDim]);

        for (int b = 0; b < batchSize; b++)
        {
            for (int m = 0; m < _numMembers; m++)
            {
                int rowIdx = b * _numMembers + m;
                for (int i = 0; i < _inputDim; i++)
                {
                    var inputVal = input[b * _inputDim + i];
                    var rVal = _rVectors[m * _inputDim + i];
                    expandedInput[rowIdx * _inputDim + i] = inputVal;
                    scaledInput[rowIdx * _inputDim + i] = _numOps.Multiply(inputVal, rVal);
                }
            }
        }

        _inputCache = expandedInput;
        _scaledInputCache = scaledInput;

        // Compute output: scaledInput @ weights
        var output = new Tensor<T>([expandedBatchSize, _outputDim]);

        for (int row = 0; row < expandedBatchSize; row++)
        {
            int memberIdx = row % _numMembers;

            for (int j = 0; j < _outputDim; j++)
            {
                var sum = _numOps.Zero;
                for (int i = 0; i < _inputDim; i++)
                {
                    sum = _numOps.Add(sum,
                        _numOps.Multiply(scaledInput[row * _inputDim + i], _weights[i * _outputDim + j]));
                }

                // Apply s-vector scaling
                var sVal = _sVectors[memberIdx * _outputDim + j];
                sum = _numOps.Multiply(sum, sVal);

                // Add bias
                if (_bias != null)
                {
                    sum = _numOps.Add(sum, _bias[j]);
                }

                output[row * _outputDim + j] = sum;
            }
        }

        return output;
    }

    /// <summary>
    /// Performs the backward pass to compute gradients.
    /// </summary>
    /// <param name="outputGradient">Gradient from the next layer [batchSize * numMembers, outputDim].</param>
    /// <returns>Gradient with respect to input [batchSize, inputDim].</returns>
    public Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_inputCache == null || _scaledInputCache == null)
        {
            throw new InvalidOperationException("Forward pass must be called before backward pass.");
        }

        int expandedBatchSize = outputGradient.Shape[0];
        int batchSize = expandedBatchSize / _numMembers;

        // Initialize gradients
        _weightsGrad = new Tensor<T>(_weights.Shape);
        _weightsGrad.Fill(_numOps.Zero);

        if (_bias != null)
        {
            _biasGrad = new Tensor<T>(_bias.Shape);
            _biasGrad.Fill(_numOps.Zero);
        }

        _rVectorsGrad = new Tensor<T>(_rVectors.Shape);
        _rVectorsGrad.Fill(_numOps.Zero);

        _sVectorsGrad = new Tensor<T>(_sVectors.Shape);
        _sVectorsGrad.Fill(_numOps.Zero);

        // Input gradient (accumulated across members)
        var inputGrad = new Tensor<T>([batchSize, _inputDim]);
        inputGrad.Fill(_numOps.Zero);

        for (int row = 0; row < expandedBatchSize; row++)
        {
            int batchIdx = row / _numMembers;
            int memberIdx = row % _numMembers;

            for (int j = 0; j < _outputDim; j++)
            {
                var grad = outputGradient[row * _outputDim + j];
                var sVal = _sVectors[memberIdx * _outputDim + j];

                // Bias gradient: sum of output gradients
                if (_biasGrad != null)
                {
                    _biasGrad[j] = _numOps.Add(_biasGrad[j], grad);
                }

                // s-vector gradient: grad * pre_s_output
                // pre_s_output = scaledInput @ weights[:, j]
                var preSOutput = _numOps.Zero;
                for (int i = 0; i < _inputDim; i++)
                {
                    preSOutput = _numOps.Add(preSOutput,
                        _numOps.Multiply(_scaledInputCache[row * _inputDim + i], _weights[i * _outputDim + j]));
                }
                _sVectorsGrad[memberIdx * _outputDim + j] = _numOps.Add(
                    _sVectorsGrad[memberIdx * _outputDim + j],
                    _numOps.Multiply(grad, preSOutput));

                // Gradient after s-vector: grad * s
                var gradAfterS = _numOps.Multiply(grad, sVal);

                for (int i = 0; i < _inputDim; i++)
                {
                    var rVal = _rVectors[memberIdx * _inputDim + i];
                    var inputVal = _inputCache[row * _inputDim + i];

                    // Weights gradient: scaledInput[i] * gradAfterS
                    _weightsGrad[i * _outputDim + j] = _numOps.Add(
                        _weightsGrad[i * _outputDim + j],
                        _numOps.Multiply(_scaledInputCache[row * _inputDim + i], gradAfterS));

                    // r-vector gradient: input[i] * weight[i,j] * gradAfterS
                    _rVectorsGrad[memberIdx * _inputDim + i] = _numOps.Add(
                        _rVectorsGrad[memberIdx * _inputDim + i],
                        _numOps.Multiply(_numOps.Multiply(inputVal, _weights[i * _outputDim + j]), gradAfterS));

                    // Input gradient: r[i] * weight[i,j] * gradAfterS
                    inputGrad[batchIdx * _inputDim + i] = _numOps.Add(
                        inputGrad[batchIdx * _inputDim + i],
                        _numOps.Multiply(_numOps.Multiply(rVal, _weights[i * _outputDim + j]), gradAfterS));
                }
            }
        }

        return inputGrad;
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

        var averaged = new Tensor<T>([batchSize, _outputDim]);
        var scale = _numOps.FromDouble(1.0 / _numMembers);

        for (int b = 0; b < batchSize; b++)
        {
            for (int j = 0; j < _outputDim; j++)
            {
                var sum = _numOps.Zero;
                for (int m = 0; m < _numMembers; m++)
                {
                    sum = _numOps.Add(sum, output[(b * _numMembers + m) * _outputDim + j]);
                }
                averaged[b * _outputDim + j] = _numOps.Multiply(sum, scale);
            }
        }

        return averaged;
    }

    /// <summary>
    /// Gets all trainable parameters as a single vector.
    /// </summary>
    public Vector<T> GetParameters()
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
    public void SetParameters(Vector<T> parameters)
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
    public Vector<T> GetParameterGradients()
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
                gradsList.Add(_numOps.Zero);
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
                    gradsList.Add(_numOps.Zero);
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
                gradsList.Add(_numOps.Zero);
        }

        if (_sVectorsGrad != null)
        {
            for (int i = 0; i < _sVectorsGrad.Length; i++)
                gradsList.Add(_sVectorsGrad[i]);
        }
        else
        {
            for (int i = 0; i < _sVectors.Length; i++)
                gradsList.Add(_numOps.Zero);
        }

        return new Vector<T>([.. gradsList]);
    }

    /// <summary>
    /// Updates parameters using the calculated gradients.
    /// </summary>
    public void UpdateParameters(T learningRate)
    {
        if (_weightsGrad != null)
        {
            for (int i = 0; i < _weights.Length; i++)
            {
                _weights[i] = _numOps.Subtract(_weights[i],
                    _numOps.Multiply(learningRate, _weightsGrad[i]));
            }
        }

        if (_bias != null && _biasGrad != null)
        {
            for (int i = 0; i < _bias.Length; i++)
            {
                _bias[i] = _numOps.Subtract(_bias[i],
                    _numOps.Multiply(learningRate, _biasGrad[i]));
            }
        }

        if (_rVectorsGrad != null)
        {
            for (int i = 0; i < _rVectors.Length; i++)
            {
                _rVectors[i] = _numOps.Subtract(_rVectors[i],
                    _numOps.Multiply(learningRate, _rVectorsGrad[i]));
            }
        }

        if (_sVectorsGrad != null)
        {
            for (int i = 0; i < _sVectors.Length; i++)
            {
                _sVectors[i] = _numOps.Subtract(_sVectors[i],
                    _numOps.Multiply(learningRate, _sVectorsGrad[i]));
            }
        }
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
    public Tensor<T> GetWeights() => _weights;

    /// <summary>
    /// Gets the r vectors tensor.
    /// </summary>
    public Tensor<T> GetRVectors() => _rVectors;

    /// <summary>
    /// Gets the s vectors tensor.
    /// </summary>
    public Tensor<T> GetSVectors() => _sVectors;
}
