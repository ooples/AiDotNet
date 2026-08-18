using AiDotNet.Attributes;
using AiDotNet.Autodiff;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Memory;

namespace AiDotNet.NeuralNetworks.Layers.SSM;

/// <summary>
/// Implements the Extended LSTM (xLSTM) layer from Hochreiter et al., 2024.
/// </summary>
/// <remarks>
/// <para>
/// xLSTM modernizes the classic LSTM architecture with two key innovations:
/// 1. <b>sLSTM (scalar LSTM)</b>: Enhanced gating with exponential activation functions and
///    a new memory mixing mechanism. Uses scalar (diagonal) memory cells.
/// 2. <b>mLSTM (matrix LSTM)</b>: Replaces the scalar memory cell with a matrix-valued memory,
///    connecting LSTMs to modern linear attention/state space models.
/// </para>
/// <para>
/// This layer implements the mLSTM variant, which is the more impactful innovation:
/// <code>
///   // Gate computations
///   i_t = exp(W_i * x_t + b_i)    // Input gate (exponential, not sigmoid!)
///   f_t = sigmoid(W_f * x_t + b_f) OR exp(W_f * x_t + b_f)  // Forget gate
///   o_t = sigmoid(W_o * x_t + b_o)  // Output gate
///
///   // Key-Value projections (connecting to linear attention)
///   k_t = W_k * x_t / sqrt(d)
///   v_t = W_v * x_t
///   q_t = W_q * x_t
///
///   // Matrix memory cell update (covariance-based)
///   C_t = f_t * C_{t-1} + i_t * v_t * k_t^T    // Matrix cell = gated outer product
///   n_t = f_t * n_{t-1} + i_t * k_t              // Normalizer state
///
///   // Output
///   h_t = o_t * (C_t * q_t) / max(|n_t^T * q_t|, 1)
/// </code>
/// </para>
/// <para>
/// The connection to linear attention: if f_t = 1 and i_t = 1, the matrix cell C_t accumulates
/// k*v outer products exactly like the state matrix in linear attention. The gates allow
/// selective forgetting and input scaling, which is what makes xLSTM competitive.
/// </para>
/// <para><b>For Beginners:</b> xLSTM is a modernized version of the classic LSTM (1997).
///
/// The original LSTM was the dominant sequence model for years, but was overtaken by Transformers.
/// xLSTM brings it back by fixing key limitations:
///
/// 1. <b>Exponential gating</b>: Instead of sigmoid (0 to 1), gates use exp() which can amplify
///    important signals, not just dampen them.
///
/// 2. <b>Matrix memory</b>: Instead of a vector cell, mLSTM uses a matrix. This is like having
///    a lookup table that maps keys to values, similar to attention but stored as a running sum.
///
/// The result: an LSTM that matches Transformer performance at scale while maintaining the
/// efficient O(1) per-step inference of RNNs.
/// </para>
/// <para>
/// <b>Reference:</b> Beck et al., "xLSTM: Extended Long Short-Term Memory", 2024.
/// https://arxiv.org/abs/2405.04517
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
[LayerCategory(LayerCategory.StateSpaceModel)]
[LayerCategory(LayerCategory.Recurrent)]
[LayerTask(LayerTask.SequenceModeling)]
[LayerTask(LayerTask.TemporalProcessing)]
[LayerProperty(IsTrainable = true, IsStateful = true, Cost = ComputeCost.High, TestInputShape = "4, 256", TestConstructorArgs = "4")]
// Shape-preserving; relations DISCOVERED by probing, roles read from the forward. Like every layer in
// this folder it takes seqLen = Shape[rank-2] and modelDim = Shape[rank-1], so rank 2 is
// [Time, Features] with NO batch axis. OutputAxesFor is generated from these layouts.
[TensorLayout(TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Output)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class ExtendedLSTMLayer<T> : LayerBase<T>, IShapeContract
{
    private readonly int _modelDimension;
    private readonly int _headDimension;
    private readonly int _numHeads;

    // Input gate projection: [modelDim, modelDim]
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _inputGateWeights;
    [TrainableParameter(Role = PersistentTensorRole.Biases)]

    private Tensor<T> _inputGateBias;

    // Forget gate projection: [modelDim, modelDim]
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _forgetGateWeights;
    [TrainableParameter(Role = PersistentTensorRole.Biases)]

    private Tensor<T> _forgetGateBias;

    // Output gate projection: [modelDim, modelDim]
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _outputGateWeights;
    [TrainableParameter(Role = PersistentTensorRole.Biases)]

    private Tensor<T> _outputGateBias;

    // Query, Key, Value projections: [modelDim, modelDim]
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _queryWeights;
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _keyWeights;
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _valueWeights;

    // Output projection: [modelDim, modelDim]
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _outputProjectionWeights;
    [TrainableParameter(Role = PersistentTensorRole.Biases)]

    private Tensor<T> _outputProjectionBias;

    // Cached values
    [Scratch]
    private Tensor<T>? _lastInput;
    [Scratch]
    private Tensor<T>? _lastOutput;
    [Scratch]
    private Tensor<T>? _lastCellStates;
    [Scratch]
    private Tensor<T>? _lastNormStates;
    [Scratch]
    private Tensor<T>? _lastInputGates;
    [Scratch]
    private Tensor<T>? _lastForgetGates;
    [Scratch]
    private Tensor<T>? _lastOutputGates;
    [Scratch]
    private Tensor<T>? _lastQ;
    [Scratch]
    private Tensor<T>? _lastK;
    [Scratch]
    private Tensor<T>? _lastV;
    [Scratch]
    private Tensor<T>? _lastHiddenPreProj;
    private int[]? _originalInputShape;

    // Fixed (non-trainable) unit-gamma / zero-beta for the mLSTM output
    // normalization. Beck et al. 2024 ("xLSTM", §2.3) normalize the mLSTM cell
    // output before the up-projection so the covariance-cell signal — a product
    // of sub-unit q/k/v projections that the max(|n·q|, 1) normalizer never
    // up-scales — stays at unit scale. Without it, stacking N cells collapses the
    // activations by ~5 orders of magnitude per layer (≈1e-77 by 4 layers), and
    // the backward pass underflows to zero gradient everywhere so no parameter
    // ever updates. Standardization (mean 0, var 1) with fixed gamma/beta keeps
    // the layer output unit-scale without introducing extra trainable parameters
    // (which would change the serialized parameter layout).
    private Tensor<T>? _outputNormGamma;
    private Tensor<T>? _outputNormBeta;

    // Gradients
    [Scratch]
    private Tensor<T>? _inputGateWeightsGradient;
    [Scratch]
    private Tensor<T>? _inputGateBiasGradient;
    [Scratch]
    private Tensor<T>? _forgetGateWeightsGradient;
    [Scratch]
    private Tensor<T>? _forgetGateBiasGradient;
    [Scratch]
    private Tensor<T>? _outputGateWeightsGradient;
    [Scratch]
    private Tensor<T>? _outputGateBiasGradient;
    [Scratch]
    private Tensor<T>? _queryWeightsGradient;
    [Scratch]
    private Tensor<T>? _keyWeightsGradient;
    [Scratch]
    private Tensor<T>? _valueWeightsGradient;
    [Scratch]
    private Tensor<T>? _outputProjectionWeightsGradient;
    [Scratch]
    private Tensor<T>? _outputProjectionBiasGradient;

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <summary>
    /// Gets the model dimension.
    /// </summary>
    public int ModelDimension => _modelDimension;

    /// <summary>
    /// Gets the number of heads for the matrix memory.
    /// </summary>
    public int NumHeads => _numHeads;

    /// <summary>
    /// Gets the dimension per head.
    /// </summary>
    public int HeadDimension => _headDimension;

    /// <summary>Construction state: the 'sequenceLength' the layer was built with.</summary>
    private readonly int _sequenceLength;

    /// <summary>
    /// Creates a new Extended LSTM (xLSTM) layer using the mLSTM (matrix memory) variant.
    /// </summary>
    /// <param name="sequenceLength">Maximum sequence length.</param>
    /// <param name="modelDimension">
    /// Model dimension (d_model). Default: 256.
    /// </param>
    /// <param name="numHeads">
    /// Number of heads for matrix memory. Default: 8.
    /// <para><b>For Beginners:</b> Each head maintains its own matrix memory cell.
    /// Must evenly divide modelDimension.</para>
    /// </param>
    /// <param name="activationFunction">Optional activation function applied to the final output.</param>
    /// <exception cref="ArgumentException">Thrown when parameters are invalid.</exception>
    public ExtendedLSTMLayer(
        int sequenceLength,
        int modelDimension = 256,
        int numHeads = 8,
        IActivationFunction<T>? activationFunction = null,
        IInitializationStrategy<T>? initializationStrategy = null)
        : base(
            [sequenceLength, modelDimension],
            [sequenceLength, modelDimension],
            activationFunction ?? new IdentityActivation<T>())
    {
        _sequenceLength = sequenceLength;
        InitializationStrategy = initializationStrategy ?? InitializationStrategies<T>.Eager;

        if (modelDimension <= 0)
            throw new ArgumentException($"Model dimension ({modelDimension}) must be positive.", nameof(modelDimension));
        if (numHeads <= 0)
            throw new ArgumentException($"Number of heads ({numHeads}) must be positive.", nameof(numHeads));
        if (modelDimension % numHeads != 0)
            throw new ArgumentException($"Model dimension ({modelDimension}) must be divisible by numHeads ({numHeads}).", nameof(numHeads));

        _modelDimension = modelDimension;
        _numHeads = numHeads;
        _headDimension = modelDimension / numHeads;

        _inputGateWeights = new Tensor<T>([modelDimension, modelDimension]);
        _inputGateBias = new Tensor<T>([modelDimension]);
        _forgetGateWeights = new Tensor<T>([modelDimension, modelDimension]);
        _forgetGateBias = new Tensor<T>([modelDimension]);
        _outputGateWeights = new Tensor<T>([modelDimension, modelDimension]);
        _outputGateBias = new Tensor<T>([modelDimension]);
        _queryWeights = new Tensor<T>([modelDimension, modelDimension]);
        _keyWeights = new Tensor<T>([modelDimension, modelDimension]);
        _valueWeights = new Tensor<T>([modelDimension, modelDimension]);
        _outputProjectionWeights = new Tensor<T>([modelDimension, modelDimension]);
        _outputProjectionBias = new Tensor<T>([modelDimension]);

        InitializeParameters();
    }

    private void InitializeParameters()
    {
        InitializeTensor(_inputGateWeights);
        _inputGateBias.Fill(NumOps.Zero);
        InitializeTensor(_forgetGateWeights);
        // Forget gate bias initialized to positive values for long memory (LSTM best practice)
        for (int i = 0; i < _forgetGateBias.Length; i++)
            _forgetGateBias[i] = NumOps.FromDouble(1.0);
        InitializeTensor(_outputGateWeights);
        _outputGateBias.Fill(NumOps.Zero);
        InitializeTensor(_queryWeights);
        InitializeTensor(_keyWeights);
        InitializeTensor(_valueWeights);
        InitializeTensor(_outputProjectionWeights);
        _outputProjectionBias.Fill(NumOps.Zero);
    }

    private void InitializeTensor(Tensor<T> tensor)
    {
        InitializeLayerWeights(tensor, tensor.Shape[0], tensor.Shape[1]);
    }

    /// <inheritdoc />
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        _originalInputShape = input._shape;

        int rank = input.Shape.Length;
        int seqLen = rank >= 2 ? input.Shape[rank - 2] : 1;
        int modelDim = input.Shape[rank - 1];

        int batchSize = 1;
        for (int d = 0; d < rank - 2; d++)
            batchSize *= input.Shape[d];
        if (rank < 3) batchSize = 1;

        var input3D = rank == 2
            ? Engine.Reshape(input, new[] { 1, seqLen, modelDim })
            : Engine.Reshape(input, new[] { batchSize, seqLen, modelDim });

        _lastInput = input3D;

        int headBatch = batchSize * _numHeads;
        var outputList = new List<Tensor<T>>(seqLen);
        var qList = new List<Tensor<T>>(seqLen);
        var kList = new List<Tensor<T>>(seqLen);
        var vList = new List<Tensor<T>>(seqLen);
        var iList = new List<Tensor<T>>(seqLen);
        var fList = new List<Tensor<T>>(seqLen);
        var oList = new List<Tensor<T>>(seqLen);
        var hiddenList = new List<Tensor<T>>(seqLen);

        // Keep every recurrent update on the engine graph. The former scalar writes made C_t and
        // n_t detached values; the residual allowed an input gradient to exist, but it was not the
        // derivative of the actual mLSTM output and none of the gate/Q/K/V weights could train.
        var cellState = Tensor<T>.CreateDefault(
            new[] { headBatch, _headDimension, _headDimension }, NumOps.Zero);
        var normState = Tensor<T>.CreateDefault(
            new[] { headBatch, _headDimension, 1 }, NumOps.Zero);
        var mState = Tensor<T>.CreateDefault(new[] { headBatch, 1, 1 }, NumOps.FromDouble(-1e30));
        var keyScale = NumOps.FromDouble(1.0 / Math.Sqrt(_headDimension));

        for (int t = 0; t < seqLen; t++)
        {
            var x_t = Engine.TensorSliceAxis(input3D, axis: 1, index: t); // [batch, modelDim]

            // Gate computations
            var iGateRaw = Engine.TensorBroadcastAdd(
                Engine.TensorMatMul(x_t, _inputGateWeights),
                Engine.Reshape(_inputGateBias, new[] { 1, _modelDimension }));
            var fGateRaw = Engine.TensorBroadcastAdd(
                Engine.TensorMatMul(x_t, _forgetGateWeights),
                Engine.Reshape(_forgetGateBias, new[] { 1, _modelDimension }));
            var oGate = Engine.Sigmoid(Engine.TensorBroadcastAdd(
                Engine.TensorMatMul(x_t, _outputGateWeights),
                Engine.Reshape(_outputGateBias, new[] { 1, _modelDimension })));

            var fGate = Engine.Sigmoid(fGateRaw);

            // Q, K, V projections
            var q = Engine.TensorMatMul(x_t, _queryWeights);
            var k = Engine.TensorMultiplyScalar(
                Engine.TensorMatMul(x_t, _keyWeights), keyScale);
            var v = Engine.TensorMatMul(x_t, _valueWeights);

            Tensor<T> AsHeads(Tensor<T> z) => Engine.Reshape(z,
                new[] { headBatch, _headDimension });
            var qHead = AsHeads(q);
            var kHead = AsHeads(k);
            var vHead = AsHeads(v);
            var iHead = AsHeads(iGateRaw);
            var fHead = AsHeads(fGate);
            var oHead = AsHeads(oGate);

            // The paper uses one stabilizer/gate scalar per head; match the historical forward's
            // first coordinate selection, but use a tape-tracked slice rather than scalar reads.
            var logI = Engine.Reshape(Engine.TensorSlice(iHead,
                new[] { 0, 0 }, new[] { headBatch, 1 }), new[] { headBatch, 1, 1 });
            var fScalar = Engine.Reshape(Engine.TensorSlice(fHead,
                new[] { 0, 0 }, new[] { headBatch, 1 }), new[] { headBatch, 1, 1 });
            var logF = Engine.TensorLog(Engine.TensorMax(
                fScalar, Tensor<T>.CreateDefault(fScalar.Shape.ToArray(), NumOps.FromDouble(1e-30))));
            var carryLog = Engine.TensorAdd(logF, mState);
            var mNew = Engine.TensorMax(carryLog, logI);
            var iScale = Engine.TensorExp(Engine.TensorSubtract(logI, mNew));
            var fScale = Engine.TensorExp(Engine.TensorSubtract(carryLog, mNew));
            mState = mNew;

            var qCol = Engine.Reshape(qHead, new[] { headBatch, _headDimension, 1 });
            var kCol = Engine.Reshape(kHead, new[] { headBatch, _headDimension, 1 });
            var vCol = Engine.Reshape(vHead, new[] { headBatch, _headDimension, 1 });
            var kRow = Engine.TensorPermute(kCol, new[] { 0, 2, 1 });
            cellState = Engine.TensorAdd(
                Engine.TensorBroadcastMultiply(cellState, fScale),
                Engine.TensorBroadcastMultiply(Engine.BatchMatMul(vCol, kRow), iScale));
            normState = Engine.TensorAdd(
                Engine.TensorBroadcastMultiply(normState, fScale),
                Engine.TensorBroadcastMultiply(kCol, iScale));

            var numerator = Engine.BatchMatMul(cellState, qCol);
            var denominator = Engine.TensorMax(
                Engine.TensorAbs(Engine.BatchMatMul(
                    Engine.TensorPermute(normState, new[] { 0, 2, 1 }), qCol)),
                Tensor<T>.CreateDefault(new[] { headBatch, 1, 1 }, NumOps.One));
            var oScalar = Engine.Reshape(Engine.TensorSlice(oHead,
                new[] { 0, 0 }, new[] { headBatch, 1 }), new[] { headBatch, 1, 1 });
            var normalized = Engine.TensorDivide(
                Engine.TensorBroadcastMultiply(numerator, oScalar),
                Engine.TensorTile(denominator, new[] { 1, _headDimension, 1 }));
            var h_t = Engine.Reshape(normalized, new[] { batchSize, _modelDimension });

            iList.Add(Engine.Reshape(iGateRaw, new[] { batchSize, 1, _modelDimension }));
            fList.Add(Engine.Reshape(fGate, new[] { batchSize, 1, _modelDimension }));
            oList.Add(Engine.Reshape(oGate, new[] { batchSize, 1, _modelDimension }));
            qList.Add(Engine.Reshape(q, new[] { batchSize, 1, _modelDimension }));
            kList.Add(Engine.Reshape(k, new[] { batchSize, 1, _modelDimension }));
            vList.Add(Engine.Reshape(v, new[] { batchSize, 1, _modelDimension }));
            hiddenList.Add(Engine.Reshape(h_t, new[] { batchSize, 1, _modelDimension }));

            // Output projection
            var y_t = Engine.TensorMatMul(h_t, _outputProjectionWeights);
            var outBias = Engine.Reshape(_outputProjectionBias, new[] { 1, _modelDimension });
            y_t = Engine.TensorBroadcastAdd(y_t, outBias);

            outputList.Add(Engine.Reshape(y_t, new[] { batchSize, 1, _modelDimension }));
        }

        // Assemble the [batch, seqLen, modelDim] output on the tape so gradients
        // reach the output-projection weights (and bias).
        var output = Engine.TensorConcatenate(outputList.ToArray(), axis: 1);

        // Residual + output normalization — the xLSTM block (Beck et al. 2024,
        // §2.3 / Fig. 3: each mLSTM cell lives in a normalized residual block).
        //
        // Two problems are fixed here together:
        //  1. Gradient flow. The covariance-cell recurrence above is computed with
        //     in-place scalar updates (the matrix state C_t / normalizer n_t are
        //     inherently sequential), so it is OFF the autodiff tape: the gradient
        //     cannot cross it to reach the input/gate/q/k/v projections or any
        //     upstream layer. The residual skip `output + input` re-attaches the
        //     block output to its input ON the tape, so gradients reach every
        //     projection in this cell AND propagate to the layers below it.
        //  2. Signal collapse. The cell output C_t q_t is a product of sub-unit
        //     projections that the max(|n·q|, 1) normalizer never up-scales, so
        //     stacking N cells shrinks the activations ~5 orders of magnitude per
        //     layer (≈1e-77 by 4 layers) and the backward pass underflows to zero.
        //     Normalizing the block output to unit scale stops the collapse.
        //
        // Fixed unit-gamma / zero-beta (created once) keeps the serialized
        // parameter layout unchanged (no extra trainable tensors).
        output = Engine.TensorAdd(output, input3D);
        if (_outputNormGamma is null || _outputNormBeta is null)
        {
            var gamma = new Tensor<T>(new[] { _modelDimension });
            var beta = new Tensor<T>(new[] { _modelDimension });
            for (int j = 0; j < _modelDimension; j++)
            {
                gamma[j] = NumOps.One;
                beta[j] = NumOps.Zero;
            }
            _outputNormGamma = gamma;
            _outputNormBeta = beta;
        }
        output = Engine.LayerNorm(output, _outputNormGamma, _outputNormBeta, 1e-5, out _, out _);

        _lastCellStates = cellState;
        _lastNormStates = normState;
        _lastInputGates = Engine.TensorConcatenate(iList.ToArray(), 1);
        _lastForgetGates = Engine.TensorConcatenate(fList.ToArray(), 1);
        _lastOutputGates = Engine.TensorConcatenate(oList.ToArray(), 1);
        _lastQ = Engine.TensorConcatenate(qList.ToArray(), 1);
        _lastK = Engine.TensorConcatenate(kList.ToArray(), 1);
        _lastV = Engine.TensorConcatenate(vList.ToArray(), 1);
        _lastHiddenPreProj = Engine.TensorConcatenate(hiddenList.ToArray(), 1);

        var result = ApplyActivation(output);
        _lastOutput = result;

        if (rank == 2)
            return Engine.Reshape(result, new[] { seqLen, _modelDimension });

        var outputShape = new int[rank];
        for (int i = 0; i < rank - 2; i++)
            outputShape[i] = input.Shape[i];
        outputShape[rank - 2] = seqLen;
        outputShape[rank - 1] = _modelDimension;
        return Engine.Reshape(result, outputShape);
    }

    #region Parameter Management

    /// <inheritdoc />
    public override void UpdateParameters(T learningRate)
    {
        if (_inputGateWeightsGradient == null)
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");

        T negLR = NumOps.Negate(learningRate);
        _inputGateWeights = Engine.TensorAdd(_inputGateWeights, Engine.TensorMultiplyScalar(_inputGateWeightsGradient, negLR));
        _inputGateBias = Engine.TensorAdd(_inputGateBias, Engine.TensorMultiplyScalar(_inputGateBiasGradient!, negLR));
        _forgetGateWeights = Engine.TensorAdd(_forgetGateWeights, Engine.TensorMultiplyScalar(_forgetGateWeightsGradient!, negLR));
        _forgetGateBias = Engine.TensorAdd(_forgetGateBias, Engine.TensorMultiplyScalar(_forgetGateBiasGradient!, negLR));
        _outputGateWeights = Engine.TensorAdd(_outputGateWeights, Engine.TensorMultiplyScalar(_outputGateWeightsGradient!, negLR));
        _outputGateBias = Engine.TensorAdd(_outputGateBias, Engine.TensorMultiplyScalar(_outputGateBiasGradient!, negLR));
        _queryWeights = Engine.TensorAdd(_queryWeights, Engine.TensorMultiplyScalar(_queryWeightsGradient!, negLR));
        _keyWeights = Engine.TensorAdd(_keyWeights, Engine.TensorMultiplyScalar(_keyWeightsGradient!, negLR));
        _valueWeights = Engine.TensorAdd(_valueWeights, Engine.TensorMultiplyScalar(_valueWeightsGradient!, negLR));
        _outputProjectionWeights = Engine.TensorAdd(_outputProjectionWeights, Engine.TensorMultiplyScalar(_outputProjectionWeightsGradient!, negLR));
        _outputProjectionBias = Engine.TensorAdd(_outputProjectionBias, Engine.TensorMultiplyScalar(_outputProjectionBiasGradient!, negLR));

        // Register trainable parameters for tape-based autodiff
        RegisterTrainableParameter(_inputGateWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_inputGateBias, PersistentTensorRole.Biases);
        RegisterTrainableParameter(_forgetGateWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_forgetGateBias, PersistentTensorRole.Biases);
        RegisterTrainableParameter(_outputGateWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_outputGateBias, PersistentTensorRole.Biases);
        RegisterTrainableParameter(_queryWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_keyWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_valueWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_outputProjectionWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_outputProjectionBias, PersistentTensorRole.Biases);

    }

    private Tensor<T>[] GetAllTensors() =>
    [
        _inputGateWeights, _inputGateBias,
        _forgetGateWeights, _forgetGateBias,
        _outputGateWeights, _outputGateBias,
        _queryWeights, _keyWeights, _valueWeights,
        _outputProjectionWeights, _outputProjectionBias
    ];

    public override Vector<T> GetParameterGradients()
    {
        if (_inputGateWeightsGradient == null) return new Vector<T>(ParameterCountHelper.ToFlatVectorSize(ParameterCount));

        Vector<T> G(Tensor<T>? grad, Tensor<T> param) =>
            grad != null ? new Vector<T>(grad.ToArray()) : new Vector<T>(param.Length);

        return Vector<T>.Concatenate(
            G(_inputGateWeightsGradient, _inputGateWeights),
            G(_inputGateBiasGradient, _inputGateBias),
            G(_forgetGateWeightsGradient, _forgetGateWeights),
            G(_forgetGateBiasGradient, _forgetGateBias),
            G(_outputGateWeightsGradient, _outputGateWeights),
            G(_outputGateBiasGradient, _outputGateBias),
            G(_queryWeightsGradient, _queryWeights),
            G(_keyWeightsGradient, _keyWeights),
            G(_valueWeightsGradient, _valueWeights),
            G(_outputProjectionWeightsGradient, _outputProjectionWeights),
            G(_outputProjectionBiasGradient, _outputProjectionBias));
    }

    public override void ClearGradients()
    {
        base.ClearGradients();
        _inputGateWeightsGradient = null; _inputGateBiasGradient = null;
        _forgetGateWeightsGradient = null; _forgetGateBiasGradient = null;
        _outputGateWeightsGradient = null; _outputGateBiasGradient = null;
        _queryWeightsGradient = null; _keyWeightsGradient = null; _valueWeightsGradient = null;
        _outputProjectionWeightsGradient = null; _outputProjectionBiasGradient = null;
    }

    /// <inheritdoc />
    public override void ResetState()
    {
        _lastInput = null;
        _lastOutput = null;
        _lastCellStates = null;
        _lastNormStates = null;
        _lastInputGates = null;
        _lastForgetGates = null;
        _lastOutputGates = null;
        _lastQ = null;
        _lastK = null;
        _lastV = null;
        _lastHiddenPreProj = null;
        _originalInputShape = null;
        _inputGateWeightsGradient = null;
        _inputGateBiasGradient = null;
        _forgetGateWeightsGradient = null;
        _forgetGateBiasGradient = null;
        _outputGateWeightsGradient = null;
        _outputGateBiasGradient = null;
        _queryWeightsGradient = null;
        _keyWeightsGradient = null;
        _valueWeightsGradient = null;
        _outputProjectionWeightsGradient = null;
        _outputProjectionBiasGradient = null;
    }

    #endregion

    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        metadata["ModelDimension"] = _modelDimension.ToString();
        metadata["NumHeads"] = _numHeads.ToString();
        metadata["HeadDimension"] = _headDimension.ToString();
        return metadata;
    }

    /// <summary>
    /// Gets the output projection weights for external inspection.
    /// </summary>
    public Tensor<T> GetOutputProjectionWeights() => _outputProjectionWeights;

    /// <summary>
    /// Gets the forget gate weights for external inspection.
    /// </summary>
    public Tensor<T> GetForgetGateWeights() => _forgetGateWeights;
}
