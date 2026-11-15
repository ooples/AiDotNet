using AiDotNet.Attributes;
using AiDotNet.Autodiff;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Memory;

namespace AiDotNet.NeuralNetworks.Layers.SSM;

/// <summary>
/// Implements the Megalodon layer from "Megalodon: Efficient LLM Pretraining and Inference with
/// Unlimited Context Length" (Ma et al., 2024, arXiv:2404.08801).
/// </summary>
/// <remarks>
/// <para>
/// Megalodon extends the MEGA (Moving Average Equipped Gated Attention) architecture with two key
/// innovations that enable unlimited context length:
/// <code>
///   1. CEMA (Complex Exponential Moving Average):
///      h_t = alpha * h_{t-1} + (1 - alpha) * x_t
///      where alpha is a complex-valued coefficient (alpha_real + i * alpha_imag).
///      The complex coefficients create damped oscillatory dynamics, enabling the model
///      to capture periodic and quasi-periodic patterns in sequences naturally.
///
///   2. Timestep Normalization:
///      Normalizes hidden states per-timestep to prevent magnitude drift during
///      long-context processing. Without this, EMA states can grow unbounded
///      over very long sequences, degrading performance.
///
///   3. Gated Attention:
///      Combines CEMA output with gated attention for expressive sequence mixing.
///      gate_t = sigmoid(W_g * x_t + b_g)
///      y_t = gate_t * attention_out_t + (1 - gate_t) * cema_out_t
///
///   4. Chunk-wise Processing:
///      For efficiency on very long contexts, the sequence can be processed in chunks,
///      with CEMA state carried across chunk boundaries.
/// </code>
/// </para>
/// <para>
/// The CEMA mechanism is the core innovation. Standard EMA uses real-valued decay coefficients,
/// which can only model exponential decay. By making alpha complex-valued, the state evolves as
/// a damped oscillation: |alpha| controls the decay rate, while arg(alpha) controls the oscillation
/// frequency. This gives each EMA dimension a unique frequency response, enabling the model to
/// selectively attend to different periodicities in the input.
/// </para>
/// <para>
/// Timestep normalization is critical for unlimited context. Without it, the accumulated EMA state
/// grows with sequence length, causing numerical instability. By normalizing per-timestep, Megalodon
/// maintains stable representations regardless of context length.
/// </para>
/// <para><b>For Beginners:</b> Megalodon is like having a bank of tuning forks that each ring at
/// different frequencies.
///
/// Imagine you are analyzing a complex audio signal:
/// - A standard EMA is like a simple echo that fades over time -- it can only remember things
///   that happened recently, and the memory just gets quieter (exponential decay).
/// - Megalodon's CEMA is like a set of tuning forks. Each fork vibrates at its own frequency.
///   When you tap a tuning fork (give it input), it rings and slowly fades, but it OSCILLATES
///   while fading. This means it naturally picks up patterns that repeat at that frequency.
///
/// The "complex" in CEMA means each decay coefficient has two parts:
/// - A real part: controls how fast the oscillation fades (the damping)
/// - An imaginary part: controls how fast it oscillates (the frequency)
///
/// Together, many CEMA dimensions with different frequencies act like a Fourier-style analyzer
/// that decomposes the input into its frequency components, all within linear O(n) complexity.
///
/// The timestep normalization prevents the "tuning forks" from getting too loud over very long
/// sequences, which is why Megalodon can handle unlimited context length.
/// </para>
/// <para>
/// <b>Reference:</b> Ma et al., "Megalodon: Efficient LLM Pretraining and Inference with Unlimited
/// Context Length", arXiv:2404.08801, 2024.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
[LayerCategory(LayerCategory.StateSpaceModel)]
[LayerCategory(LayerCategory.Attention)]
[LayerTask(LayerTask.SequenceModeling)]
[LayerProperty(IsTrainable = true, IsStateful = true, Cost = ComputeCost.High, TestInputShape = "4, 256", TestConstructorArgs = "4")]
// Shape-preserving. Relations discovered by probing; roles read from the forward - this folder's
// convention is seqLen = Shape[rank-2], modelDim = Shape[rank-1], so rank 2 is [Time, Features].
[TensorLayout(TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Output)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Time, TensorAxis.Features, Direction = TensorLayoutDirection.Output)]
[AutoParameters]
public partial class MegalodonLayer<T> : LayerBase<T>, IShapeContract
{
    private readonly int _modelDimension;
    private readonly int _numHeads;
    private readonly int _headDimension;
    private readonly int _emaDimension;

    // CEMA parameters: complex-valued alpha = alpha_real + i * alpha_imag
    // Stored as real/imag pairs: [emaDimension] each
    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _emaAlphaReal;
    [TrainableParameter(Role = PersistentTensorRole.Weights)]
    private Tensor<T> _emaAlphaImag;

    // EMA input projection: [modelDim, emaDimension]
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _emaInputWeights;
    [TrainableParameter(Role = PersistentTensorRole.Biases)]

    private Tensor<T> _emaInputBias;

    // EMA output projection: [emaDimension, modelDim]
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _emaOutputWeights;
    [TrainableParameter(Role = PersistentTensorRole.Biases)]

    private Tensor<T> _emaOutputBias;

    // Timestep normalization: [emaDimension] each
    [TrainableParameter(Role = PersistentTensorRole.NormalizationParams)]
    private Tensor<T> _tsNormGamma;
    [TrainableParameter(Role = PersistentTensorRole.NormalizationParams)]
    private Tensor<T> _tsNormBeta;

    // Q, K, V projections: [modelDim, modelDim]
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _queryWeights;
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _keyWeights;
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _valueWeights;

    // Gating: [modelDim, modelDim]
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _gateWeights;
    [TrainableParameter(Role = PersistentTensorRole.Biases)]

    private Tensor<T> _gateBias;

    // Output projection: [modelDim, modelDim]
    [TrainableParameter(Role = PersistentTensorRole.Weights)]

    private Tensor<T> _outputProjectionWeights;
    [TrainableParameter(Role = PersistentTensorRole.Biases)]

    private Tensor<T> _outputProjectionBias;

    // Cached forward pass values
    private Tensor<T>? _lastEmaOutputPreNorm;
    private Tensor<T>? _lastEmaStdInv;
    private int[]? _originalInputShape;

    // Gradients
    private Tensor<T>? _emaAlphaRealGradient;
    private Tensor<T>? _emaAlphaImagGradient;
    private Tensor<T>? _emaInputWeightsGradient;
    private Tensor<T>? _emaInputBiasGradient;
    private Tensor<T>? _emaOutputWeightsGradient;
    private Tensor<T>? _emaOutputBiasGradient;
    private Tensor<T>? _tsNormGammaGradient;
    private Tensor<T>? _tsNormBetaGradient;
    private Tensor<T>? _queryWeightsGradient;
    private Tensor<T>? _keyWeightsGradient;
    private Tensor<T>? _valueWeightsGradient;
    private Tensor<T>? _gateWeightsGradient;
    private Tensor<T>? _gateBiasGradient;
    private Tensor<T>? _outputProjectionWeightsGradient;
    private Tensor<T>? _outputProjectionBiasGradient;

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <summary>
    /// Gets the model dimension.
    /// </summary>
    public int ModelDimension => _modelDimension;

    /// <summary>
    /// Gets the number of attention heads.
    /// </summary>
    public int NumHeads => _numHeads;

    /// <summary>
    /// Gets the dimension per attention head.
    /// </summary>
    public int HeadDimension => _headDimension;

    /// <summary>
    /// Gets the EMA state dimension (number of complex EMA channels).
    /// </summary>
    public int EmaDimension => _emaDimension;

    /// <summary>
    /// Creates a new Megalodon layer with CEMA, timestep normalization, and gated attention.
    /// </summary>
    /// <param name="sequenceLength">Maximum sequence length.</param>
    /// <param name="modelDimension">
    /// Model dimension (d_model). Default: 256.
    /// <para><b>For Beginners:</b> The width of data flowing through the layer. Larger values
    /// give the model more capacity to represent complex patterns but use more memory.</para>
    /// </param>
    /// <param name="numHeads">
    /// Number of attention heads. Default: 8.
    /// <para><b>For Beginners:</b> Each head learns to attend to different types of relationships
    /// in the input. Must evenly divide modelDimension.</para>
    /// </param>
    /// <param name="emaDimension">
    /// Number of complex EMA channels. Default: 16.
    /// <para><b>For Beginners:</b> Each EMA channel acts like a tuning fork at a different frequency.
    /// More channels allow the model to track more distinct periodicities.</para>
    /// </param>
    /// <param name="activationFunction">Optional activation function applied to the final output.</param>
    /// <exception cref="ArgumentException">Thrown when parameters are invalid.</exception>
    /// <summary>Construction state: kept so the generated clone factory can rebuild this layer.</summary>
    private readonly int _sequenceLength;

    public MegalodonLayer(
        int sequenceLength,
        int modelDimension = 256,
        int numHeads = 8,
        int emaDimension = 16,
        IActivationFunction<T>? activationFunction = null,
        IInitializationStrategy<T>? initializationStrategy = null)
        : base(
            [sequenceLength, modelDimension],
            [sequenceLength, modelDimension],
            activationFunction ?? new IdentityActivation<T>())
    {
        _sequenceLength = sequenceLength;
        InitializationStrategy = initializationStrategy ?? InitializationStrategies<T>.Eager;

        if (sequenceLength <= 0)
            throw new ArgumentException($"Sequence length ({sequenceLength}) must be positive.", nameof(sequenceLength));
        if (modelDimension <= 0)
            throw new ArgumentException($"Model dimension ({modelDimension}) must be positive.", nameof(modelDimension));
        if (numHeads <= 0)
            throw new ArgumentException($"Number of heads ({numHeads}) must be positive.", nameof(numHeads));
        if (modelDimension % numHeads != 0)
            throw new ArgumentException($"Model dimension ({modelDimension}) must be divisible by numHeads ({numHeads}).", nameof(numHeads));
        if (emaDimension <= 0)
            throw new ArgumentException($"EMA dimension ({emaDimension}) must be positive.", nameof(emaDimension));

        _modelDimension = modelDimension;
        _numHeads = numHeads;
        _headDimension = modelDimension / numHeads;
        _emaDimension = emaDimension;

        // CEMA complex alpha parameters
        _emaAlphaReal = new Tensor<T>([emaDimension]);
        _emaAlphaImag = new Tensor<T>([emaDimension]);

        // EMA projections
        _emaInputWeights = new Tensor<T>([modelDimension, emaDimension]);
        _emaInputBias = new Tensor<T>([emaDimension]);
        _emaOutputWeights = new Tensor<T>([emaDimension, modelDimension]);
        _emaOutputBias = new Tensor<T>([modelDimension]);

        // Timestep normalization parameters
        _tsNormGamma = new Tensor<T>([emaDimension]);
        _tsNormBeta = new Tensor<T>([emaDimension]);

        // Attention projections
        _queryWeights = new Tensor<T>([modelDimension, modelDimension]);
        _keyWeights = new Tensor<T>([modelDimension, modelDimension]);
        _valueWeights = new Tensor<T>([modelDimension, modelDimension]);

        // Gate
        _gateWeights = new Tensor<T>([modelDimension, modelDimension]);
        _gateBias = new Tensor<T>([modelDimension]);

        // Output projection
        _outputProjectionWeights = new Tensor<T>([modelDimension, modelDimension]);
        _outputProjectionBias = new Tensor<T>([modelDimension]);

        InitializeParameters();
    }

    private void InitializeParameters()
    {
        // CEMA alpha: initialize magnitudes close to 1 (slow decay) with spread of frequencies.
        // alpha = magnitude * exp(i * theta), we store as (real, imag).
        // magnitude ~ 0.9-0.999 for long memory, theta spread across [0, pi].
        for (int i = 0; i < _emaDimension; i++)
        {
            double magnitude = 0.9 + 0.099 * Random.NextDouble(); // [0.9, 0.999]
            double theta = Math.PI * (i + 1) / (_emaDimension + 1);  // Spread frequencies
            _emaAlphaReal[i] = NumOps.FromDouble(magnitude * Math.Cos(theta));
            _emaAlphaImag[i] = NumOps.FromDouble(magnitude * Math.Sin(theta));
        }

        InitializeTensor2D(_emaInputWeights);
        _emaInputBias.Fill(NumOps.Zero);
        InitializeTensor2D(_emaOutputWeights);
        _emaOutputBias.Fill(NumOps.Zero);

        // Timestep norm: gamma=1, beta=0 (identity initialization)
        _tsNormGamma.Fill(NumOps.One);
        _tsNormBeta.Fill(NumOps.Zero);

        InitializeTensor2D(_queryWeights);
        InitializeTensor2D(_keyWeights);
        InitializeTensor2D(_valueWeights);
        InitializeTensor2D(_gateWeights);
        _gateBias.Fill(NumOps.Zero);
        InitializeTensor2D(_outputProjectionWeights);
        _outputProjectionBias.Fill(NumOps.Zero);
    }

    private void InitializeTensor2D(Tensor<T> tensor)
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

        // Step 1: CEMA (Complex Exponential Moving Average) via kernel convolution
        var cemaOutput = CEMAKernelForward(input3D, batchSize, seqLen);

        // Step 2: Project CEMA output back to model dimension
        var cemaFlat = Engine.Reshape(cemaOutput, new[] { batchSize * seqLen, _emaDimension });
        var emaProjected = Engine.TensorMatMul(cemaFlat, _emaOutputWeights);
        emaProjected = Engine.TensorAdd(emaProjected, Engine.Reshape(_emaOutputBias, new[] { 1, _modelDimension }));
        var emaOut3D = Engine.Reshape(emaProjected, new[] { batchSize, seqLen, _modelDimension });

        // Step 3: Q, K, V projections for gated attention
        var inputFlat = Engine.Reshape(input3D, new[] { batchSize * seqLen, _modelDimension });
        var q = Engine.Reshape(Engine.TensorMatMul(inputFlat, _queryWeights), new[] { batchSize, seqLen, _modelDimension });
        var k = Engine.Reshape(Engine.TensorMatMul(inputFlat, _keyWeights), new[] { batchSize, seqLen, _modelDimension });
        var v = Engine.Reshape(Engine.TensorMatMul(inputFlat, _valueWeights), new[] { batchSize, seqLen, _modelDimension });

        // Step 4: Scaled dot-product attention per head (causal)
        var attentionOutput = CausalLinearAttention.ScaledDotProduct(
            Engine, q, k, v, _numHeads, causal: true);

        // Step 5: Gating between attention and CEMA outputs
        var gateRaw = Engine.Reshape(Engine.TensorAdd(
            Engine.TensorMatMul(inputFlat, _gateWeights),
            Engine.Reshape(_gateBias, new[] { 1, _modelDimension })), new[] { batchSize, seqLen, _modelDimension });
        var gate = Engine.Sigmoid(gateRaw);

        // y = gate * attention_out + (1 - gate) * cema_projected
        var ones = CreateOnesLike(gate);
        var oneMinusGate = Engine.TensorSubtract(ones, gate);
        var mixed = Engine.TensorAdd(
            Engine.TensorMultiply(gate, attentionOutput),
            Engine.TensorMultiply(oneMinusGate, emaOut3D));

        // Step 6: Output projection
        var mixedFlat = Engine.Reshape(mixed, new[] { batchSize * seqLen, _modelDimension });
        var outputFlat = Engine.TensorMatMul(mixedFlat, _outputProjectionWeights);
        outputFlat = Engine.TensorAdd(outputFlat, Engine.Reshape(_outputProjectionBias, new[] { 1, _modelDimension }));
        var output3D = Engine.Reshape(outputFlat, new[] { batchSize, seqLen, _modelDimension });

        var result = ApplyActivation(output3D);

        if (rank == 2)
            return Engine.Reshape(result, new[] { seqLen, _modelDimension });

        var outputShape = new int[rank];
        for (int i = 0; i < rank - 2; i++)
            outputShape[i] = input.Shape[i];
        outputShape[rank - 2] = seqLen;
        outputShape[rank - 1] = _modelDimension;
        return Engine.Reshape(result, outputShape);
    }

    /// <summary>
    /// Complex Exponential Moving Average forward pass with timestep normalization.
    /// </summary>
    /// <remarks>
    /// Computes h_t = alpha * h_{t-1} + (1 - alpha) * x_t where alpha is complex-valued.
    /// The real part of the state is extracted and normalized per-timestep.
    /// </remarks>
    /// <summary>
    /// Fused complex EMA forward. Each EMA channel is one width-one SSM group:
    /// h_t = alpha * h_(t-1) + (1-alpha) * x_t; y_t = Re(h_t).
    /// </summary>
    private Tensor<T> CEMAKernelForward(Tensor<T> input, int batchSize, int seqLen)
    {
        var inputFlat = Engine.Reshape(input, new[] { batchSize * seqLen, _modelDimension });
        var emaInput = Engine.TensorMatMul(inputFlat, _emaInputWeights);
        emaInput = Engine.TensorAdd(
            emaInput,
            Engine.Reshape(_emaInputBias, new[] { 1, _emaDimension }));
        var emaInput3D = Engine.Reshape(
            emaInput,
            new[] { batchSize, seqLen, _emaDimension });

        // ComplexDiagonalSsmScanForward accepts [B,T,G,W]. With G=emaDim,
        // W=state=1, its maps are scalar and this is an O(B*T*emaDim)
        // recurrence with one analytic tape node on CPU and every GPU backend.
        var one = new Tensor<T>(new[] { _emaDimension });
        one.Fill(NumOps.One);
        var zero = new Tensor<T>(new[] { _emaDimension });
        zero.Fill(NumOps.Zero);
        var inputMapReal = Engine.Reshape(
            Engine.TensorSubtract(one, _emaAlphaReal),
            new[] { _emaDimension, 1, 1 });
        var inputMapImag = Engine.Reshape(
            Engine.TensorNegate(_emaAlphaImag),
            new[] { _emaDimension, 1, 1 });
        var outputMapReal = Engine.Reshape(one, new[] { _emaDimension, 1, 1 });
        var outputMapImag = Engine.Reshape(zero, new[] { _emaDimension, 1, 1 });
        var skip = Engine.Reshape(zero, new[] { _emaDimension, 1 });
        var outputPreNorm = Engine.Reshape(
            Engine.ComplexDiagonalSsmScanForward(
                Engine.Reshape(
                    emaInput3D,
                    new[] { batchSize, seqLen, _emaDimension, 1 }),
                Engine.Reshape(_emaAlphaReal, new[] { _emaDimension, 1 }),
                Engine.Reshape(_emaAlphaImag, new[] { _emaDimension, 1 }),
                inputMapReal,
                inputMapImag,
                outputMapReal,
                outputMapImag,
                skip),
            new[] { batchSize, seqLen, _emaDimension });

        _lastEmaOutputPreNorm = outputPreNorm;
        const double eps = 1e-5;
        var normed = Engine.LayerNorm(
            outputPreNorm,
            _tsNormGamma,
            _tsNormBeta,
            eps,
            out _,
            out var variance);
        _lastEmaStdInv = Engine.TensorReciprocal(
            Engine.TensorSqrt(
                Engine.TensorAddScalar(variance, NumOps.FromDouble(eps))));
        return normed;
    }

    private Tensor<T> CreateOnesLike(Tensor<T> template)
    {
        var result = TensorAllocator.Rent<T>(template._shape);
        result.Fill(NumOps.One);
        return result;
    }

    #region Parameter Management

    /// <inheritdoc />
    public override void UpdateParameters(T learningRate)
    {
        if (_emaAlphaRealGradient == null)
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");

        T negLR = NumOps.Negate(learningRate);
        Engine.TensorAddInPlace(_emaAlphaReal, Engine.TensorMultiplyScalar(_emaAlphaRealGradient, negLR));
        Engine.TensorAddInPlace(_emaAlphaImag, Engine.TensorMultiplyScalar((_emaAlphaImagGradient ?? throw new InvalidOperationException("_emaAlphaImagGradient is null where it is required.")), negLR));
        Engine.TensorAddInPlace(_emaInputWeights, Engine.TensorMultiplyScalar((_emaInputWeightsGradient ?? throw new InvalidOperationException("_emaInputWeightsGradient is null where it is required.")), negLR));
        Engine.TensorAddInPlace(_emaInputBias, Engine.TensorMultiplyScalar((_emaInputBiasGradient ?? throw new InvalidOperationException("_emaInputBiasGradient is null where it is required.")), negLR));
        Engine.TensorAddInPlace(_emaOutputWeights, Engine.TensorMultiplyScalar((_emaOutputWeightsGradient ?? throw new InvalidOperationException("_emaOutputWeightsGradient is null where it is required.")), negLR));
        Engine.TensorAddInPlace(_emaOutputBias, Engine.TensorMultiplyScalar((_emaOutputBiasGradient ?? throw new InvalidOperationException("_emaOutputBiasGradient is null where it is required.")), negLR));
        Engine.TensorAddInPlace(_tsNormGamma, Engine.TensorMultiplyScalar((_tsNormGammaGradient ?? throw new InvalidOperationException("_tsNormGammaGradient is null where it is required.")), negLR));
        Engine.TensorAddInPlace(_tsNormBeta, Engine.TensorMultiplyScalar((_tsNormBetaGradient ?? throw new InvalidOperationException("_tsNormBetaGradient is null where it is required.")), negLR));
        Engine.TensorAddInPlace(_queryWeights, Engine.TensorMultiplyScalar((_queryWeightsGradient ?? throw new InvalidOperationException("_queryWeightsGradient is null where it is required.")), negLR));
        Engine.TensorAddInPlace(_keyWeights, Engine.TensorMultiplyScalar((_keyWeightsGradient ?? throw new InvalidOperationException("_keyWeightsGradient is null where it is required.")), negLR));
        Engine.TensorAddInPlace(_valueWeights, Engine.TensorMultiplyScalar((_valueWeightsGradient ?? throw new InvalidOperationException("_valueWeightsGradient is null where it is required.")), negLR));
        Engine.TensorAddInPlace(_gateWeights, Engine.TensorMultiplyScalar((_gateWeightsGradient ?? throw new InvalidOperationException("_gateWeightsGradient is null where it is required.")), negLR));
        Engine.TensorAddInPlace(_gateBias, Engine.TensorMultiplyScalar((_gateBiasGradient ?? throw new InvalidOperationException("_gateBiasGradient is null where it is required.")), negLR));
        Engine.TensorAddInPlace(_outputProjectionWeights, Engine.TensorMultiplyScalar((_outputProjectionWeightsGradient ?? throw new InvalidOperationException("_outputProjectionWeightsGradient is null where it is required.")), negLR));
        Engine.TensorAddInPlace(_outputProjectionBias, Engine.TensorMultiplyScalar((_outputProjectionBiasGradient ?? throw new InvalidOperationException("_outputProjectionBiasGradient is null where it is required.")), negLR));

        // Register trainable parameters for tape-based autodiff
        RegisterTrainableParameter(_emaInputWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_emaInputBias, PersistentTensorRole.Biases);
        RegisterTrainableParameter(_emaOutputWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_emaOutputBias, PersistentTensorRole.Biases);
        RegisterTrainableParameter(_queryWeights, PersistentTensorRole.Weights);
        // Register trainable parameters for tape-based autodiff. These belong here with the rest:
        // UpdateParameters above applies gradients to every one of them, so they are trained on the
        // legacy path, but leaving them unregistered made them invisible to the tape path -- which
        // trains exclusively from registered parameters -- and to GetParameters/SetParameters, and
        // therefore to serialization.
        RegisterTrainableParameter(_emaAlphaReal, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_emaAlphaImag, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_tsNormGamma, PersistentTensorRole.NormalizationParams);
        RegisterTrainableParameter(_tsNormBeta, PersistentTensorRole.NormalizationParams);
        RegisterTrainableParameter(_keyWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_valueWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_gateWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_gateBias, PersistentTensorRole.Biases);
        RegisterTrainableParameter(_outputProjectionWeights, PersistentTensorRole.Weights);
        RegisterTrainableParameter(_outputProjectionBias, PersistentTensorRole.Biases);

    }

    private Tensor<T>[] GetAllTensors() =>
    [
        _emaAlphaReal, _emaAlphaImag,
        _emaInputWeights, _emaInputBias,
        _emaOutputWeights, _emaOutputBias,
        _tsNormGamma, _tsNormBeta,
        _queryWeights, _keyWeights, _valueWeights,
        _gateWeights, _gateBias,
        _outputProjectionWeights, _outputProjectionBias
    ];

    /// <inheritdoc />
    public override void ClearGradients()
    {
        base.ClearGradients();
        _emaAlphaRealGradient = null; _emaAlphaImagGradient = null; _emaInputWeightsGradient = null; _emaInputBiasGradient = null; _emaOutputWeightsGradient = null; _emaOutputBiasGradient = null; _tsNormGammaGradient = null; _tsNormBetaGradient = null; _queryWeightsGradient = null; _keyWeightsGradient = null; _valueWeightsGradient = null; _gateWeightsGradient = null; _gateBiasGradient = null;
        _outputProjectionWeightsGradient = null; _outputProjectionBiasGradient = null;
    }

    /// <inheritdoc />
    public override void ResetState()
    {
        _lastEmaOutputPreNorm = null;
        _lastEmaStdInv = null;
        _originalInputShape = null;
        _emaAlphaRealGradient = null;
        _emaAlphaImagGradient = null;
        _emaInputWeightsGradient = null;
        _emaInputBiasGradient = null;
        _emaOutputWeightsGradient = null;
        _emaOutputBiasGradient = null;
        _tsNormGammaGradient = null;
        _tsNormBetaGradient = null;
        _queryWeightsGradient = null;
        _keyWeightsGradient = null;
        _valueWeightsGradient = null;
        _gateWeightsGradient = null;
        _gateBiasGradient = null;
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
        metadata["EmaDimension"] = _emaDimension.ToString();
        return metadata;
    }

    /// <summary>
    /// Gets the output projection weights for external inspection.
    /// </summary>
    public Tensor<T> GetOutputProjectionWeights() => _outputProjectionWeights;

    /// <summary>
    /// Gets the query weights for external inspection.
    /// </summary>
    public Tensor<T> GetQueryWeights() => _queryWeights;

    /// <summary>
    /// Gets the CEMA alpha coefficients (real part) for external inspection.
    /// </summary>
    public Tensor<T> GetEmaAlphaReal() => _emaAlphaReal;

    /// <summary>
    /// Gets the CEMA alpha coefficients (imaginary part) for external inspection.
    /// </summary>
    public Tensor<T> GetEmaAlphaImag() => _emaAlphaImag;
}
