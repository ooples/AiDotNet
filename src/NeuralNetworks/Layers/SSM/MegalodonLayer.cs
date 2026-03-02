using AiDotNet.Autodiff;
using AiDotNet.Helpers;

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
public class MegalodonLayer<T> : LayerBase<T>
{
    private readonly int _modelDimension;
    private readonly int _numHeads;
    private readonly int _headDimension;
    private readonly int _emaDimension;

    // CEMA parameters: complex-valued alpha = alpha_real + i * alpha_imag
    // Stored as real/imag pairs: [emaDimension] each
    private Tensor<T> _emaAlphaReal;
    private Tensor<T> _emaAlphaImag;

    // EMA input projection: [modelDim, emaDimension]
    private Tensor<T> _emaInputWeights;
    private Tensor<T> _emaInputBias;

    // EMA output projection: [emaDimension, modelDim]
    private Tensor<T> _emaOutputWeights;
    private Tensor<T> _emaOutputBias;

    // Timestep normalization: [emaDimension] each
    private Tensor<T> _tsNormGamma;
    private Tensor<T> _tsNormBeta;

    // Q, K, V projections: [modelDim, modelDim]
    private Tensor<T> _queryWeights;
    private Tensor<T> _keyWeights;
    private Tensor<T> _valueWeights;

    // Gating: [modelDim, modelDim]
    private Tensor<T> _gateWeights;
    private Tensor<T> _gateBias;

    // Output projection: [modelDim, modelDim]
    private Tensor<T> _outputProjectionWeights;
    private Tensor<T> _outputProjectionBias;

    // Cached forward pass values
    private Tensor<T>? _lastInput;
    private Tensor<T>? _lastOutput;
    private Tensor<T>? _lastEmaInput;
    private Tensor<T>? _lastEmaStatesReal;
    private Tensor<T>? _lastEmaStatesImag;
    private Tensor<T>? _lastEmaOutputNorm;
    private Tensor<T>? _lastEmaOutputPreNorm;
    private Tensor<T>? _lastEmaStdInv;
    private Tensor<T>? _lastQuery;
    private Tensor<T>? _lastKey;
    private Tensor<T>? _lastValue;
    private Tensor<T>? _lastGate;
    private Tensor<T>? _lastGateRaw;
    private Tensor<T>? _lastAttentionOutput;
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

    /// <inheritdoc />
    public override bool SupportsJitCompilation => false;

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
    /// Gets the total number of trainable parameters.
    /// </summary>
    public override int ParameterCount =>
        _emaAlphaReal.Length + _emaAlphaImag.Length +
        _emaInputWeights.Length + _emaInputBias.Length +
        _emaOutputWeights.Length + _emaOutputBias.Length +
        _tsNormGamma.Length + _tsNormBeta.Length +
        _queryWeights.Length + _keyWeights.Length + _valueWeights.Length +
        _gateWeights.Length + _gateBias.Length +
        _outputProjectionWeights.Length + _outputProjectionBias.Length;

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
    public MegalodonLayer(
        int sequenceLength,
        int modelDimension = 256,
        int numHeads = 8,
        int emaDimension = 16,
        IActivationFunction<T>? activationFunction = null)
        : base(
            [sequenceLength, modelDimension],
            [sequenceLength, modelDimension],
            activationFunction ?? new IdentityActivation<T>())
    {
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
        for (int i = 0; i < _emaDimension; i++)
            _tsNormGamma[i] = NumOps.One;
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
        int fanIn = tensor.Shape[0];
        int fanOut = tensor.Shape[1];
        T scale = NumOps.Sqrt(NumOps.FromDouble(2.0 / (fanIn + fanOut)));
        for (int i = 0; i < tensor.Length; i++)
            tensor[i] = NumOps.Multiply(NumOps.FromDouble(Random.NextDouble() - 0.5), scale);
    }

    /// <inheritdoc />
    public override Tensor<T> Forward(Tensor<T> input)
    {
        _originalInputShape = input.Shape;

        int rank = input.Shape.Length;
        int seqLen = rank >= 2 ? input.Shape[rank - 2] : 1;
        int modelDim = input.Shape[rank - 1];

        int batchSize = 1;
        for (int d = 0; d < rank - 2; d++)
            batchSize *= input.Shape[d];
        if (rank < 3) batchSize = 1;

        var input3D = rank == 2
            ? input.Reshape(1, seqLen, modelDim)
            : input.Reshape(batchSize, seqLen, modelDim);

        _lastInput = input3D;

        // Step 1: CEMA (Complex Exponential Moving Average)
        var cemaOutput = CEMAForward(input3D, batchSize, seqLen);

        // Step 2: Project CEMA output back to model dimension
        var cemaFlat = cemaOutput.Reshape(batchSize * seqLen, _emaDimension);
        var emaProjected = Engine.TensorMatMul(cemaFlat, _emaOutputWeights);
        emaProjected = Engine.TensorBroadcastAdd(emaProjected, _emaOutputBias.Reshape(1, _modelDimension));
        var emaOut3D = emaProjected.Reshape(batchSize, seqLen, _modelDimension);

        // Step 3: Q, K, V projections for gated attention
        var inputFlat = input3D.Reshape(batchSize * seqLen, _modelDimension);
        var q = Engine.TensorMatMul(inputFlat, _queryWeights).Reshape(batchSize, seqLen, _modelDimension);
        var k = Engine.TensorMatMul(inputFlat, _keyWeights).Reshape(batchSize, seqLen, _modelDimension);
        var v = Engine.TensorMatMul(inputFlat, _valueWeights).Reshape(batchSize, seqLen, _modelDimension);
        _lastQuery = q;
        _lastKey = k;
        _lastValue = v;

        // Step 4: Scaled dot-product attention per head (causal)
        var attentionOutput = MultiHeadAttentionForward(q, k, v, batchSize, seqLen);
        _lastAttentionOutput = attentionOutput;

        // Step 5: Gating between attention and CEMA outputs
        var gateRaw = Engine.TensorBroadcastAdd(
            Engine.TensorMatMul(inputFlat, _gateWeights),
            _gateBias.Reshape(1, _modelDimension)).Reshape(batchSize, seqLen, _modelDimension);
        var gate = Engine.Sigmoid(gateRaw);
        _lastGate = gate;
        _lastGateRaw = gateRaw;

        // y = gate * attention_out + (1 - gate) * cema_projected
        var ones = CreateOnesLike(gate);
        var oneMinusGate = Engine.TensorSubtract(ones, gate);
        var mixed = Engine.TensorAdd(
            Engine.TensorMultiply(gate, attentionOutput),
            Engine.TensorMultiply(oneMinusGate, emaOut3D));

        // Step 6: Output projection
        var mixedFlat = mixed.Reshape(batchSize * seqLen, _modelDimension);
        var outputFlat = Engine.TensorMatMul(mixedFlat, _outputProjectionWeights);
        outputFlat = Engine.TensorBroadcastAdd(outputFlat, _outputProjectionBias.Reshape(1, _modelDimension));
        var output3D = outputFlat.Reshape(batchSize, seqLen, _modelDimension);

        var result = ApplyActivation(output3D);
        _lastOutput = result;

        if (rank == 2)
            return result.Reshape(seqLen, _modelDimension);

        var outputShape = new int[rank];
        for (int i = 0; i < rank - 2; i++)
            outputShape[i] = input.Shape[i];
        outputShape[rank - 2] = seqLen;
        outputShape[rank - 1] = _modelDimension;
        return result.Reshape(outputShape);
    }

    /// <summary>
    /// Complex Exponential Moving Average forward pass with timestep normalization.
    /// </summary>
    /// <remarks>
    /// Computes h_t = alpha * h_{t-1} + (1 - alpha) * x_t where alpha is complex-valued.
    /// The real part of the state is extracted and normalized per-timestep.
    /// </remarks>
    private Tensor<T> CEMAForward(Tensor<T> input, int batchSize, int seqLen)
    {
        // Project input to EMA dimension
        var inputFlat = input.Reshape(batchSize * seqLen, _modelDimension);
        var emaInput = Engine.TensorMatMul(inputFlat, _emaInputWeights);
        emaInput = Engine.TensorBroadcastAdd(emaInput, _emaInputBias.Reshape(1, _emaDimension));
        var emaInput3D = emaInput.Reshape(batchSize, seqLen, _emaDimension);
        _lastEmaInput = emaInput3D;

        // CEMA recurrence with complex coefficients
        // State is complex: h_real + i * h_imag
        var statesReal = new Tensor<T>(new[] { batchSize, seqLen + 1, _emaDimension });
        var statesImag = new Tensor<T>(new[] { batchSize, seqLen + 1, _emaDimension });
        var outputPreNorm = new Tensor<T>(new[] { batchSize, seqLen, _emaDimension });

        for (int t = 0; t < seqLen; t++)
        {
            for (int bi = 0; bi < batchSize; bi++)
            {
                for (int d = 0; d < _emaDimension; d++)
                {
                    T alphaR = _emaAlphaReal[d];
                    T alphaI = _emaAlphaImag[d];
                    T xVal = emaInput3D[new[] { bi, t, d }];

                    // One minus alpha (complex): (1 - alphaR) - i * alphaI
                    T oneMinusAlphaR = NumOps.Subtract(NumOps.One, alphaR);
                    T negAlphaI = NumOps.Negate(alphaI);

                    // Previous state
                    T hPrevR = statesReal[new[] { bi, t, d }];
                    T hPrevI = statesImag[new[] { bi, t, d }];

                    // Complex multiply: alpha * h_prev
                    // (aR + i*aI) * (hR + i*hI) = (aR*hR - aI*hI) + i*(aR*hI + aI*hR)
                    T ahR = NumOps.Subtract(
                        NumOps.Multiply(alphaR, hPrevR),
                        NumOps.Multiply(alphaI, hPrevI));
                    T ahI = NumOps.Add(
                        NumOps.Multiply(alphaR, hPrevI),
                        NumOps.Multiply(alphaI, hPrevR));

                    // Complex multiply: (1 - alpha) * x (x is real, so imag part of x is 0)
                    // (oneMinusAlphaR + i*negAlphaI) * x = oneMinusAlphaR*x + i*negAlphaI*x
                    T bxR = NumOps.Multiply(oneMinusAlphaR, xVal);
                    T bxI = NumOps.Multiply(negAlphaI, xVal);

                    // h_t = alpha * h_{t-1} + (1 - alpha) * x_t
                    T hNewR = NumOps.Add(ahR, bxR);
                    T hNewI = NumOps.Add(ahI, bxI);

                    statesReal[new[] { bi, t + 1, d }] = hNewR;
                    statesImag[new[] { bi, t + 1, d }] = hNewI;

                    // Extract real part as output (standard for complex EMA in Megalodon)
                    outputPreNorm[new[] { bi, t, d }] = hNewR;
                }
            }
        }

        _lastEmaStatesReal = statesReal;
        _lastEmaStatesImag = statesImag;
        _lastEmaOutputPreNorm = outputPreNorm;

        // Timestep normalization: normalize each (batch, time) vector across EMA dimensions
        var outputNorm = new Tensor<T>(new[] { batchSize, seqLen, _emaDimension });
        var stdInv = new Tensor<T>(new[] { batchSize, seqLen });
        T eps = NumOps.FromDouble(1e-5);

        for (int bi = 0; bi < batchSize; bi++)
        {
            for (int t = 0; t < seqLen; t++)
            {
                // Compute mean
                T mean = NumOps.Zero;
                for (int d = 0; d < _emaDimension; d++)
                    mean = NumOps.Add(mean, outputPreNorm[new[] { bi, t, d }]);
                mean = NumOps.Divide(mean, NumOps.FromDouble(_emaDimension));

                // Compute variance
                T variance = NumOps.Zero;
                for (int d = 0; d < _emaDimension; d++)
                {
                    T diff = NumOps.Subtract(outputPreNorm[new[] { bi, t, d }], mean);
                    variance = NumOps.Add(variance, NumOps.Multiply(diff, diff));
                }
                variance = NumOps.Divide(variance, NumOps.FromDouble(_emaDimension));

                // Normalize: (x - mean) / sqrt(var + eps) * gamma + beta
                T invStd = NumOps.Divide(NumOps.One, NumOps.Sqrt(NumOps.Add(variance, eps)));
                stdInv[new[] { bi, t }] = invStd;

                for (int d = 0; d < _emaDimension; d++)
                {
                    T normalized = NumOps.Multiply(
                        NumOps.Subtract(outputPreNorm[new[] { bi, t, d }], mean),
                        invStd);
                    outputNorm[new[] { bi, t, d }] = NumOps.Add(
                        NumOps.Multiply(_tsNormGamma[d], normalized),
                        _tsNormBeta[d]);
                }
            }
        }

        _lastEmaOutputNorm = outputNorm;
        _lastEmaStdInv = stdInv;
        return outputNorm;
    }

    /// <summary>
    /// Multi-head causal attention forward pass with scaled dot-product.
    /// </summary>
    private Tensor<T> MultiHeadAttentionForward(
        Tensor<T> q, Tensor<T> k, Tensor<T> v,
        int batchSize, int seqLen)
    {
        var output = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });
        T scale = NumOps.FromDouble(1.0 / Math.Sqrt(_headDimension));

        for (int bi = 0; bi < batchSize; bi++)
        {
            for (int hi = 0; hi < _numHeads; hi++)
            {
                int dimStart = hi * _headDimension;

                for (int t = 0; t < seqLen; t++)
                {
                    // Compute attention scores for query at position t
                    // Only attend to positions <= t (causal)
                    var scores = new T[t + 1];
                    T maxScore = NumOps.FromDouble(double.NegativeInfinity);

                    for (int s = 0; s <= t; s++)
                    {
                        T dot = NumOps.Zero;
                        for (int d = 0; d < _headDimension; d++)
                        {
                            int flatD = dimStart + d;
                            dot = NumOps.Add(dot,
                                NumOps.Multiply(q[new[] { bi, t, flatD }], k[new[] { bi, s, flatD }]));
                        }
                        scores[s] = NumOps.Multiply(dot, scale);
                        if (NumOps.GreaterThan(scores[s], maxScore))
                            maxScore = scores[s];
                    }

                    // Softmax over causal window
                    T sumExp = NumOps.Zero;
                    var expScores = new T[t + 1];
                    for (int s = 0; s <= t; s++)
                    {
                        expScores[s] = NumOps.Exp(NumOps.Subtract(scores[s], maxScore));
                        sumExp = NumOps.Add(sumExp, expScores[s]);
                    }

                    T invSum = NumOps.Divide(NumOps.One, NumOps.Add(sumExp, NumOps.FromDouble(1e-10)));

                    // Weighted sum of values
                    for (int d = 0; d < _headDimension; d++)
                    {
                        int flatD = dimStart + d;
                        T val = NumOps.Zero;
                        for (int s = 0; s <= t; s++)
                        {
                            T attnWeight = NumOps.Multiply(expScores[s], invSum);
                            val = NumOps.Add(val,
                                NumOps.Multiply(attnWeight, v[new[] { bi, s, flatD }]));
                        }
                        output[new[] { bi, t, flatD }] = val;
                    }
                }
            }
        }

        return output;
    }

    /// <inheritdoc />
    public override Tensor<T> Backward(Tensor<T> outputGradient)
    {
        if (_lastInput == null || _lastOutput == null || _lastQuery == null ||
            _lastKey == null || _lastValue == null || _lastGate == null ||
            _lastGateRaw == null || _lastAttentionOutput == null ||
            _lastEmaInput == null || _lastEmaOutputNorm == null ||
            _lastEmaOutputPreNorm == null || _lastEmaStatesReal == null ||
            _lastEmaStatesImag == null || _lastEmaStdInv == null)
            throw new InvalidOperationException("Forward pass must be called before backward pass.");

        int batchSize = _lastInput.Shape[0];
        int seqLen = _lastInput.Shape[1];

        var grad3D = outputGradient.Rank == 2
            ? outputGradient.Reshape(1, outputGradient.Shape[0], _modelDimension)
            : outputGradient.Reshape(batchSize, seqLen, _modelDimension);

        var activationGrad = ApplyActivationDerivative(_lastOutput, grad3D);

        // Initialize gradients
        _emaAlphaRealGradient = new Tensor<T>([_emaDimension]);
        _emaAlphaImagGradient = new Tensor<T>([_emaDimension]);
        _emaInputWeightsGradient = new Tensor<T>([_modelDimension, _emaDimension]);
        _emaInputBiasGradient = new Tensor<T>([_emaDimension]);
        _emaOutputWeightsGradient = new Tensor<T>([_emaDimension, _modelDimension]);
        _emaOutputBiasGradient = new Tensor<T>([_modelDimension]);
        _tsNormGammaGradient = new Tensor<T>([_emaDimension]);
        _tsNormBetaGradient = new Tensor<T>([_emaDimension]);
        _queryWeightsGradient = new Tensor<T>([_modelDimension, _modelDimension]);
        _keyWeightsGradient = new Tensor<T>([_modelDimension, _modelDimension]);
        _valueWeightsGradient = new Tensor<T>([_modelDimension, _modelDimension]);
        _gateWeightsGradient = new Tensor<T>([_modelDimension, _modelDimension]);
        _gateBiasGradient = new Tensor<T>([_modelDimension]);
        _outputProjectionWeightsGradient = new Tensor<T>([_modelDimension, _modelDimension]);
        _outputProjectionBiasGradient = new Tensor<T>([_modelDimension]);

        // Step 6 backward: output projection
        var gradFlat = activationGrad.Reshape(batchSize * seqLen, _modelDimension);
        _outputProjectionBiasGradient = Engine.ReduceSum(activationGrad, new int[] { 0, 1 });

        // Reconstruct mixed for output projection gradient
        var ones = CreateOnesLike(_lastGate);
        var oneMinusGate = Engine.TensorSubtract(ones, _lastGate);

        // Need CEMA projected output for backward
        var cemaFlat = _lastEmaOutputNorm.Reshape(batchSize * seqLen, _emaDimension);
        var emaProjected = Engine.TensorMatMul(cemaFlat, _emaOutputWeights);
        emaProjected = Engine.TensorBroadcastAdd(emaProjected, _emaOutputBias.Reshape(1, _modelDimension));
        var emaOut3D = emaProjected.Reshape(batchSize, seqLen, _modelDimension);

        var mixed = Engine.TensorAdd(
            Engine.TensorMultiply(_lastGate, _lastAttentionOutput),
            Engine.TensorMultiply(oneMinusGate, emaOut3D));
        var mixedFlat = mixed.Reshape(batchSize * seqLen, _modelDimension);

        _outputProjectionWeightsGradient = Engine.TensorMatMul(mixedFlat.Transpose([1, 0]), gradFlat);

        var dMixed = Engine.TensorMatMul(gradFlat, _outputProjectionWeights.Transpose([1, 0]))
            .Reshape(batchSize, seqLen, _modelDimension);

        // Step 5 backward: gating
        // y = gate * attn + (1-gate) * ema_proj
        // dAttn = dMixed * gate
        // dEmaProjOut = dMixed * (1 - gate)
        // dGate = dMixed * (attn - ema_proj)
        var dAttn = Engine.TensorMultiply(dMixed, _lastGate);
        var dEmaProjOut = Engine.TensorMultiply(dMixed, oneMinusGate);
        var attnMinusEma = Engine.TensorSubtract(_lastAttentionOutput, emaOut3D);
        var dGateSigmoid = Engine.TensorMultiply(dMixed, attnMinusEma);

        // Sigmoid derivative: gate * (1 - gate)
        var sigmoidDeriv = Engine.TensorMultiply(_lastGate, oneMinusGate);
        var dGateRaw = Engine.TensorMultiply(dGateSigmoid, sigmoidDeriv);

        var inputFlat = _lastInput.Reshape(batchSize * seqLen, _modelDimension);
        var dGateRawFlat = dGateRaw.Reshape(batchSize * seqLen, _modelDimension);
        _gateWeightsGradient = Engine.TensorMatMul(inputFlat.Transpose([1, 0]), dGateRawFlat);
        _gateBiasGradient = Engine.ReduceSum(dGateRaw, new int[] { 0, 1 });

        var dInputFromGate = Engine.TensorMatMul(dGateRawFlat, _gateWeights.Transpose([1, 0]));

        // Step 4 backward: attention Q, K, V gradients (simplified)
        var dAttnFlat = dAttn.Reshape(batchSize * seqLen, _modelDimension);
        var dQ = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });
        var dK = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });
        var dV = new Tensor<T>(new[] { batchSize, seqLen, _modelDimension });

        // Backprop through attention
        AttentionBackward(dAttn, _lastQuery, _lastKey, _lastValue, dQ, dK, dV, batchSize, seqLen);

        var dQFlat = dQ.Reshape(batchSize * seqLen, _modelDimension);
        var dKFlat = dK.Reshape(batchSize * seqLen, _modelDimension);
        var dVFlat = dV.Reshape(batchSize * seqLen, _modelDimension);

        _queryWeightsGradient = Engine.TensorMatMul(inputFlat.Transpose([1, 0]), dQFlat);
        _keyWeightsGradient = Engine.TensorMatMul(inputFlat.Transpose([1, 0]), dKFlat);
        _valueWeightsGradient = Engine.TensorMatMul(inputFlat.Transpose([1, 0]), dVFlat);

        var dInputFromQ = Engine.TensorMatMul(dQFlat, _queryWeights.Transpose([1, 0]));
        var dInputFromK = Engine.TensorMatMul(dKFlat, _keyWeights.Transpose([1, 0]));
        var dInputFromV = Engine.TensorMatMul(dVFlat, _valueWeights.Transpose([1, 0]));

        // Step 3 backward: EMA output projection
        var dEmaProjFlat = dEmaProjOut.Reshape(batchSize * seqLen, _modelDimension);
        _emaOutputBiasGradient = Engine.ReduceSum(dEmaProjOut, new int[] { 0, 1 });
        _emaOutputWeightsGradient = Engine.TensorMatMul(cemaFlat.Transpose([1, 0]), dEmaProjFlat);

        var dCemaNorm = Engine.TensorMatMul(dEmaProjFlat, _emaOutputWeights.Transpose([1, 0]))
            .Reshape(batchSize, seqLen, _emaDimension);

        // Step 2 backward: timestep normalization
        var dPreNorm = TimestepNormBackward(dCemaNorm, batchSize, seqLen);

        // Step 1 backward: CEMA recurrence
        var dEmaInput = CEMABackward(dPreNorm, batchSize, seqLen);

        // EMA input projection backward
        var dEmaInputFlat = dEmaInput.Reshape(batchSize * seqLen, _emaDimension);
        _emaInputBiasGradient = Engine.ReduceSum(dEmaInput, new int[] { 0, 1 });
        _emaInputWeightsGradient = Engine.TensorMatMul(inputFlat.Transpose([1, 0]), dEmaInputFlat);

        var dInputFromEma = Engine.TensorMatMul(dEmaInputFlat, _emaInputWeights.Transpose([1, 0]));

        // Combine all input gradients
        var dInputTotal = Engine.TensorAdd(dInputFromGate, dInputFromQ);
        dInputTotal = Engine.TensorAdd(dInputTotal, dInputFromK);
        dInputTotal = Engine.TensorAdd(dInputTotal, dInputFromV);
        dInputTotal = Engine.TensorAdd(dInputTotal, dInputFromEma);

        var dInput3D = dInputTotal.Reshape(batchSize, seqLen, _modelDimension);

        if (_originalInputShape != null && _originalInputShape.Length == 2)
            return dInput3D.Reshape(seqLen, _modelDimension);

        if (_originalInputShape != null)
            return dInput3D.Reshape(_originalInputShape);

        return dInput3D;
    }

    /// <summary>
    /// Backward pass through the causal multi-head attention.
    /// </summary>
    private void AttentionBackward(
        Tensor<T> dOutput, Tensor<T> q, Tensor<T> k, Tensor<T> v,
        Tensor<T> dQ, Tensor<T> dK, Tensor<T> dV,
        int batchSize, int seqLen)
    {
        T scale = NumOps.FromDouble(1.0 / Math.Sqrt(_headDimension));

        for (int bi = 0; bi < batchSize; bi++)
        {
            for (int hi = 0; hi < _numHeads; hi++)
            {
                int dimStart = hi * _headDimension;

                for (int t = 0; t < seqLen; t++)
                {
                    // Recompute attention weights for this position
                    var scores = new T[t + 1];
                    T maxScore = NumOps.FromDouble(double.NegativeInfinity);

                    for (int s = 0; s <= t; s++)
                    {
                        T dot = NumOps.Zero;
                        for (int d = 0; d < _headDimension; d++)
                        {
                            int flatD = dimStart + d;
                            dot = NumOps.Add(dot,
                                NumOps.Multiply(q[new[] { bi, t, flatD }], k[new[] { bi, s, flatD }]));
                        }
                        scores[s] = NumOps.Multiply(dot, scale);
                        if (NumOps.GreaterThan(scores[s], maxScore))
                            maxScore = scores[s];
                    }

                    T sumExp = NumOps.Zero;
                    var weights = new T[t + 1];
                    for (int s = 0; s <= t; s++)
                    {
                        weights[s] = NumOps.Exp(NumOps.Subtract(scores[s], maxScore));
                        sumExp = NumOps.Add(sumExp, weights[s]);
                    }
                    T invSum = NumOps.Divide(NumOps.One, NumOps.Add(sumExp, NumOps.FromDouble(1e-10)));
                    for (int s = 0; s <= t; s++)
                        weights[s] = NumOps.Multiply(weights[s], invSum);

                    // dV: dV_s += weight_s * dOutput_t
                    // dWeights_s = dOutput_t . v_s
                    var dWeights = new T[t + 1];
                    for (int s = 0; s <= t; s++)
                    {
                        dWeights[s] = NumOps.Zero;
                        for (int d = 0; d < _headDimension; d++)
                        {
                            int flatD = dimStart + d;
                            T dOut = dOutput[new[] { bi, t, flatD }];

                            dV[new[] { bi, s, flatD }] = NumOps.Add(
                                dV[new[] { bi, s, flatD }],
                                NumOps.Multiply(weights[s], dOut));

                            dWeights[s] = NumOps.Add(dWeights[s],
                                NumOps.Multiply(dOut, v[new[] { bi, s, flatD }]));
                        }
                    }

                    // Softmax backward: dScore_s = weight_s * (dWeight_s - sum_j(weight_j * dWeight_j))
                    T weightedDWeightSum = NumOps.Zero;
                    for (int s = 0; s <= t; s++)
                        weightedDWeightSum = NumOps.Add(weightedDWeightSum,
                            NumOps.Multiply(weights[s], dWeights[s]));

                    for (int s = 0; s <= t; s++)
                    {
                        T dScore = NumOps.Multiply(weights[s],
                            NumOps.Subtract(dWeights[s], weightedDWeightSum));
                        dScore = NumOps.Multiply(dScore, scale);

                        for (int d = 0; d < _headDimension; d++)
                        {
                            int flatD = dimStart + d;
                            dQ[new[] { bi, t, flatD }] = NumOps.Add(
                                dQ[new[] { bi, t, flatD }],
                                NumOps.Multiply(dScore, k[new[] { bi, s, flatD }]));

                            dK[new[] { bi, s, flatD }] = NumOps.Add(
                                dK[new[] { bi, s, flatD }],
                                NumOps.Multiply(dScore, q[new[] { bi, t, flatD }]));
                        }
                    }
                }
            }
        }
    }

    /// <summary>
    /// Backward pass through timestep normalization.
    /// </summary>
    private Tensor<T> TimestepNormBackward(Tensor<T> dNormOutput, int batchSize, int seqLen)
    {
        var dPreNorm = new Tensor<T>(new[] { batchSize, seqLen, _emaDimension });
        T eps = NumOps.FromDouble(1e-5);
        var emaPreNorm = _lastEmaOutputPreNorm
            ?? throw new InvalidOperationException("Forward pass must be called before backward pass.");

        for (int bi = 0; bi < batchSize; bi++)
        {
            for (int t = 0; t < seqLen; t++)
            {
                // Recompute mean for this timestep
                T mean = NumOps.Zero;
                for (int d = 0; d < _emaDimension; d++)
                    mean = NumOps.Add(mean, emaPreNorm[new[] { bi, t, d }]);
                mean = NumOps.Divide(mean, NumOps.FromDouble(_emaDimension));

                T invStd = _lastEmaStdInv![new[] { bi, t }];
                T invN = NumOps.FromDouble(1.0 / _emaDimension);

                // Compute normalized values and gradient sums
                T sumDGamma = NumOps.Zero;
                T sumDGammaXhat = NumOps.Zero;

                for (int d = 0; d < _emaDimension; d++)
                {
                    T xHat = NumOps.Multiply(
                        NumOps.Subtract(emaPreNorm[new[] { bi, t, d }], mean),
                        invStd);
                    T dOut = dNormOutput[new[] { bi, t, d }];
                    T gammaDOut = NumOps.Multiply(_tsNormGamma[d], dOut);

                    sumDGamma = NumOps.Add(sumDGamma, gammaDOut);
                    sumDGammaXhat = NumOps.Add(sumDGammaXhat, NumOps.Multiply(gammaDOut, xHat));

                    // Accumulate norm parameter gradients
                    _tsNormGammaGradient![d] = NumOps.Add(_tsNormGammaGradient[d],
                        NumOps.Multiply(dOut, xHat));
                    _tsNormBetaGradient![d] = NumOps.Add(_tsNormBetaGradient[d], dOut);
                }

                // Input gradient: invStd * (gamma * dOut - invN * sumDGamma - invN * xHat * sumDGammaXhat)
                for (int d = 0; d < _emaDimension; d++)
                {
                    T xHat = NumOps.Multiply(
                        NumOps.Subtract(emaPreNorm[new[] { bi, t, d }], mean),
                        invStd);
                    T gammaDOut = NumOps.Multiply(_tsNormGamma[d], dNormOutput[new[] { bi, t, d }]);

                    T grad = NumOps.Subtract(gammaDOut, NumOps.Multiply(invN, sumDGamma));
                    grad = NumOps.Subtract(grad, NumOps.Multiply(NumOps.Multiply(invN, xHat), sumDGammaXhat));
                    dPreNorm[new[] { bi, t, d }] = NumOps.Multiply(invStd, grad);
                }
            }
        }

        return dPreNorm;
    }

    /// <summary>
    /// Backward pass through the CEMA recurrence.
    /// </summary>
    private Tensor<T> CEMABackward(Tensor<T> dPreNorm, int batchSize, int seqLen)
    {
        var dEmaInput = new Tensor<T>(new[] { batchSize, seqLen, _emaDimension });

        // Gradient flows backward through real part of the state
        var dStateR = new T[_emaDimension];
        var dStateI = new T[_emaDimension];

        for (int bi = 0; bi < batchSize; bi++)
        {
            // Reset state gradients for each batch
            for (int d = 0; d < _emaDimension; d++)
            {
                dStateR[d] = NumOps.Zero;
                dStateI[d] = NumOps.Zero;
            }

            for (int t = seqLen - 1; t >= 0; t--)
            {
                for (int d = 0; d < _emaDimension; d++)
                {
                    T alphaR = _emaAlphaReal[d];
                    T alphaI = _emaAlphaImag[d];

                    // Output was the real part of state: dStateR += dPreNorm
                    dStateR[d] = NumOps.Add(dStateR[d], dPreNorm[new[] { bi, t, d }]);

                    T hPrevR = _lastEmaStatesReal![new[] { bi, t, d }];
                    T hPrevI = _lastEmaStatesImag![new[] { bi, t, d }];

                    // h_t = alpha * h_{t-1} + (1-alpha) * x
                    // dAlphaR += dStateR * hPrevR + dStateI * hPrevI  (partial of complex mult)
                    // Actually: d/dAlphaR of (alphaR*hPrevR - alphaI*hPrevI) = hPrevR for real part
                    // d/dAlphaR of (alphaR*hPrevI + alphaI*hPrevR) = hPrevI for imag part
                    _emaAlphaRealGradient![d] = NumOps.Add(_emaAlphaRealGradient[d],
                        NumOps.Add(
                            NumOps.Multiply(dStateR[d], hPrevR),
                            NumOps.Multiply(dStateI[d], hPrevI)));

                    // d/dAlphaI of (alphaR*hPrevR - alphaI*hPrevI) = -hPrevI for real part
                    // d/dAlphaI of (alphaR*hPrevI + alphaI*hPrevR) = hPrevR for imag part
                    _emaAlphaImagGradient![d] = NumOps.Add(_emaAlphaImagGradient[d],
                        NumOps.Add(
                            NumOps.Multiply(dStateR[d], NumOps.Negate(hPrevI)),
                            NumOps.Multiply(dStateI[d], hPrevR)));

                    // dX from (1-alpha)*x: real part contributes (1-alphaR)*dStateR + alphaI*dStateI
                    T oneMinusAlphaR = NumOps.Subtract(NumOps.One, alphaR);
                    T dX = NumOps.Add(
                        NumOps.Multiply(oneMinusAlphaR, dStateR[d]),
                        NumOps.Multiply(NumOps.Negate(alphaI), dStateI[d]));
                    dEmaInput[new[] { bi, t, d }] = dX;

                    // Propagate gradient to previous state through complex multiplication
                    // dh_prev_R = alphaR * dStateR + alphaI * dStateI
                    // dh_prev_I = -alphaI * dStateR + alphaR * dStateI
                    T newDStateR = NumOps.Add(
                        NumOps.Multiply(alphaR, dStateR[d]),
                        NumOps.Multiply(alphaI, dStateI[d]));
                    T newDStateI = NumOps.Add(
                        NumOps.Multiply(NumOps.Negate(alphaI), dStateR[d]),
                        NumOps.Multiply(alphaR, dStateI[d]));
                    dStateR[d] = newDStateR;
                    dStateI[d] = newDStateI;
                }
            }
        }

        return dEmaInput;
    }

    private Tensor<T> CreateOnesLike(Tensor<T> template)
    {
        var result = new Tensor<T>(template.Shape);
        for (int i = 0; i < result.Length; i++) result[i] = NumOps.One;
        return result;
    }

    #region Parameter Management

    /// <inheritdoc />
    public override void UpdateParameters(T learningRate)
    {
        if (_emaAlphaRealGradient == null)
            throw new InvalidOperationException("Backward pass must be called before updating parameters.");

        T negLR = NumOps.Negate(learningRate);
        _emaAlphaReal = Engine.TensorAdd(_emaAlphaReal, Engine.TensorMultiplyScalar(_emaAlphaRealGradient, negLR));
        _emaAlphaImag = Engine.TensorAdd(_emaAlphaImag, Engine.TensorMultiplyScalar(_emaAlphaImagGradient!, negLR));
        _emaInputWeights = Engine.TensorAdd(_emaInputWeights, Engine.TensorMultiplyScalar(_emaInputWeightsGradient!, negLR));
        _emaInputBias = Engine.TensorAdd(_emaInputBias, Engine.TensorMultiplyScalar(_emaInputBiasGradient!, negLR));
        _emaOutputWeights = Engine.TensorAdd(_emaOutputWeights, Engine.TensorMultiplyScalar(_emaOutputWeightsGradient!, negLR));
        _emaOutputBias = Engine.TensorAdd(_emaOutputBias, Engine.TensorMultiplyScalar(_emaOutputBiasGradient!, negLR));
        _tsNormGamma = Engine.TensorAdd(_tsNormGamma, Engine.TensorMultiplyScalar(_tsNormGammaGradient!, negLR));
        _tsNormBeta = Engine.TensorAdd(_tsNormBeta, Engine.TensorMultiplyScalar(_tsNormBetaGradient!, negLR));
        _queryWeights = Engine.TensorAdd(_queryWeights, Engine.TensorMultiplyScalar(_queryWeightsGradient!, negLR));
        _keyWeights = Engine.TensorAdd(_keyWeights, Engine.TensorMultiplyScalar(_keyWeightsGradient!, negLR));
        _valueWeights = Engine.TensorAdd(_valueWeights, Engine.TensorMultiplyScalar(_valueWeightsGradient!, negLR));
        _gateWeights = Engine.TensorAdd(_gateWeights, Engine.TensorMultiplyScalar(_gateWeightsGradient!, negLR));
        _gateBias = Engine.TensorAdd(_gateBias, Engine.TensorMultiplyScalar(_gateBiasGradient!, negLR));
        _outputProjectionWeights = Engine.TensorAdd(_outputProjectionWeights, Engine.TensorMultiplyScalar(_outputProjectionWeightsGradient!, negLR));
        _outputProjectionBias = Engine.TensorAdd(_outputProjectionBias, Engine.TensorMultiplyScalar(_outputProjectionBiasGradient!, negLR));
    }

    /// <inheritdoc />
    public override Vector<T> GetParameters()
    {
        var parameters = new Vector<T>(ParameterCount);
        int index = 0;
        foreach (var tensor in GetAllTensors())
            for (int i = 0; i < tensor.Length; i++)
                parameters[index++] = tensor[i];
        return parameters;
    }

    /// <inheritdoc />
    public override void SetParameters(Vector<T> parameters)
    {
        if (parameters.Length != ParameterCount)
            throw new ArgumentException($"Expected {ParameterCount} parameters, got {parameters.Length}");
        int index = 0;
        foreach (var tensor in GetAllTensors())
            for (int i = 0; i < tensor.Length; i++)
                tensor[i] = parameters[index++];
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
    public override void ResetState()
    {
        _lastInput = null;
        _lastOutput = null;
        _lastEmaInput = null;
        _lastEmaStatesReal = null;
        _lastEmaStatesImag = null;
        _lastEmaOutputNorm = null;
        _lastEmaOutputPreNorm = null;
        _lastEmaStdInv = null;
        _lastQuery = null;
        _lastKey = null;
        _lastValue = null;
        _lastGate = null;
        _lastGateRaw = null;
        _lastAttentionOutput = null;
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

    /// <inheritdoc />
    public override ComputationNode<T> ExportComputationGraph(List<ComputationNode<T>> inputNodes)
    {
        if (inputNodes == null)
            throw new ArgumentNullException(nameof(inputNodes));

        var xPlaceholder = new Tensor<T>(new int[] { 1, _modelDimension });
        var xNode = TensorOperations<T>.Variable(xPlaceholder, "x_t");
        var outWeightsNode = TensorOperations<T>.Variable(_outputProjectionWeights, "W_out");
        var outBiasNode = TensorOperations<T>.Variable(_outputProjectionBias, "b_out");

        inputNodes.Add(xNode);
        inputNodes.Add(outWeightsNode);
        inputNodes.Add(outBiasNode);

        var outT = TensorOperations<T>.Transpose(outWeightsNode);
        var finalOutput = TensorOperations<T>.MatrixMultiply(xNode, outT);
        var outputWithBias = TensorOperations<T>.Add(finalOutput, outBiasNode);

        return outputWithBias;
    }

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
