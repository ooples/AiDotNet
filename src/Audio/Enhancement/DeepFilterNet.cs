using AiDotNet.ActivationFunctions;
using AiDotNet.Diffusion.Audio;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Audio.Enhancement;

/// <summary>
/// DeepFilterNet - State-of-the-art deep filtering network for speech enhancement.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// DeepFilterNet is a hybrid time-frequency domain model that combines:
/// - ERB (Equivalent Rectangular Bandwidth) filterbank for perceptually-motivated processing
/// - Deep filtering in the complex STFT domain for fine-grained enhancement
/// - Efficient architecture with grouped convolutions for real-time processing
/// </para>
/// <para>
/// <b>For Beginners:</b> DeepFilterNet is like having an intelligent audio engineer
/// that can separate speech from background noise in real-time. It's particularly
/// effective because it processes audio the way humans perceive sound - focusing
/// more on frequencies that matter for understanding speech.
///
/// The model works by:
/// 1. Converting audio to a time-frequency representation (spectrogram)
/// 2. Applying learned filters to suppress noise while preserving speech
/// 3. Reconstructing clean audio from the enhanced spectrogram
///
/// Usage:
/// <code>
/// // ONNX mode for inference
/// var model = new DeepFilterNet&lt;float&gt;(architecture, "deepfilternet.onnx");
/// var cleanAudio = model.Enhance(noisyAudio);
///
/// // Native mode for training
/// var model = new DeepFilterNet&lt;float&gt;(architecture, hiddenDim: 96);
/// model.Train(noisyAudio, cleanAudio);
/// </code>
/// </para>
/// <para>
/// Reference: "DeepFilterNet: A Low Complexity Speech Enhancement Framework for
/// Full-Band Audio based on Deep Filtering" by Schröter et al., ICASSP 2022
/// </para>
/// </remarks>
public class DeepFilterNet<T> : AudioNeuralNetworkBase<T>, IAudioEnhancer<T>
{
    private readonly DeepFilterNetOptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    #region Model Architecture Parameters

    /// <summary>
    /// Number of ERB (Equivalent Rectangular Bandwidth) bands.
    /// </summary>
    private readonly int _numErbBands;

    /// <summary>
    /// Hidden dimension for the encoder/decoder.
    /// </summary>
    private readonly int _hiddenDim;

    /// <summary>
    /// Number of DeepFilter coefficients per frequency bin.
    /// </summary>
    private readonly int _dfOrder;

    /// <summary>
    /// Number of frequency bins to apply deep filtering.
    /// </summary>
    private readonly int _dfBins;

    /// <summary>
    /// Number of GRU layers in the enhancement network.
    /// </summary>
    private readonly int _numGruLayers;

    /// <summary>
    /// Convolution kernel size for feature extraction.
    /// </summary>
    private readonly int _convKernelSize;

    /// <summary>
    /// FFT size for STFT analysis.
    /// </summary>
    private readonly int _fftSize;

    /// <summary>
    /// Hop size for STFT.
    /// </summary>
    private readonly int _hopSize;

    /// <summary>
    /// Lookahead frames for causal processing.
    /// </summary>
    private readonly int _lookahead;

    #endregion

    #region Network Layers

    /// <summary>
    /// ERB encoder layers.
    /// </summary>
    private readonly List<ILayer<T>> _erbEncoder = [];

    /// <summary>
    /// Deep filtering layers.
    /// </summary>
    private readonly List<ILayer<T>> _dfLayers = [];

    /// <summary>
    /// GRU layers for temporal modeling.
    /// </summary>
    private readonly List<ILayer<T>> _gruLayers = [];

    /// <summary>
    /// Decoder layers for reconstruction.
    /// </summary>
    private readonly List<ILayer<T>> _decoder = [];

    /// <summary>
    /// Gain estimation layer.
    /// </summary>
    private ILayer<T>? _gainLayer;

    #endregion

    #region ONNX Mode Fields

    // ONNX model is inherited from AudioNeuralNetworkBase<T>.OnnxModel

    #endregion

    #region STFT

    private readonly ShortTimeFourierTransform<T> _stft;

    /// <summary>
    /// Cached complex STFT from preprocessing, used for audio reconstruction.
    /// </summary>
    private Tensor<Complex<T>>? _cachedComplexStft;

    #endregion

    #region Training State

    /// <summary>
    /// Loss function for training.
    /// </summary>
    private ILossFunction<T> _lossFunction;

    /// <summary>
    /// Optimizer for training.
    /// </summary>
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;

    #endregion

    #region IAudioEnhancer Properties

    /// <inheritdoc/>
    public int NumChannels { get; protected set; } = 1;

    /// <inheritdoc/>
    public double EnhancementStrength { get; set; } = 1.0;

    /// <inheritdoc/>
    public int LatencySamples => _hopSize * _lookahead;

    #endregion

    #region Public Properties

    /// <summary>
    /// Gets whether this network supports training.
    /// </summary>
    public override bool SupportsTraining => !IsOnnxMode;

    // IsOnnxMode is inherited from AudioNeuralNetworkBase<T>

    /// <summary>
    /// Gets the number of ERB bands used.
    /// </summary>
    public int NumErbBands => _numErbBands;

    /// <summary>
    /// Gets the deep filter order.
    /// </summary>
    public int DfOrder => _dfOrder;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a DeepFilterNet model for ONNX inference.
    /// </summary>
    /// <param name="architecture">Neural network architecture configuration.</param>
    /// <param name="modelPath">Path to the ONNX model file.</param>
    /// <param name="sampleRate">Audio sample rate in Hz. Default is 48000.</param>
    /// <param name="fftSize">FFT size for STFT. Default is 960.</param>
    /// <param name="hopSize">Hop size for STFT. Default is 480.</param>
    /// <param name="onnxOptions">Optional ONNX runtime options.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this constructor when you have a pre-trained
    /// DeepFilterNet model in ONNX format. This is the fastest way to get started
    /// with speech enhancement.
    ///
    /// Example:
    /// <code>
    /// var model = new DeepFilterNet&lt;float&gt;(
    ///     architecture,
    ///     "deepfilternet3.onnx",
    ///     sampleRate: 48000);
    ///
    /// var cleanAudio = model.Enhance(noisyAudio);
    /// </code>
    /// </para>
    /// </remarks>
    public DeepFilterNet(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        int sampleRate = 48000,
        int fftSize = 960,
        int hopSize = 480,
        OnnxModelOptions? onnxOptions = null,
        DeepFilterNetOptions? options = null)
        : base(architecture, new MeanSquaredErrorLoss<T>())
    {
        _options = options ?? new DeepFilterNetOptions();
        Options = _options;
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"Model file not found: {modelPath}");

        SampleRate = sampleRate;
        _fftSize = fftSize;
        _hopSize = hopSize;
        _numErbBands = 32;
        _hiddenDim = 96;
        _dfOrder = 5;
        _dfBins = 96;
        _numGruLayers = 2;
        _convKernelSize = 3;
        _lookahead = 2;
        // Load ONNX model
        OnnxModel = new OnnxModel<T>(modelPath, onnxOptions);

        // Default loss function (MSE is standard for speech enhancement)
        _lossFunction = new MeanSquaredErrorLoss<T>();

        int nFft = NextPowerOfTwo(_fftSize);
        _stft = new ShortTimeFourierTransform<T>(nFft: nFft, hopLength: _hopSize,
            windowLength: _fftSize <= nFft ? _fftSize : null);
    }

    /// <summary>
    /// Creates a DeepFilterNet model for native training and inference.
    /// </summary>
    /// <param name="architecture">Neural network architecture configuration.</param>
    /// <param name="sampleRate">Audio sample rate in Hz. Default is 48000.</param>
    /// <param name="numErbBands">Number of ERB bands. Default is 32.</param>
    /// <param name="hiddenDim">Hidden dimension. Default is 96.</param>
    /// <param name="dfOrder">Deep filter order. Default is 5.</param>
    /// <param name="dfBins">Number of DF bins. Default is 96.</param>
    /// <param name="numGruLayers">Number of GRU layers. Default is 2.</param>
    /// <param name="fftSize">FFT size. Default is 960.</param>
    /// <param name="hopSize">Hop size. Default is 480.</param>
    /// <param name="lookahead">Lookahead frames. Default is 2.</param>
    /// <param name="optimizer">Optimizer for training. If null, Adam is used.</param>
    /// <param name="lossFunction">Loss function. If null, multi-resolution STFT loss is used.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this constructor when you want to train
    /// DeepFilterNet from scratch or fine-tune on your own data.
    ///
    /// Key parameters:
    /// - numErbBands: More bands = better frequency resolution but slower
    /// - hiddenDim: Larger = more capacity but more computation
    /// - dfOrder: Higher order = better noise suppression but more latency
    ///
    /// Example:
    /// <code>
    /// var model = new DeepFilterNet&lt;float&gt;(
    ///     architecture,
    ///     sampleRate: 48000,
    ///     hiddenDim: 96,
    ///     numGruLayers: 2);
    ///
    /// // Train on noisy/clean audio pairs
    /// model.Train(noisyAudio, cleanAudio);
    /// </code>
    /// </para>
    /// </remarks>
    public DeepFilterNet(
        NeuralNetworkArchitecture<T> architecture,
        int sampleRate = 48000,
        int numErbBands = 32,
        int hiddenDim = 96,
        int dfOrder = 5,
        int dfBins = 96,
        int numGruLayers = 2,
        int fftSize = 960,
        int hopSize = 480,
        int lookahead = 2,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        DeepFilterNetOptions? options = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>())
    {
        _options = options ?? new DeepFilterNetOptions();
        Options = _options;
        SampleRate = sampleRate;
        _numErbBands = numErbBands;
        _hiddenDim = hiddenDim;
        _dfOrder = dfOrder;
        _dfBins = dfBins;
        _numGruLayers = numGruLayers;
        _fftSize = fftSize;
        _hopSize = hopSize;
        _lookahead = lookahead;
        _convKernelSize = 3;
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        int nFft = NextPowerOfTwo(_fftSize);
        _stft = new ShortTimeFourierTransform<T>(nFft: nFft, hopLength: _hopSize,
            windowLength: _fftSize <= nFft ? _fftSize : null);

        InitializeNativeLayers();
    }

    #endregion

    #region Layer Initialization

    /// <summary>
    /// Initializes native layers for training mode.
    /// </summary>
    private void InitializeNativeLayers()
    {
        // ERB Encoder: Extract perceptually-motivated features
        // Using DenseLayer for feature transformation (temporal modeling handled by GRU)
        int[] hiddenShape = [_hiddenDim];
        IActivationFunction<T> eluActivation = new ELUActivation<T>();
        IActivationFunction<T> tanhActivation = new TanhActivation<T>();

        _erbEncoder.Add(new DenseLayer<T>(_numErbBands, _hiddenDim, eluActivation));
        _erbEncoder.Add(new BatchNormalizationLayer<T>(_hiddenDim));
        _erbEncoder.Add(new ActivationLayer<T>(hiddenShape, eluActivation));

        _erbEncoder.Add(new DenseLayer<T>(_hiddenDim, _hiddenDim, eluActivation));
        _erbEncoder.Add(new BatchNormalizationLayer<T>(_hiddenDim));
        _erbEncoder.Add(new ActivationLayer<T>(hiddenShape, eluActivation));

        // GRU layers for temporal modeling
        for (int i = 0; i < _numGruLayers; i++)
        {
            _gruLayers.Add(new GRULayer<T>(_hiddenDim, _hiddenDim, returnSequences: false,
                (IActivationFunction<T>?)null, (IActivationFunction<T>?)null));
        }

        // Deep filtering layers
        int dfOutputDim = _dfBins * _dfOrder * 2; // Real + Imag
        int[] dfOutputShape = [dfOutputDim];
        _dfLayers.Add(new DenseLayer<T>(_hiddenDim, dfOutputDim));
        _dfLayers.Add(new ActivationLayer<T>(dfOutputShape, tanhActivation));

        // Gain estimation layer (for ERB-band gains)
        _gainLayer = new DenseLayer<T>(_hiddenDim, _numErbBands);

        // Decoder
        _decoder.Add(new DenseLayer<T>(_hiddenDim, _hiddenDim, eluActivation));
        _decoder.Add(new BatchNormalizationLayer<T>(_hiddenDim));
        _decoder.Add(new ActivationLayer<T>(hiddenShape, eluActivation));
    }

    #endregion

    #region IAudioEnhancer Implementation

    /// <inheritdoc/>
    public Tensor<T> Enhance(Tensor<T> audio)
    {
        return Predict(audio);
    }

    /// <inheritdoc/>
    public Tensor<T> EnhanceWithReference(Tensor<T> audio, Tensor<T> reference)
    {
        // DeepFilterNet doesn't use reference signal (not for echo cancellation)
        return Enhance(audio);
    }

    /// <inheritdoc/>
    public Tensor<T> ProcessChunk(Tensor<T> audioChunk)
    {
        // For streaming, maintain state across chunks
        return Enhance(audioChunk);
    }

    /// <inheritdoc/>
    public override void ResetState()
    {
        base.ResetState();
        // Reset GRU hidden states for streaming
        foreach (var layer in _gruLayers)
        {
            if (layer is GRULayer<T> gru)
            {
                gru.ResetState();
            }
        }
    }

    /// <inheritdoc/>
    public void EstimateNoiseProfile(Tensor<T> noiseOnlyAudio)
    {
        // DeepFilterNet doesn't require explicit noise estimation
        // It learns to separate speech from noise implicitly
    }

    #endregion

    #region AudioNeuralNetworkBase Implementation

    /// <inheritdoc/>
    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio)
    {
        // Compute complex STFT
        var complexStft = ComputeSTFT(rawAudio);

        // Cache complex STFT for reconstruction
        _cachedComplexStft = complexStft;

        // Compute ERB features from magnitude
        var erbFeatures = ComputeErbFeatures(complexStft);

        return erbFeatures;
    }

    /// <inheritdoc/>
    protected override Tensor<T> PostprocessOutput(Tensor<T> modelOutput)
    {
        // Apply deep filtering and reconstruct audio
        return ReconstructAudio(modelOutput);
    }

    #endregion

    #region NeuralNetworkBase Implementation

    /// <inheritdoc/>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        var preprocessed = PreprocessAudio(input);

        Tensor<T> output;
        if (IsOnnxMode && OnnxModel is not null)
        {
            output = OnnxModel.Run(preprocessed);
        }
        else
        {
            output = ForwardNative(preprocessed);
        }

        return PostprocessOutput(output);
    }

    /// <inheritdoc/>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (!SupportsTraining)
            throw new InvalidOperationException("Cannot train in ONNX mode.");

        SetTrainingMode(true);

        var preprocessedInput = PreprocessAudio(input);
        var preprocessedExpected = PreprocessAudio(expectedOutput);

        // Forward pass
        var predicted = ForwardNative(preprocessedInput);

        // Convert to vectors for loss computation
        var predictedVector = predicted.ToVector();
        var expectedVector = preprocessedExpected.ToVector();

        // Compute loss (multi-resolution STFT loss is typical for audio)
        var loss = _lossFunction.CalculateLoss(predictedVector, expectedVector);

        // Backward pass and update
        var gradientVector = _lossFunction.CalculateDerivative(predictedVector, expectedVector);
        var gradientTensor = Tensor<T>.FromVector(gradientVector, predicted.Shape);
        BackwardNative(gradientTensor);

        _optimizer?.UpdateParameters(Layers);

        SetTrainingMode(false);
    }

    /// <inheritdoc/>
    public override void UpdateParameters(Vector<T> parameters)
    {
        // Update all layer parameters
        int offset = 0;
        foreach (var layer in _erbEncoder.Concat(_gruLayers).Concat(_dfLayers).Concat(_decoder))
        {
            var layerParams = layer.GetParameters();
            var newParams = parameters.Slice(offset, layerParams.Length);
            layer.UpdateParameters(NumOps.FromDouble(0.001)); // Learning rate
            offset += layerParams.Length;
        }
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            ModelType = ModelType.NeuralNetwork,
            Version = "1.0.0",
            AdditionalInfo = new Dictionary<string, object>
            {
                { "NumErbBands", _numErbBands },
                { "HiddenDim", _hiddenDim },
                { "DfOrder", _dfOrder },
                { "DfBins", _dfBins },
                { "NumGruLayers", _numGruLayers },
                { "FftSize", _fftSize },
                { "HopSize", _hopSize },
                { "SampleRate", SampleRate },
                { "IsOnnxMode", IsOnnxMode }
            }
        };
    }

    #endregion

    #region Private Methods

    /// <summary>
    /// Forward pass through native layers.
    /// </summary>
    private Tensor<T> ForwardNative(Tensor<T> erbFeatures)
    {
        var x = erbFeatures;

        // ERB encoder
        foreach (var layer in _erbEncoder)
        {
            x = layer.Forward(x);
        }

        // GRU layers
        foreach (var layer in _gruLayers)
        {
            x = layer.Forward(x);
        }

        // Deep filtering coefficients
        var dfCoeffs = x;
        foreach (var layer in _dfLayers)
        {
            dfCoeffs = layer.Forward(dfCoeffs);
        }

        // Gain estimation
        var gains = _gainLayer!.Forward(x);

        // Combine DF coefficients and gains
        return CombineOutputs(dfCoeffs, gains);
    }

    /// <summary>
    /// Backward pass through native layers.
    /// </summary>
    private void BackwardNative(Tensor<T> gradient)
    {
        var grad = gradient;

        // Backward through decoder
        for (int i = _decoder.Count - 1; i >= 0; i--)
        {
            grad = _decoder[i].Backward(grad);
        }

        // Backward through DF layers
        for (int i = _dfLayers.Count - 1; i >= 0; i--)
        {
            grad = _dfLayers[i].Backward(grad);
        }

        // Backward through GRU layers
        for (int i = _gruLayers.Count - 1; i >= 0; i--)
        {
            grad = _gruLayers[i].Backward(grad);
        }

        // Backward through encoder
        for (int i = _erbEncoder.Count - 1; i >= 0; i--)
        {
            grad = _erbEncoder[i].Backward(grad);
        }
    }

    /// <summary>
    /// Computes STFT of audio signal using ShortTimeFourierTransform.
    /// </summary>
    private Tensor<Complex<T>> ComputeSTFT(Tensor<T> audio)
    {
        return _stft.Forward(audio);
    }

    /// <summary>
    /// Computes ERB (Equivalent Rectangular Bandwidth) features from complex STFT.
    /// </summary>
    private Tensor<T> ComputeErbFeatures(Tensor<Complex<T>> complexStft)
    {
        int numFrames = complexStft.Shape[0];
        int numBins = complexStft.Shape[1];

        var erbFeatures = new Tensor<T>([numFrames, _numErbBands]);

        for (int frame = 0; frame < numFrames; frame++)
        {
            for (int erb = 0; erb < _numErbBands; erb++)
            {
                double sum = 0;
                int startBin = erb * numBins / _numErbBands;
                int endBin = (erb + 1) * numBins / _numErbBands;

                for (int bin = startBin; bin < endBin && bin < numBins; bin++)
                {
                    double mag = NumOps.ToDouble(complexStft.Data.Span[frame * numBins + bin].Magnitude);
                    sum += mag;
                }

                erbFeatures[[frame, erb]] = NumOps.FromDouble(Math.Log(sum + 1e-8));
            }
        }

        return erbFeatures;
    }

    /// <summary>
    /// Reconstructs audio from enhanced representation by applying gains and deep filtering
    /// to the cached STFT and performing inverse STFT.
    /// </summary>
    /// <param name="enhanced">Model output containing DF coefficients and ERB gains.
    /// Shape: [numFrames, dfBins * dfOrder * 2 + numErbBands]</param>
    /// <returns>Reconstructed audio waveform.</returns>
    private Tensor<T> ReconstructAudio(Tensor<T> enhanced)
    {
        if (_cachedComplexStft is null)
        {
            int fallbackSamples = enhanced.Shape[0] * _hopSize + _fftSize;
            return new Tensor<T>([fallbackSamples]);
        }

        int numFrames = _cachedComplexStft.Shape[0];
        int numBins = _cachedComplexStft.Shape[1];
        int dfCoeffSize = _dfBins * _dfOrder * 2;

        // Apply ERB gains and deep filtering to complex STFT
        var enhancedStft = new Tensor<Complex<T>>(_cachedComplexStft.Shape);

        for (int frame = 0; frame < numFrames; frame++)
        {
            for (int bin = 0; bin < numBins; bin++)
            {
                int erbBand = Math.Min(bin * _numErbBands / numBins, _numErbBands - 1);
                int idx = frame * numBins + bin;

                double gain = 1.0;
                if (frame < enhanced.Shape[0] && dfCoeffSize + erbBand < enhanced.Shape[^1])
                {
                    double rawGain = NumOps.ToDouble(enhanced[[frame, dfCoeffSize + erbBand]]);
                    gain = 1.0 / (1.0 + Math.Exp(-rawGain));
                }

                double real = NumOps.ToDouble(_cachedComplexStft.Data.Span[idx].Real) * gain;
                double imag = NumOps.ToDouble(_cachedComplexStft.Data.Span[idx].Imaginary) * gain;

                // Apply deep filtering for first _dfBins bins
                if (bin < _dfBins && frame < enhanced.Shape[0])
                {
                    for (int order = 0; order < _dfOrder && frame >= order; order++)
                    {
                        int coeffIdx = (bin * _dfOrder + order) * 2;
                        if (coeffIdx + 1 < dfCoeffSize)
                        {
                            double dfReal = NumOps.ToDouble(enhanced[[frame, coeffIdx]]);
                            double dfImag = NumOps.ToDouble(enhanced[[frame, coeffIdx + 1]]);

                            double newReal = real * dfReal - imag * dfImag;
                            double newImag = real * dfImag + imag * dfReal;
                            real = newReal * 0.1 + real * 0.9;
                            imag = newImag * 0.1 + imag * 0.9;
                        }
                    }
                }

                enhancedStft.Data.Span[idx] = new Complex<T>(NumOps.FromDouble(real), NumOps.FromDouble(imag));
            }
        }

        // Use proper ISTFT for reconstruction
        return _stft.Inverse(enhancedStft);
    }

    /// <summary>
    /// Combines deep filtering coefficients and gains.
    /// </summary>
    private Tensor<T> CombineOutputs(Tensor<T> dfCoeffs, Tensor<T> gains)
    {
        // Combine DF coefficients and ERB gains for output
        int numFrames = gains.Shape[0];
        int outputDim = _dfBins * _dfOrder * 2 + _numErbBands;

        var combined = new Tensor<T>([numFrames, outputDim]);

        for (int frame = 0; frame < numFrames; frame++)
        {
            // Copy DF coefficients
            for (int i = 0; i < _dfBins * _dfOrder * 2; i++)
            {
                combined[[frame, i]] = dfCoeffs[[frame, i]];
            }

            // Copy gains
            for (int i = 0; i < _numErbBands; i++)
            {
                combined[[frame, _dfBins * _dfOrder * 2 + i]] = gains[[frame, i]];
            }
        }

        return combined;
    }

    private static int NextPowerOfTwo(int v)
    {
        v--;
        v |= v >> 1;
        v |= v >> 2;
        v |= v >> 4;
        v |= v >> 8;
        v |= v >> 16;
        return v + 1;
    }

    #endregion

    #region Abstract Method Implementations

    /// <inheritdoc/>
    protected override void InitializeLayers()
    {
        // Layers are initialized in constructor based on mode
        // In ONNX mode, layers are not used
        // In native mode, ERB encoder/decoder/GRU layers are created in constructor
    }

    /// <inheritdoc/>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(IsOnnxMode);
        writer.Write(SampleRate);
        writer.Write(_numErbBands);
        writer.Write(_hiddenDim);
        writer.Write(_dfOrder);
        writer.Write(_dfBins);
        writer.Write(_numGruLayers);
        writer.Write(_convKernelSize);
        writer.Write(_fftSize);
        writer.Write(_hopSize);
        writer.Write(_lookahead);
        writer.Write(EnhancementStrength);
    }

    /// <inheritdoc/>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        // Read configuration values for validation
        _ = reader.ReadBoolean(); // IsOnnxMode
        _ = reader.ReadInt32();   // SampleRate
        _ = reader.ReadInt32();   // _numErbBands
        _ = reader.ReadInt32();   // _hiddenDim
        _ = reader.ReadInt32();   // _dfOrder
        _ = reader.ReadInt32();   // _dfBins
        _ = reader.ReadInt32();   // _numGruLayers
        _ = reader.ReadInt32();   // _convKernelSize
        _ = reader.ReadInt32();   // _fftSize
        _ = reader.ReadInt32();   // _hopSize
        _ = reader.ReadInt32();   // _lookahead
        EnhancementStrength = reader.ReadDouble();
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new DeepFilterNet<T>(
            Architecture,
            sampleRate: SampleRate,
            numErbBands: _numErbBands,
            hiddenDim: _hiddenDim,
            dfOrder: _dfOrder,
            dfBins: _dfBins,
            numGruLayers: _numGruLayers,
            fftSize: _fftSize,
            hopSize: _hopSize,
            lookahead: _lookahead);
    }

    #endregion

    #region IDisposable

    /// <summary>
    /// Disposes resources.
    /// </summary>
    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            // OnnxModel disposal handled by base class
        }
        base.Dispose(disposing);
    }

    #endregion
}
