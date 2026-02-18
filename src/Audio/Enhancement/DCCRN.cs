using AiDotNet.ActivationFunctions;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Audio.Enhancement;

/// <summary>
/// DCCRN - Deep Complex Convolution Recurrent Network for speech enhancement.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// DCCRN operates directly on complex-valued spectrograms, preserving phase information
/// for high-quality speech enhancement. Key features:
/// - Complex-valued convolutions for better spectral modeling
/// - LSTM layers for temporal dependencies
/// - Skip connections for gradient flow
/// - Mask-based enhancement for clean speech estimation
/// </para>
/// <para>
/// <b>For Beginners:</b> DCCRN is a neural network designed specifically for
/// cleaning up noisy audio. Unlike simpler methods that only work with the
/// "loudness" of frequencies, DCCRN also considers the "timing" (phase),
/// which results in more natural-sounding enhanced audio.
///
/// Think of it like this: regular enhancement is like adjusting volume
/// of different frequencies, while DCCRN can also adjust the timing of
/// sound waves to better reconstruct the original clean speech.
///
/// Usage:
/// <code>
/// var model = new DCCRN&lt;float&gt;(architecture, "dccrn.onnx");
/// var cleanAudio = model.Enhance(noisyAudio);
/// </code>
/// </para>
/// <para>
/// Reference: "DCCRN: Deep Complex Convolution Recurrent Network for Phase-Aware
/// Speech Enhancement" by Hu et al., Interspeech 2020
/// </para>
/// </remarks>
public class DCCRN<T> : AudioNeuralNetworkBase<T>, IAudioEnhancer<T>
{
    private readonly DCCRNOptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    #region Model Architecture Parameters

    /// <summary>
    /// Number of encoder/decoder stages.
    /// </summary>
    private int _numStages;

    /// <summary>
    /// Base number of channels.
    /// </summary>
    private int _baseChannels;

    /// <summary>
    /// LSTM hidden dimension.
    /// </summary>
    private int _lstmHiddenDim;

    /// <summary>
    /// Number of LSTM layers.
    /// </summary>
    private int _numLstmLayers;

    /// <summary>
    /// FFT size for STFT.
    /// </summary>
    private int _fftSize;

    /// <summary>
    /// Hop size for STFT.
    /// </summary>
    private int _hopSize;

    /// <summary>
    /// Whether to use complex mask estimation.
    /// </summary>
    private bool _useComplexMask;

    /// <summary>
    /// Kernel size for convolutions.
    /// </summary>
    private int _kernelSize;

    /// <summary>
    /// Stride for convolutions.
    /// </summary>
    private int _stride;

    #endregion

    #region Network Layers

    /// <summary>
    /// Encoder layers (complex convolutions).
    /// </summary>
    private readonly List<ILayer<T>> _encoder = [];

    /// <summary>
    /// LSTM layers.
    /// </summary>
    private readonly List<ILayer<T>> _lstmLayers = [];

    /// <summary>
    /// Projection layer to map LSTM output back to encoder spatial dimensions.
    /// </summary>
    private ILayer<T>? _lstmProjection;

    /// <summary>
    /// Encoder output dimension (channels * freqBins) for LSTM projection.
    /// </summary>
    private int _encoderOutputDim;

    /// <summary>
    /// Decoder layers (complex transposed convolutions).
    /// </summary>
    private readonly List<ILayer<T>> _decoder = [];

    /// <summary>
    /// Skip connection layers.
    /// </summary>
    private readonly List<ILayer<T>> _skipLayers = [];

    /// <summary>
    /// Mask estimation layer.
    /// </summary>
    private ILayer<T>? _maskLayer;

    #endregion

    #region ONNX Mode Fields

    // ONNX model is inherited from AudioNeuralNetworkBase<T>.OnnxModel

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

    /// <summary>
    /// Cached encoder outputs for skip connections.
    /// </summary>
    private readonly List<Tensor<T>> _encoderOutputs = [];

    #endregion

    #region IAudioEnhancer Properties

    /// <inheritdoc/>
    public int NumChannels { get; protected set; } = 1;

    /// <inheritdoc/>
    public double EnhancementStrength { get; set; } = 1.0;

    /// <inheritdoc/>
    public int LatencySamples => _fftSize;

    #endregion

    #region Public Properties

    /// <summary>
    /// Gets whether this network supports training.
    /// </summary>
    public override bool SupportsTraining => !IsOnnxMode;

    // IsOnnxMode is inherited from AudioNeuralNetworkBase<T>

    /// <summary>
    /// Gets the number of encoder/decoder stages.
    /// </summary>
    public int NumStages => _numStages;

    /// <summary>
    /// Gets whether complex mask is used.
    /// </summary>
    public bool UseComplexMask => _useComplexMask;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a DCCRN model for ONNX inference.
    /// </summary>
    /// <param name="architecture">Neural network architecture configuration.</param>
    /// <param name="modelPath">Path to the ONNX model file.</param>
    /// <param name="sampleRate">Audio sample rate in Hz. Default is 16000.</param>
    /// <param name="fftSize">FFT size for STFT. Default is 512.</param>
    /// <param name="hopSize">Hop size for STFT. Default is 256.</param>
    /// <param name="onnxOptions">Optional ONNX runtime options.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this constructor to load a pre-trained DCCRN
    /// model for speech enhancement.
    ///
    /// Example:
    /// <code>
    /// var model = new DCCRN&lt;float&gt;(architecture, "dccrn_16k.onnx");
    /// var clean = model.Enhance(noisy);
    /// </code>
    /// </para>
    /// </remarks>
    public DCCRN(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        int sampleRate = 16000,
        int fftSize = 512,
        int hopSize = 256,
        OnnxModelOptions? onnxOptions = null,
        DCCRNOptions? options = null)
        : base(architecture, new MeanSquaredErrorLoss<T>())
    {
        _options = options ?? new DCCRNOptions();
        Options = _options;
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"Model file not found: {modelPath}");

        SampleRate = sampleRate;
        _fftSize = fftSize;
        _hopSize = hopSize;
        _numStages = 6;
        _baseChannels = 32;
        _lstmHiddenDim = 256;
        _numLstmLayers = 2;
        _useComplexMask = true;
        _kernelSize = 5;
        _stride = 2;
        OnnxModel = new OnnxModel<T>(modelPath, onnxOptions);

        // Default loss function (MSE is standard for speech enhancement)
        _lossFunction = new MeanSquaredErrorLoss<T>();
    }

    /// <summary>
    /// Creates a DCCRN model for native training and inference.
    /// </summary>
    /// <param name="architecture">Neural network architecture configuration.</param>
    /// <param name="sampleRate">Audio sample rate in Hz. Default is 16000.</param>
    /// <param name="numStages">Number of encoder/decoder stages. Default is 6.</param>
    /// <param name="baseChannels">Base number of channels. Default is 32.</param>
    /// <param name="lstmHiddenDim">LSTM hidden dimension. Default is 256.</param>
    /// <param name="numLstmLayers">Number of LSTM layers. Default is 2.</param>
    /// <param name="fftSize">FFT size. Default is 512.</param>
    /// <param name="hopSize">Hop size. Default is 256.</param>
    /// <param name="useComplexMask">Use complex mask estimation. Default is true.</param>
    /// <param name="kernelSize">Convolution kernel size. Default is 5.</param>
    /// <param name="stride">Convolution stride. Default is 2.</param>
    /// <param name="optimizer">Optimizer for training.</param>
    /// <param name="lossFunction">Loss function for training.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this constructor to train DCCRN from scratch.
    ///
    /// Key parameters:
    /// - numStages: More stages = deeper network, better results but slower
    /// - baseChannels: More channels = more capacity
    /// - useComplexMask: true preserves phase for better quality
    ///
    /// Example:
    /// <code>
    /// var model = new DCCRN&lt;float&gt;(
    ///     architecture,
    ///     numStages: 6,
    ///     baseChannels: 32,
    ///     useComplexMask: true);
    ///
    /// model.Train(noisyBatch, cleanBatch);
    /// </code>
    /// </para>
    /// </remarks>
    public DCCRN(
        NeuralNetworkArchitecture<T> architecture,
        int sampleRate = 16000,
        int numStages = 6,
        int baseChannels = 32,
        int lstmHiddenDim = 256,
        int numLstmLayers = 2,
        int fftSize = 512,
        int hopSize = 256,
        bool useComplexMask = true,
        int kernelSize = 5,
        int stride = 2,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        DCCRNOptions? options = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>())
    {
        _options = options ?? new DCCRNOptions();
        Options = _options;
        // Validate parameters
        if (sampleRate <= 0)
            throw new ArgumentOutOfRangeException(nameof(sampleRate), "Sample rate must be positive.");
        if (numStages <= 0)
            throw new ArgumentOutOfRangeException(nameof(numStages), "Number of stages must be positive.");
        if (baseChannels <= 0)
            throw new ArgumentOutOfRangeException(nameof(baseChannels), "Base channels must be positive.");
        if (lstmHiddenDim <= 0)
            throw new ArgumentOutOfRangeException(nameof(lstmHiddenDim), "LSTM hidden dimension must be positive.");
        if (numLstmLayers <= 0)
            throw new ArgumentOutOfRangeException(nameof(numLstmLayers), "Number of LSTM layers must be positive.");
        if (fftSize <= 0 || (fftSize & (fftSize - 1)) != 0)
            throw new ArgumentOutOfRangeException(nameof(fftSize), "FFT size must be a positive power of 2.");
        if (hopSize <= 0 || hopSize > fftSize)
            throw new ArgumentOutOfRangeException(nameof(hopSize), "Hop size must be positive and not exceed FFT size.");

        SampleRate = sampleRate;
        _numStages = numStages;
        _baseChannels = baseChannels;
        _lstmHiddenDim = lstmHiddenDim;
        _numLstmLayers = numLstmLayers;
        _fftSize = fftSize;
        _hopSize = hopSize;
        _useComplexMask = useComplexMask;
        _kernelSize = kernelSize;
        _stride = stride;
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        InitializeNativeLayers();
    }

    #endregion

    #region Layer Initialization

    /// <summary>
    /// Initializes native layers for training mode.
    /// </summary>
    private void InitializeNativeLayers()
    {
        int freqBins = _fftSize / 2 + 1;

        var layers = (Architecture.Layers != null && Architecture.Layers.Count > 0)
            ? Architecture.Layers.ToList()
            : LayerHelper<T>.CreateDCCRNLayers(
                fftSize: _fftSize, baseChannels: _baseChannels, numStages: _numStages,
                numLstmLayers: _numLstmLayers, lstmHiddenDim: _lstmHiddenDim,
                kernelSize: _kernelSize, stride: _stride).ToList();

        Layers.Clear();
        _encoder.Clear();
        _skipLayers.Clear();
        _lstmLayers.Clear();
        _decoder.Clear();
        Layers.AddRange(layers);

        // Distribute to internal sub-lists for forward pass
        int idx = 0;

        // Encoder: per stage = Conv + BN + SkipConv = 3 layers
        for (int i = 0; i < _numStages && idx + 2 < layers.Count; i++)
        {
            _encoder.Add(layers[idx++]); // Conv
            _encoder.Add(layers[idx++]); // BatchNorm
            _skipLayers.Add(layers[idx++]); // Skip connection
        }

        // Store encoder output dim for projection using freqBins (not _fftSize)
        int inChannels = _baseChannels * (int)Math.Pow(2, Math.Min(_numStages - 1, 4));
        _encoderOutputDim = inChannels * (freqBins / (int)Math.Pow(_stride, _numStages));

        // LSTM layers
        for (int i = 0; i < _numLstmLayers && idx < layers.Count; i++)
            _lstmLayers.Add(layers[idx++]);

        // LSTM projection
        if (idx < layers.Count)
            _lstmProjection = layers[idx++];

        // Decoder: per stage = Deconv + optional BN (last stage has no BN)
        for (int i = 0; i < _numStages && idx < layers.Count; i++)
        {
            _decoder.Add(layers[idx++]); // Deconv
            if (i < _numStages - 1 && idx < layers.Count)
                _decoder.Add(layers[idx++]); // BatchNorm
        }

        // Mask layer: use from layer list if remaining, otherwise create
        if (idx < layers.Count)
        {
            _maskLayer = layers[idx];
        }
        else
        {
            int[] maskShape = [1, 2, freqBins, 1];
            _maskLayer = _useComplexMask
                ? new ActivationLayer<T>(maskShape, (IActivationFunction<T>)new TanhActivation<T>())
                : new ActivationLayer<T>(maskShape, (IActivationFunction<T>)new SigmoidActivation<T>());
            Layers.Add(_maskLayer);
        }
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
        return Enhance(audio);
    }

    /// <inheritdoc/>
    public Tensor<T> ProcessChunk(Tensor<T> audioChunk)
    {
        return Enhance(audioChunk);
    }

    /// <inheritdoc/>
    public override void ResetState()
    {
        base.ResetState();
        foreach (var layer in _lstmLayers)
        {
            if (layer is LSTMLayer<T> lstm)
            {
                lstm.ResetState();
            }
        }
    }

    /// <inheritdoc/>
    public void EstimateNoiseProfile(Tensor<T> noiseOnlyAudio)
    {
        // DCCRN learns noise suppression implicitly
    }

    #endregion

    #region AudioNeuralNetworkBase Implementation

    /// <inheritdoc/>
    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio)
    {
        return ComputeComplexSTFT(rawAudio);
    }

    /// <inheritdoc/>
    protected override Tensor<T> PostprocessOutput(Tensor<T> modelOutput)
    {
        return ComputeInverseSTFT(modelOutput);
    }

    #endregion

    #region NeuralNetworkBase Implementation

    /// <inheritdoc/>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        var stft = PreprocessAudio(input);

        Tensor<T> enhancedStft;
        if (IsOnnxMode && OnnxModel is not null)
        {
            enhancedStft = OnnxModel.Run(stft);
        }
        else
        {
            enhancedStft = ForwardNative(stft);
        }

        return PostprocessOutput(enhancedStft);
    }

    /// <inheritdoc/>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (!SupportsTraining)
            throw new InvalidOperationException("Cannot train in ONNX mode.");

        SetTrainingMode(true);

        var noisyStft = PreprocessAudio(input);
        var cleanStft = PreprocessAudio(expectedOutput);

        // Forward pass
        var enhancedStft = ForwardNative(noisyStft);

        // Convert to vectors for loss computation
        var enhancedVector = enhancedStft.ToVector();
        var cleanVector = cleanStft.ToVector();

        // SI-SNR or MSE loss on STFT
        var loss = _lossFunction.CalculateLoss(enhancedVector, cleanVector);

        // Backward pass
        var gradientVector = _lossFunction.CalculateDerivative(enhancedVector, cleanVector);
        var gradientTensor = Tensor<T>.FromVector(gradientVector, enhancedStft.Shape);
        BackwardNative(gradientTensor);

        // Update parameters via optimizer
        _optimizer?.UpdateParameters(Layers);

        SetTrainingMode(false);
    }

    /// <inheritdoc/>
    public override void UpdateParameters(Vector<T> parameters)
    {
        int offset = 0;
        var allLayers = _encoder.Concat(_lstmLayers).Concat(_decoder);

        foreach (var layer in allLayers)
        {
            var layerParams = layer.GetParameters();
            var newParams = parameters.Slice(offset, layerParams.Length);
            // Apply actual parameter updates from optimizer
            for (int i = 0; i < layerParams.Length; i++)
            {
                layerParams[i] = newParams[i];
            }
            layer.SetParameters(layerParams);
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
                { "NumStages", _numStages },
                { "BaseChannels", _baseChannels },
                { "LstmHiddenDim", _lstmHiddenDim },
                { "NumLstmLayers", _numLstmLayers },
                { "FftSize", _fftSize },
                { "HopSize", _hopSize },
                { "UseComplexMask", _useComplexMask },
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
    private Tensor<T> ForwardNative(Tensor<T> stft)
    {
        _encoderOutputs.Clear();
        var x = stft;

        // Encoder with skip connection caching
        // Each encoder stage has 2 layers: Conv (with built-in activation) + BatchNorm
        int skipIdx = 0;
        for (int i = 0; i < _encoder.Count; i += 2)
        {
            x = _encoder[i].Forward(x);     // Conv (includes LeakyReLU activation)
            x = _encoder[i + 1].Forward(x); // BatchNorm

            if (skipIdx < _skipLayers.Count)
            {
                var skip = _skipLayers[skipIdx].Forward(x);
                _encoderOutputs.Add(skip);
                skipIdx++;
            }
        }

        // Reshape for LSTM
        var batchSize = x.Shape[0];
        var channels = x.Shape[1];
        var freqBins = x.Shape[2];
        var timeFrames = x.Shape[3];

        var reshaped = x.Reshape([batchSize, timeFrames, channels * freqBins]);

        // LSTM layers
        foreach (var lstm in _lstmLayers)
        {
            reshaped = lstm.Forward(reshaped);
        }

        // Project LSTM output back to encoder spatial dimensions
        if (_lstmProjection is not null)
        {
            // Apply projection per timestep: [batch, time, hidden] -> [batch, time, encoderDim]
            var projectedData = new T[batchSize * timeFrames * _encoderOutputDim];
            for (int b = 0; b < batchSize; b++)
            {
                for (int t = 0; t < timeFrames; t++)
                {
                    // Extract timestep data
                    var timestepInput = new Tensor<T>([_lstmHiddenDim]);
                    for (int h = 0; h < _lstmHiddenDim; h++)
                    {
                        int idx = b * timeFrames * _lstmHiddenDim + t * _lstmHiddenDim + h;
                        if (idx < reshaped.Length)
                            timestepInput[h] = reshaped.GetFlat(idx);
                    }

                    // Project
                    var projected = _lstmProjection.Forward(timestepInput);

                    // Store result
                    for (int e = 0; e < _encoderOutputDim && e < projected.Length; e++)
                    {
                        int outIdx = b * timeFrames * _encoderOutputDim + t * _encoderOutputDim + e;
                        if (outIdx < projectedData.Length)
                            projectedData[outIdx] = projected.GetFlat(e);
                    }
                }
            }
            reshaped = new Tensor<T>(projectedData, [batchSize, timeFrames, _encoderOutputDim]);
        }

        // Reshape back (now dimensions match: channels * freqBins == _encoderOutputDim)
        x = reshaped.Reshape([batchSize, channels, freqBins, timeFrames]);

        // Decoder with skip connections
        int decoderLayerIdx = 0;
        for (int i = _numStages - 1; i >= 0; i--)
        {
            // Concatenate skip connection
            if (i < _encoderOutputs.Count)
            {
                x = ConcatenateChannels(x, _encoderOutputs[i]);
            }

            // Decoder layers: 2 per stage (DeconvT + BN) except last output stage which has 1 (DeconvT only)
            // Initialization creates layers in stage order: stages 0 to numStages-2 have 2 layers each,
            // stage numStages-1 has 1 layer. The loop iterates i from (numStages-1) down to 0,
            // but decoderLayerIdx increases from 0, so we process decoder stages in order 0, 1, ..., numStages-1.
            // When i > 0: we're NOT yet at the last loop iteration, process 2 layers (stages 0 to numStages-2)
            // When i == 0: we're at the last loop iteration, process 1 layer (stage numStages-1, final output)
            int layersThisStage = (i > 0) ? 2 : 1;
            for (int j = 0; j < layersThisStage; j++)
            {
                if (decoderLayerIdx < _decoder.Count)
                {
                    x = _decoder[decoderLayerIdx].Forward(x);
                    decoderLayerIdx++;
                }
            }
        }

        // Apply mask
        var mask = _maskLayer!.Forward(x);

        // Apply mask to input STFT
        return ApplyComplexMask(stft, mask);
    }

    /// <summary>
    /// Backward pass through native layers.
    /// </summary>
    private void BackwardNative(Tensor<T> gradient)
    {
        var grad = gradient;

        // Backward through mask
        grad = _maskLayer!.Backward(grad);

        // Backward through decoder
        for (int i = _decoder.Count - 1; i >= 0; i--)
        {
            grad = _decoder[i].Backward(grad);
        }

        // Backward through LSTM
        for (int i = _lstmLayers.Count - 1; i >= 0; i--)
        {
            grad = _lstmLayers[i].Backward(grad);
        }

        // Backward through encoder
        for (int i = _encoder.Count - 1; i >= 0; i--)
        {
            grad = _encoder[i].Backward(grad);
        }
    }

    /// <summary>
    /// Computes complex STFT.
    /// </summary>
    private Tensor<T> ComputeComplexSTFT(Tensor<T> audio)
    {
        var samples = audio.ToVector().ToArray();
        int numFrames = (samples.Length - _fftSize) / _hopSize + 1;
        int numBins = _fftSize / 2 + 1;

        // Shape: [1, 2, numBins, numFrames] for batch, real/imag, freq, time
        var stft = new Tensor<T>([1, 2, numBins, numFrames]);
        var window = CreateHannWindow(_fftSize);

        for (int frame = 0; frame < numFrames; frame++)
        {
            int start = frame * _hopSize;
            var frameData = new double[_fftSize];

            for (int i = 0; i < _fftSize && start + i < samples.Length; i++)
            {
                frameData[i] = NumOps.ToDouble(samples[start + i]) * window[i];
            }

            for (int k = 0; k < numBins; k++)
            {
                double real = 0, imag = 0;
                for (int n = 0; n < _fftSize; n++)
                {
                    double angle = -2 * Math.PI * k * n / _fftSize;
                    real += frameData[n] * Math.Cos(angle);
                    imag += frameData[n] * Math.Sin(angle);
                }
                stft[[0, 0, k, frame]] = NumOps.FromDouble(real);
                stft[[0, 1, k, frame]] = NumOps.FromDouble(imag);
            }
        }

        return stft;
    }

    /// <summary>
    /// Computes inverse STFT.
    /// </summary>
    private Tensor<T> ComputeInverseSTFT(Tensor<T> stft)
    {
        int numBins = stft.Shape[2];
        int numFrames = stft.Shape[3];
        int numSamples = numFrames * _hopSize + _fftSize;

        var audio = new Tensor<T>([numSamples]);
        var window = CreateHannWindow(_fftSize);

        for (int frame = 0; frame < numFrames; frame++)
        {
            int start = frame * _hopSize;

            for (int n = 0; n < _fftSize && start + n < numSamples; n++)
            {
                double sum = 0;
                for (int k = 0; k < numBins; k++)
                {
                    double real = NumOps.ToDouble(stft[[0, 0, k, frame]]);
                    double imag = NumOps.ToDouble(stft[[0, 1, k, frame]]);
                    double angle = 2 * Math.PI * k * n / _fftSize;
                    sum += real * Math.Cos(angle) - imag * Math.Sin(angle);
                }
                double currentVal = NumOps.ToDouble(audio[[start + n]]);
                audio[[start + n]] = NumOps.FromDouble(currentVal + sum * window[n] / _fftSize);
            }
        }

        return audio;
    }

    /// <summary>
    /// Applies complex mask to STFT.
    /// </summary>
    private Tensor<T> ApplyComplexMask(Tensor<T> stft, Tensor<T> mask)
    {
        var result = new Tensor<T>(stft.Shape);

        for (int i = 0; i < stft.Shape[0]; i++)
        {
            for (int k = 0; k < stft.Shape[2]; k++)
            {
                for (int t = 0; t < stft.Shape[3]; t++)
                {
                    if (_useComplexMask)
                    {
                        // Complex multiplication
                        double sr = NumOps.ToDouble(stft[[i, 0, k, t]]);
                        double si = NumOps.ToDouble(stft[[i, 1, k, t]]);
                        double mr = NumOps.ToDouble(mask[[i, 0, k, t]]);
                        double mi = NumOps.ToDouble(mask[[i, 1, k, t]]);

                        result[[i, 0, k, t]] = NumOps.FromDouble(sr * mr - si * mi);
                        result[[i, 1, k, t]] = NumOps.FromDouble(sr * mi + si * mr);
                    }
                    else
                    {
                        // Magnitude mask
                        double m = NumOps.ToDouble(mask[[i, 0, k, t]]);
                        result[[i, 0, k, t]] = NumOps.Multiply(stft[[i, 0, k, t]], NumOps.FromDouble(m));
                        result[[i, 1, k, t]] = NumOps.Multiply(stft[[i, 1, k, t]], NumOps.FromDouble(m));
                    }
                }
            }
        }

        return result;
    }

    /// <summary>
    /// Concatenates tensors along channel dimension with dimension validation.
    /// </summary>
    private Tensor<T> ConcatenateChannels(Tensor<T> a, Tensor<T> b)
    {
        // Validate spatial dimensions match for skip connections
        if (a.Shape[0] != b.Shape[0])
        {
            throw new ArgumentException(
                $"Batch dimension mismatch in skip connection: encoder={a.Shape[0]}, decoder={b.Shape[0]}");
        }

        // Warn if spatial dimensions don't match (use minimum to avoid index errors)
        if (a.Shape[2] != b.Shape[2] || a.Shape[3] != b.Shape[3])
        {
            System.Diagnostics.Debug.WriteLine(
                $"DCCRN: Spatial dimension mismatch in skip connection. " +
                $"Encoder: [{a.Shape[2]},{a.Shape[3]}], Decoder: [{b.Shape[2]},{b.Shape[3]}]. " +
                "Using minimum dimensions.");
        }

        int height = Math.Min(a.Shape[2], b.Shape[2]);
        int width = Math.Min(a.Shape[3], b.Shape[3]);
        int newChannels = a.Shape[1] + b.Shape[1];
        var result = new Tensor<T>([a.Shape[0], newChannels, height, width]);

        // Copy a (use minimum dimensions for safety)
        for (int i = 0; i < a.Shape[0]; i++)
            for (int c = 0; c < a.Shape[1]; c++)
                for (int h = 0; h < height; h++)
                    for (int w = 0; w < width; w++)
                        result[[i, c, h, w]] = a[[i, c, h, w]];

        // Copy b
        for (int i = 0; i < b.Shape[0]; i++)
            for (int c = 0; c < b.Shape[1]; c++)
                for (int h = 0; h < height; h++)
                    for (int w = 0; w < width; w++)
                        result[[i, a.Shape[1] + c, h, w]] = b[[i, c, h, w]];

        return result;
    }

    /// <summary>
    /// Creates a Hann window.
    /// </summary>
    private double[] CreateHannWindow(int size)
    {
        var window = new double[size];
        for (int i = 0; i < size; i++)
        {
            window[i] = 0.5 * (1 - Math.Cos(2 * Math.PI * i / (size - 1)));
        }
        return window;
    }

    #endregion

    #region Abstract Method Implementations

    /// <inheritdoc/>
    protected override void InitializeLayers()
    {
        // Layers are initialized in constructor based on mode
        // In ONNX mode, layers are not used
        // In native mode, encoder/decoder/LSTM layers are created in constructor
    }

    /// <inheritdoc/>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(IsOnnxMode);
        writer.Write(SampleRate);
        writer.Write(_numStages);
        writer.Write(_baseChannels);
        writer.Write(_lstmHiddenDim);
        writer.Write(_numLstmLayers);
        writer.Write(_fftSize);
        writer.Write(_hopSize);
        writer.Write(_useComplexMask);
        writer.Write(_kernelSize);
        writer.Write(_stride);
        writer.Write(EnhancementStrength);
    }

    /// <inheritdoc/>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        // Restore configuration values
        _ = reader.ReadBoolean(); // IsOnnxMode (determined by constructor, cannot change)
        SampleRate = reader.ReadInt32();
        _numStages = reader.ReadInt32();
        _baseChannels = reader.ReadInt32();
        _lstmHiddenDim = reader.ReadInt32();
        _numLstmLayers = reader.ReadInt32();
        _fftSize = reader.ReadInt32();
        _hopSize = reader.ReadInt32();
        _useComplexMask = reader.ReadBoolean();
        _kernelSize = reader.ReadInt32();
        _stride = reader.ReadInt32();
        EnhancementStrength = reader.ReadDouble();

        // Re-initialize layers if needed for native mode
        if (!IsOnnxMode)
        {
            InitializeNativeLayers();
        }
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new DCCRN<T>(
            Architecture,
            sampleRate: SampleRate,
            numStages: _numStages,
            baseChannels: _baseChannels,
            lstmHiddenDim: _lstmHiddenDim,
            numLstmLayers: _numLstmLayers,
            fftSize: _fftSize,
            hopSize: _hopSize,
            useComplexMask: _useComplexMask);
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
