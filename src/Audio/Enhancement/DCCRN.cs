using AiDotNet.ActivationFunctions;
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
    #region Model Architecture Parameters

    /// <summary>
    /// Number of encoder/decoder stages.
    /// </summary>
    private readonly int _numStages;

    /// <summary>
    /// Base number of channels.
    /// </summary>
    private readonly int _baseChannels;

    /// <summary>
    /// LSTM hidden dimension.
    /// </summary>
    private readonly int _lstmHiddenDim;

    /// <summary>
    /// Number of LSTM layers.
    /// </summary>
    private readonly int _numLstmLayers;

    /// <summary>
    /// FFT size for STFT.
    /// </summary>
    private readonly int _fftSize;

    /// <summary>
    /// Hop size for STFT.
    /// </summary>
    private readonly int _hopSize;

    /// <summary>
    /// Whether to use complex mask estimation.
    /// </summary>
    private readonly bool _useComplexMask;

    /// <summary>
    /// Kernel size for convolutions.
    /// </summary>
    private readonly int _kernelSize;

    /// <summary>
    /// Stride for convolutions.
    /// </summary>
    private readonly int _stride;

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
    private readonly ILossFunction<T> _lossFunction;

    /// <summary>
    /// Optimizer for training.
    /// </summary>
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;

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
        OnnxModelOptions? onnxOptions = null)
        : base(architecture, new MeanSquaredErrorLoss<T>())
    {
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
        _lossFunction = new MeanSquaredErrorLoss<T>();

        OnnxModel = new OnnxModel<T>(modelPath, onnxOptions);

        // Initialize optimizer (not used in ONNX mode but required for readonly field)
        _optimizer = new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);
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
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>())
    {
        SampleRate = sampleRate;
        _numStages = numStages;
        _baseChannels = baseChannels;
        _lstmHiddenDim = lstmHiddenDim;
        _numLstmLayers = numLstmLayers;
        _fftSize = fftSize;
        _hopSize = hopSize;
        _useComplexMask = useComplexMask;
        _kernelSize = 5;
        _stride = 2;
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
        int inChannels = 2; // Real + Imaginary
        int freqBins = _fftSize / 2 + 1;
        int timeDim = 100; // Approximate time dimension for initialization

        // Encoder (complex convolutions with increasing channels)
        for (int i = 0; i < _numStages; i++)
        {
            int outChannels = _baseChannels * (int)Math.Pow(2, Math.Min(i, 4));

            // Complex convolution (implemented as 2-channel real convolution)
            // ConvolutionalLayer: inputDepth, inputHeight, inputWidth, outputDepth, kernelSize, stride, padding, activation
            int currentFreqBins = freqBins / (int)Math.Pow(_stride, i);
            _encoder.Add(new ConvolutionalLayer<T>(
                inChannels, currentFreqBins, timeDim, outChannels, _kernelSize, _stride, _kernelSize / 2,
                (IActivationFunction<T>)new LeakyReLUActivation<T>()));
            _encoder.Add(new BatchNormalizationLayer<T>(outChannels));

            // Skip connection layer (1x1 convolution)
            _skipLayers.Add(new ConvolutionalLayer<T>(
                outChannels, currentFreqBins / _stride, timeDim, outChannels, 1, 1, 0));

            inChannels = outChannels;
        }

        // LSTM layers
        int lstmInputDim = inChannels * (_fftSize / (int)Math.Pow(_stride, _numStages));
        int[] lstmInputShape = [1, lstmInputDim];
        for (int i = 0; i < _numLstmLayers; i++)
        {
            _lstmLayers.Add(new LSTMLayer<T>(lstmInputDim, _lstmHiddenDim, lstmInputShape,
                (IActivationFunction<T>)new TanhActivation<T>(), (IActivationFunction<T>)new SigmoidActivation<T>()));
            lstmInputDim = _lstmHiddenDim;
            lstmInputShape = [1, lstmInputDim];
        }

        // Decoder (transposed convolutions with decreasing channels)
        int decoderChannels = _baseChannels * (int)Math.Pow(2, Math.Min(_numStages - 1, 4));
        for (int i = 0; i < _numStages; i++)
        {
            int outChannels = i < _numStages - 1
                ? _baseChannels * (int)Math.Pow(2, Math.Min(_numStages - 2 - i, 4))
                : 2; // Final output is 2 channels (real + imag)

            // Skip connection input doubles the channels
            int skipChannels = decoderChannels * 2;
            int currentFreqBins = freqBins / (int)Math.Pow(_stride, _numStages - i);
            int[] decoderInputShape = [1, skipChannels, currentFreqBins, timeDim];

            // DeconvolutionalLayer: inputShape, outputDepth, kernelSize, stride, padding, activation
            if (i < _numStages - 1)
            {
                _decoder.Add(new DeconvolutionalLayer<T>(decoderInputShape, outChannels, _kernelSize, _stride, _kernelSize / 2,
                    (IActivationFunction<T>)new LeakyReLUActivation<T>()));
                _decoder.Add(new BatchNormalizationLayer<T>(outChannels));
            }
            else
            {
                _decoder.Add(new DeconvolutionalLayer<T>(decoderInputShape, outChannels, _kernelSize, _stride, _kernelSize / 2,
                    (IActivationFunction<T>?)null));
            }

            decoderChannels = outChannels;
        }

        // Complex mask estimation activation layer
        int[] maskShape = [1, 2, freqBins, timeDim];
        _maskLayer = _useComplexMask
            ? new ActivationLayer<T>(maskShape, (IActivationFunction<T>)new TanhActivation<T>())
            : new ActivationLayer<T>(maskShape, (IActivationFunction<T>)new SigmoidActivation<T>());
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
        _optimizer.UpdateParameters(Layers);

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

        // Reshape back
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
    /// Concatenates tensors along channel dimension.
    /// </summary>
    private Tensor<T> ConcatenateChannels(Tensor<T> a, Tensor<T> b)
    {
        int newChannels = a.Shape[1] + b.Shape[1];
        var result = new Tensor<T>([a.Shape[0], newChannels, a.Shape[2], a.Shape[3]]);

        // Copy a
        for (int i = 0; i < a.Shape[0]; i++)
            for (int c = 0; c < a.Shape[1]; c++)
                for (int h = 0; h < a.Shape[2]; h++)
                    for (int w = 0; w < a.Shape[3]; w++)
                        result[[i, c, h, w]] = a[[i, c, h, w]];

        // Copy b
        for (int i = 0; i < b.Shape[0]; i++)
            for (int c = 0; c < b.Shape[1]; c++)
                for (int h = 0; h < Math.Min(a.Shape[2], b.Shape[2]); h++)
                    for (int w = 0; w < Math.Min(a.Shape[3], b.Shape[3]); w++)
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
        // Read configuration values for validation
        _ = reader.ReadBoolean(); // IsOnnxMode
        _ = reader.ReadInt32();   // SampleRate
        _ = reader.ReadInt32();   // _numStages
        _ = reader.ReadInt32();   // _baseChannels
        _ = reader.ReadInt32();   // _lstmHiddenDim
        _ = reader.ReadInt32();   // _numLstmLayers
        _ = reader.ReadInt32();   // _fftSize
        _ = reader.ReadInt32();   // _hopSize
        _ = reader.ReadBoolean(); // _useComplexMask
        _ = reader.ReadInt32();   // _kernelSize
        _ = reader.ReadInt32();   // _stride
        EnhancementStrength = reader.ReadDouble();
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
