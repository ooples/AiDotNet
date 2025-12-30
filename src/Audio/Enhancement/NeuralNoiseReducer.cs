using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Onnx;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Audio.Enhancement;

/// <summary>
/// Neural network-based noise reducer for high-quality audio enhancement.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// This model uses an encoder-bottleneck-decoder architecture inspired by U-Net
/// to learn the mapping from noisy audio to clean audio. It operates in the
/// time-frequency domain using STFT for analysis and synthesis.
/// Note: The current implementation uses a simplified dense decoder; a full U-Net
/// with transposed convolutions and skip connections is planned for future versions.
/// </para>
/// <para><b>For Beginners:</b> This is like a "magic eraser" for audio noise!
///
/// How it works:
/// 1. Converts audio to a spectrogram (picture of sound)
/// 2. Neural network learns to identify and remove noise patterns
/// 3. Converts cleaned spectrogram back to audio
///
/// Key features:
/// - Works on any type of noise (AC hum, fan noise, traffic, etc.)
/// - Preserves speech/music quality while removing noise
/// - Can be trained on your specific noise conditions
/// - Supports real-time streaming for live applications
///
/// Use cases:
/// - Podcast/video production (remove background noise)
/// - Voice calls (improve speech clarity)
/// - Music restoration (remove hiss/crackle from old recordings)
/// - Hearing aids (enhance speech in noisy environments)
///
/// Two modes of operation:
/// 1. ONNX Mode: Load a pre-trained model for instant use
/// 2. Native Mode: Train your own model on custom data
/// </para>
/// </remarks>
public class NeuralNoiseReducer<T> : AudioNeuralNetworkBase<T>, IAudioEnhancer<T>
{
    #region Execution Mode

    /// <summary>
    /// Indicates whether to use native training mode.
    /// </summary>
    private bool _useNativeMode;

    /// <summary>
    /// Path to ONNX model (ONNX mode only).
    /// </summary>
    private string? _modelPath;

    #endregion

    #region ONNX Mode Fields

    // ONNX models are stored in base class: OnnxEncoder, OnnxDecoder, OnnxModel

    #endregion

    #region Native Mode Fields

    /// <summary>
    /// Encoder layers (downsampling path).
    /// </summary>
    private List<ILayer<T>> _encoderLayers = new();

    /// <summary>
    /// Decoder layers (upsampling path).
    /// </summary>
    private List<ILayer<T>> _decoderLayers = new();

    /// <summary>
    /// Bottleneck layer.
    /// </summary>
    private ILayer<T>? _bottleneckLayer;

    /// <summary>
    /// Output projection layer.
    /// </summary>
    private ILayer<T>? _outputLayer;

    /// <summary>
    /// Loss function for training.
    /// </summary>
    private ILossFunction<T> _lossFunction;

    #endregion

    #region Configuration

    /// <summary>
    /// FFT size for STFT analysis (non-readonly for deserialization support).
    /// </summary>
    private int _fftSize;

    /// <summary>
    /// Hop size between STFT frames (non-readonly for deserialization support).
    /// </summary>
    private int _hopSize;

    /// <summary>
    /// Number of encoder/decoder stages (non-readonly for deserialization support).
    /// </summary>
    private int _numStages;

    /// <summary>
    /// Base number of filters (doubled at each stage) (non-readonly for deserialization support).
    /// </summary>
    private int _baseFilters;

    /// <summary>
    /// Hidden dimension in bottleneck (non-readonly for deserialization support).
    /// </summary>
    private int _bottleneckDim;

    #endregion

    #region IAudioEnhancer Properties

    /// <inheritdoc/>
    public int NumChannels { get; protected set; }

    /// <inheritdoc/>
    public double EnhancementStrength { get; set; }

    /// <inheritdoc/>
    public int LatencySamples => _fftSize;

    #endregion

    #region Streaming State

    /// <summary>
    /// Input buffer for streaming mode.
    /// </summary>
    private T[]? _inputBuffer;

    /// <summary>
    /// Output buffer for overlap-add.
    /// </summary>
    private T[]? _outputBuffer;

    /// <summary>
    /// Current position in input buffer.
    /// </summary>
    private int _bufferPosition;

    /// <summary>
    /// Window function for STFT.
    /// </summary>
    private T[]? _window;

    /// <summary>
    /// Noise profile estimate (optional).
    /// </summary>
    private T[]? _noiseProfile;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a NeuralNoiseReducer in ONNX inference mode using a pre-trained model.
    /// </summary>
    /// <param name="architecture">The neural network architecture (user-defined for full customization).</param>
    /// <param name="modelPath">Path to the ONNX model file.</param>
    /// <param name="sampleRate">Audio sample rate (default: 16000).</param>
    /// <param name="fftSize">FFT window size (default: 512).</param>
    /// <param name="hopSize">Hop size between frames (default: 256).</param>
    /// <param name="numChannels">Number of audio channels (default: 1).</param>
    /// <param name="enhancementStrength">Enhancement strength 0-1 (default: 0.8).</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this constructor when you have a pre-trained model.
    /// Just point to the ONNX file and start enhancing audio immediately.
    /// </para>
    /// </remarks>
    public NeuralNoiseReducer(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        int sampleRate = 16000,
        int fftSize = 512,
        int hopSize = 256,
        int numChannels = 1,
        double enhancementStrength = 0.8)
        : base(architecture, new MeanAbsoluteErrorLoss<T>())
    {
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));

        _useNativeMode = false;
        _modelPath = modelPath;
        _lossFunction = new MeanAbsoluteErrorLoss<T>();

        SampleRate = sampleRate;
        NumChannels = numChannels;
        _fftSize = fftSize;
        _hopSize = hopSize;
        _numStages = 4;
        _baseFilters = 32;
        _bottleneckDim = 256;
        EnhancementStrength = enhancementStrength;

        // Load ONNX model
        OnnxModel = new OnnxModel<T>(modelPath);

        InitializeStreamingBuffers();
    }

    /// <summary>
    /// Creates a NeuralNoiseReducer in native training mode.
    /// </summary>
    /// <param name="architecture">The neural network architecture (user-defined for full customization).</param>
    /// <param name="sampleRate">Audio sample rate (default: 16000).</param>
    /// <param name="fftSize">FFT window size (default: 512).</param>
    /// <param name="hopSize">Hop size between frames (default: 256).</param>
    /// <param name="numChannels">Number of audio channels (default: 1).</param>
    /// <param name="numStages">Number of encoder/decoder stages (default: 4).</param>
    /// <param name="baseFilters">Base filter count (default: 32).</param>
    /// <param name="bottleneckDim">Bottleneck hidden dimension (default: 256).</param>
    /// <param name="enhancementStrength">Enhancement strength 0-1 (default: 0.8).</param>
    /// <param name="lossFunction">Loss function for training (default: L1 loss).</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this constructor when you want to train your own model.
    /// You'll need noisy/clean audio pairs for training.
    /// </para>
    /// </remarks>
    public NeuralNoiseReducer(
        NeuralNetworkArchitecture<T> architecture,
        int sampleRate = 16000,
        int fftSize = 512,
        int hopSize = 256,
        int numChannels = 1,
        int numStages = 4,
        int baseFilters = 32,
        int bottleneckDim = 256,
        double enhancementStrength = 0.8,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanAbsoluteErrorLoss<T>())
    {
        _useNativeMode = true;
        _modelPath = null;
        _lossFunction = lossFunction ?? new MeanAbsoluteErrorLoss<T>();

        SampleRate = sampleRate;
        NumChannels = numChannels;
        _fftSize = fftSize;
        _hopSize = hopSize;
        _numStages = numStages;
        _baseFilters = baseFilters;
        _bottleneckDim = bottleneckDim;
        EnhancementStrength = enhancementStrength;

        InitializeLayers();
        InitializeStreamingBuffers();
    }

    private void InitializeStreamingBuffers()
    {
        _inputBuffer = new T[_fftSize];
        _outputBuffer = new T[_fftSize];
        _bufferPosition = 0;
        _window = CreateHannWindow(_fftSize);
    }

    private T[] CreateHannWindow(int size)
    {
        var window = new T[size];
        for (int i = 0; i < size; i++)
        {
            var value = 0.5 * (1 - Math.Cos(2 * Math.PI * i / (size - 1)));
            window[i] = NumOps.FromDouble(value);
        }
        return window;
    }

    #endregion

    #region Layer Initialization

    /// <inheritdoc/>
    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;

        int freqBins = _fftSize / 2 + 1;

        // Build U-Net style encoder-decoder architecture
        // Encoder path: progressively downsample and increase channels
        int currentFilters = _baseFilters;
        int currentFreqDim = freqBins;

        for (int stage = 0; stage < _numStages; stage++)
        {
            int inputFilters = stage == 0 ? 1 : currentFilters / 2;

            // Conv layer with stride 2 for downsampling
            var convLayer = new ConvolutionalLayer<T>(
                inputDepth: inputFilters,
                inputHeight: currentFreqDim,
                inputWidth: 1,
                outputDepth: currentFilters,
                kernelSize: 4,
                stride: 2,
                padding: 1,
                activationFunction: new LeakyReLUActivation<T>());

            _encoderLayers.Add(convLayer);
            Layers.Add(convLayer);

            currentFreqDim = (currentFreqDim + 2 - 4) / 2 + 1;
            if (stage < _numStages - 1)
                currentFilters *= 2;
        }

        // Bottleneck: dense layer for learning global context
        int bottleneckInputSize = currentFilters * currentFreqDim;
        _bottleneckLayer = new DenseLayer<T>(
            inputSize: bottleneckInputSize,
            outputSize: _bottleneckDim,
            activationFunction: new ReLUActivation<T>());
        Layers.Add(_bottleneckLayer);

        // Decoder path: direct bottleneck-to-output projection
        // TODO: Implement proper U-Net decoder with:
        //   - Transposed convolutions or upsampling layers
        //   - Skip connections from encoder stages
        //   - Progressive channel reduction mirroring encoder
        //   - Proper frequency resolution restoration
        // Current implementation uses a simplified dense decoder that works
        // but has reduced capacity compared to a true U-Net architecture.
        int decoderInputSize = _bottleneckDim;
        int targetFreqDim = freqBins;

        // Final output projection (simplified decoder)
        _outputLayer = new DenseLayer<T>(
            inputSize: decoderInputSize,
            outputSize: targetFreqDim,
            activationFunction: new SigmoidActivation<T>()); // Mask output 0-1
        Layers.Add(_outputLayer);
    }

    #endregion

    #region IAudioEnhancer Implementation

    /// <inheritdoc/>
    public Tensor<T> Enhance(Tensor<T> audio)
    {
        var samples = audio.ToVector().ToArray();
        var enhanced = ProcessOverlapAdd(samples);

        var result = new Tensor<T>([enhanced.Length]);
        var resultVector = result.ToVector();
        for (int i = 0; i < enhanced.Length; i++)
        {
            resultVector[i] = enhanced[i];
        }
        return result;
    }

    /// <inheritdoc/>
    public Tensor<T> EnhanceWithReference(Tensor<T> audio, Tensor<T> reference)
    {
        // For echo cancellation, combine reference info with enhancement
        // Simplified: just enhance the audio
        return Enhance(audio);
    }

    /// <inheritdoc/>
    public Tensor<T> ProcessChunk(Tensor<T> audioChunk)
    {
        var samples = audioChunk.ToVector().ToArray();
        var enhanced = ProcessStreamingChunk(samples);

        var result = new Tensor<T>([enhanced.Length]);
        var resultVector = result.ToVector();
        for (int i = 0; i < enhanced.Length; i++)
        {
            resultVector[i] = enhanced[i];
        }
        return result;
    }

    /// <summary>
    /// Resets streaming state (explicit interface implementation to avoid conflict with base).
    /// </summary>
    void IAudioEnhancer<T>.ResetState()
    {
        ResetEnhancerState();
    }

    /// <summary>
    /// Resets the enhancer streaming state.
    /// </summary>
    public void ResetEnhancerState()
    {
        _inputBuffer = new T[_fftSize];
        _outputBuffer = new T[_fftSize];
        _bufferPosition = 0;
    }

    /// <inheritdoc/>
    public void EstimateNoiseProfile(Tensor<T> noiseOnlyAudio)
    {
        var samples = noiseOnlyAudio.ToVector().ToArray();
        _noiseProfile = EstimateNoiseSpectrum(samples);
    }

    #endregion

    #region Audio Processing

    /// <summary>
    /// Processes audio using overlap-add STFT method.
    /// </summary>
    private T[] ProcessOverlapAdd(T[] input)
    {
        if (_window is null)
            InitializeStreamingBuffers();

        int numFrames = (input.Length - _fftSize) / _hopSize + 1;
        var output = new T[input.Length];

        for (int frame = 0; frame < numFrames; frame++)
        {
            int start = frame * _hopSize;

            // Extract and window frame
            var frameData = new T[_fftSize];
            for (int i = 0; i < _fftSize && start + i < input.Length; i++)
            {
                frameData[i] = NumOps.Multiply(input[start + i], _window![i]);
            }

            // Compute STFT
            var (magnitudes, phases) = ComputeSTFT(frameData);

            // Apply neural network enhancement
            var enhancedMagnitudes = EnhanceSpectrum(magnitudes);

            // Compute inverse STFT
            var enhanced = ComputeISTFT(enhancedMagnitudes, phases);

            // Window and overlap-add
            for (int i = 0; i < _fftSize && start + i < output.Length; i++)
            {
                var windowed = NumOps.Multiply(enhanced[i], _window![i]);
                output[start + i] = NumOps.Add(output[start + i], windowed);
            }
        }

        return output;
    }

    /// <summary>
    /// Processes audio in streaming mode.
    /// </summary>
    private T[] ProcessStreamingChunk(T[] chunk)
    {
        if (_inputBuffer is null || _outputBuffer is null)
            InitializeStreamingBuffers();

        var output = new T[chunk.Length];
        int outputPos = 0;

        for (int i = 0; i < chunk.Length; i++)
        {
            _inputBuffer![_bufferPosition] = chunk[i];
            _bufferPosition++;

            if (_bufferPosition >= _hopSize)
            {
                // Process frame
                var frameData = new T[_fftSize];
                for (int j = 0; j < _fftSize; j++)
                {
                    int idx = (j + _bufferPosition - _hopSize) % _fftSize;
                    frameData[j] = NumOps.Multiply(_inputBuffer[idx], _window![j]);
                }

                var (magnitudes, phases) = ComputeSTFT(frameData);
                var enhancedMagnitudes = EnhanceSpectrum(magnitudes);
                var enhanced = ComputeISTFT(enhancedMagnitudes, phases);

                // Overlap-add to output buffer
                for (int j = 0; j < _fftSize; j++)
                {
                    var windowed = NumOps.Multiply(enhanced[j], _window![j]);
                    _outputBuffer![j] = NumOps.Add(_outputBuffer[j], windowed);
                }

                // Output hop samples
                for (int j = 0; j < _hopSize && outputPos < output.Length; j++)
                {
                    output[outputPos++] = _outputBuffer![j];
                }

                // Shift output buffer
                Array.Copy(_outputBuffer!, _hopSize, _outputBuffer!, 0, _fftSize - _hopSize);
                Array.Clear(_outputBuffer!, _fftSize - _hopSize, _hopSize);

                _bufferPosition = 0;
            }
        }

        return output;
    }

    /// <summary>
    /// Applies neural network enhancement to spectrum.
    /// </summary>
    private T[] EnhanceSpectrum(T[] magnitudes)
    {
        // Create input tensor from magnitudes
        var input = new Tensor<T>([1, magnitudes.Length, 1]);
        var inputVector = input.ToVector();
        for (int i = 0; i < magnitudes.Length; i++)
        {
            inputVector[i] = magnitudes[i];
        }

        Tensor<T> mask;

        if (IsOnnxMode)
        {
            mask = RunOnnxInference(input);
        }
        else
        {
            mask = Forward(input);
        }

        // Apply mask to magnitudes with enhancement strength
        var enhanced = new T[magnitudes.Length];
        var maskVector = mask.ToVector().ToArray();

        for (int i = 0; i < magnitudes.Length; i++)
        {
            // mask is in range 0-1, where 1 = keep, 0 = remove
            double mag = NumOps.ToDouble(magnitudes[i]);
            double m = i < maskVector.Length ? NumOps.ToDouble(maskVector[i]) : 1.0;

            // Apply enhancement strength to control how aggressively we use the mask
            m = 1.0 - EnhancementStrength * (1.0 - m);

            enhanced[i] = NumOps.FromDouble(mag * m);
        }

        return enhanced;
    }

    #endregion

    #region STFT Implementation

    /// <summary>
    /// Computes Short-Time Fourier Transform using FftSharp library (O(N log N) algorithm).
    /// </summary>
    private (T[] Magnitudes, T[] Phases) ComputeSTFT(T[] frame)
    {
        // Convert frame to double array for FFT
        var frameData = new double[_fftSize];
        for (int i = 0; i < Math.Min(frame.Length, _fftSize); i++)
        {
            frameData[i] = NumOps.ToDouble(frame[i]);
        }

        // Compute FFT using FftSharp (proper O(N log N) algorithm)
        System.Numerics.Complex[] spectrum = FftSharp.FFT.Forward(frameData);

        // Extract magnitude and phase for positive frequencies (N/2+1 bins)
        int numBins = _fftSize / 2 + 1;
        var magnitudes = new T[numBins];
        var phases = new T[numBins];

        for (int k = 0; k < numBins; k++)
        {
            double real = spectrum[k].Real;
            double imag = spectrum[k].Imaginary;
            magnitudes[k] = NumOps.FromDouble(Math.Sqrt(real * real + imag * imag));
            phases[k] = NumOps.FromDouble(Math.Atan2(imag, real));
        }

        return (magnitudes, phases);
    }

    /// <summary>
    /// Computes Inverse Short-Time Fourier Transform using FftSharp library.
    /// </summary>
    private T[] ComputeISTFT(T[] magnitudes, T[] phases)
    {
        // Reconstruct full complex spectrum from magnitude/phase
        // magnitudes/phases contain N/2+1 positive frequency bins
        var spectrum = new System.Numerics.Complex[_fftSize];

        // Fill positive frequencies
        for (int k = 0; k < magnitudes.Length; k++)
        {
            double mag = NumOps.ToDouble(magnitudes[k]);
            double phase = NumOps.ToDouble(phases[k]);
            spectrum[k] = System.Numerics.Complex.FromPolarCoordinates(mag, phase);
        }

        // Fill negative frequencies with conjugate symmetry
        // For real signals: X[N-k] = conj(X[k])
        for (int k = 1; k < magnitudes.Length - 1; k++)
        {
            spectrum[_fftSize - k] = System.Numerics.Complex.Conjugate(spectrum[k]);
        }

        // Compute inverse FFT using FftSharp (modifies spectrum in-place)
        FftSharp.FFT.Inverse(spectrum);

        // Extract real part
        var output = new T[_fftSize];
        for (int n = 0; n < _fftSize; n++)
        {
            output[n] = NumOps.FromDouble(spectrum[n].Real);
        }

        return output;
    }

    /// <summary>
    /// Estimates noise spectrum from noise-only audio.
    /// </summary>
    private T[] EstimateNoiseSpectrum(T[] noiseAudio)
    {
        int numFrames = (noiseAudio.Length - _fftSize) / _hopSize + 1;
        int numBins = _fftSize / 2 + 1;
        var avgMagnitudes = new T[numBins];

        for (int frame = 0; frame < numFrames; frame++)
        {
            int start = frame * _hopSize;
            var frameData = new T[_fftSize];

            for (int i = 0; i < _fftSize && start + i < noiseAudio.Length; i++)
            {
                frameData[i] = NumOps.Multiply(noiseAudio[start + i], _window![i]);
            }

            var (magnitudes, _) = ComputeSTFT(frameData);

            for (int i = 0; i < numBins; i++)
            {
                avgMagnitudes[i] = NumOps.Add(avgMagnitudes[i], magnitudes[i]);
            }
        }

        // Average
        var divisor = NumOps.FromDouble(numFrames);
        for (int i = 0; i < numBins; i++)
        {
            avgMagnitudes[i] = NumOps.Divide(avgMagnitudes[i], divisor);
        }

        return avgMagnitudes;
    }

    #endregion

    #region AudioNeuralNetworkBase Implementation

    /// <inheritdoc/>
    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio)
    {
        // Convert to spectrogram for neural network input
        var samples = rawAudio.ToVector().ToArray();

        // Apply window
        var windowed = new T[Math.Min(samples.Length, _fftSize)];
        for (int i = 0; i < windowed.Length; i++)
        {
            windowed[i] = NumOps.Multiply(samples[i], _window![i]);
        }

        // Compute magnitude spectrum
        var (magnitudes, _) = ComputeSTFT(windowed);

        // Reshape to [1, freq_bins, 1]
        var result = new Tensor<T>([1, magnitudes.Length, 1]);
        var resultVector = result.ToVector();
        for (int i = 0; i < magnitudes.Length; i++)
        {
            resultVector[i] = magnitudes[i];
        }

        return result;
    }

    /// <inheritdoc/>
    protected override Tensor<T> PostprocessOutput(Tensor<T> modelOutput)
    {
        // Model outputs a mask in range 0-1
        return modelOutput;
    }

    #endregion

    #region Training

    /// <inheritdoc/>
    public override void Train(Tensor<T> input, Tensor<T> expected)
    {
        if (IsOnnxMode)
            throw new InvalidOperationException("Training not supported in ONNX mode.");

        SetTrainingMode(true);

        // Forward pass through all layers
        var layerOutputs = new List<Tensor<T>> { input };
        var current = input;

        foreach (var layer in Layers)
        {
            current = layer.Forward(current);
            layerOutputs.Add(current);
        }

        // Calculate loss
        var loss = _lossFunction.CalculateLoss(current.ToVector(), expected.ToVector());

        // Calculate gradients
        var gradient = _lossFunction.CalculateDerivative(current.ToVector(), expected.ToVector());

        // Backward pass through all layers in reverse
        var gradTensor = new Tensor<T>([gradient.Length]);
        var gradVector = gradTensor.ToVector();
        for (int i = 0; i < gradient.Length; i++)
        {
            gradVector[i] = gradient[i];
        }

        for (int i = Layers.Count - 1; i >= 0; i--)
        {
            gradTensor = Layers[i].Backward(gradTensor);
        }

        SetTrainingMode(false);
    }

    /// <inheritdoc/>
    public override void UpdateParameters(Vector<T> parameters)
    {
        if (IsOnnxMode)
        {
            throw new InvalidOperationException("Cannot update parameters in ONNX mode.");
        }

        int offset = 0;

        // Update encoder layers
        foreach (var layer in _encoderLayers)
        {
            var layerParams = layer.GetParameters();
            int layerParamCount = layerParams.Length;

            if (offset + layerParamCount <= parameters.Length)
            {
                var newParams = new Vector<T>(layerParamCount);
                for (int i = 0; i < layerParamCount; i++)
                {
                    newParams[i] = parameters[offset + i];
                }
                layer.SetParameters(newParams);
                offset += layerParamCount;
            }
        }

        // Update bottleneck layer
        if (_bottleneckLayer is not null)
        {
            var layerParams = _bottleneckLayer.GetParameters();
            int layerParamCount = layerParams.Length;

            if (offset + layerParamCount <= parameters.Length)
            {
                var newParams = new Vector<T>(layerParamCount);
                for (int i = 0; i < layerParamCount; i++)
                {
                    newParams[i] = parameters[offset + i];
                }
                _bottleneckLayer.SetParameters(newParams);
                offset += layerParamCount;
            }
        }

        // Update decoder layers
        foreach (var layer in _decoderLayers)
        {
            var layerParams = layer.GetParameters();
            int layerParamCount = layerParams.Length;

            if (offset + layerParamCount <= parameters.Length)
            {
                var newParams = new Vector<T>(layerParamCount);
                for (int i = 0; i < layerParamCount; i++)
                {
                    newParams[i] = parameters[offset + i];
                }
                layer.SetParameters(newParams);
                offset += layerParamCount;
            }
        }

        // Update output layer
        if (_outputLayer is not null)
        {
            var layerParams = _outputLayer.GetParameters();
            int layerParamCount = layerParams.Length;

            if (offset + layerParamCount <= parameters.Length)
            {
                var newParams = new Vector<T>(layerParamCount);
                for (int i = 0; i < layerParamCount; i++)
                {
                    newParams[i] = parameters[offset + i];
                }
                _outputLayer.SetParameters(newParams);
            }
        }
    }

    /// <inheritdoc/>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        var preprocessed = PreprocessAudio(input);
        Tensor<T> output;

        if (IsOnnxMode)
        {
            output = RunOnnxInference(preprocessed);
        }
        else
        {
            output = Forward(preprocessed);
        }

        return PostprocessOutput(output);
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new NeuralNoiseReducer<T>(
            Architecture,
            SampleRate,
            _fftSize,
            _hopSize,
            NumChannels,
            _numStages,
            _baseFilters,
            _bottleneckDim,
            EnhancementStrength,
            _lossFunction);
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = "NeuralNoiseReducer",
            Version = "1.0",
            ModelType = ModelType.NeuralNetwork,
            Description = "Neural network-based audio noise reducer with U-Net architecture"
        };

        metadata.SetProperty("SampleRate", SampleRate);
        metadata.SetProperty("FFTSize", _fftSize);
        metadata.SetProperty("HopSize", _hopSize);
        metadata.SetProperty("NumStages", _numStages);
        metadata.SetProperty("BaseFilters", _baseFilters);
        metadata.SetProperty("BottleneckDim", _bottleneckDim);
        metadata.SetProperty("EnhancementStrength", EnhancementStrength);
        metadata.SetProperty("UseNativeMode", _useNativeMode);

        return metadata;
    }

    /// <inheritdoc/>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_useNativeMode);
        writer.Write(SampleRate);
        writer.Write(_fftSize);
        writer.Write(_hopSize);
        writer.Write(NumChannels);
        writer.Write(_numStages);
        writer.Write(_baseFilters);
        writer.Write(_bottleneckDim);
        writer.Write(EnhancementStrength);
    }

    /// <inheritdoc/>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _useNativeMode = reader.ReadBoolean();
        SampleRate = reader.ReadInt32();
        _fftSize = reader.ReadInt32();
        _hopSize = reader.ReadInt32();
        NumChannels = reader.ReadInt32();
        _numStages = reader.ReadInt32();
        _baseFilters = reader.ReadInt32();
        _bottleneckDim = reader.ReadInt32();
        EnhancementStrength = reader.ReadDouble();

        // Reinitialize layers if needed for native mode
        if (_useNativeMode && (_encoderLayers is null || _encoderLayers.Count == 0))
        {
            InitializeLayers();
        }
    }

    #endregion
}
