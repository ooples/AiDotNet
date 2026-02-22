using AiDotNet.Diffusion;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Onnx;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Audio.VoiceActivity;

/// <summary>
/// Silero Voice Activity Detection model - high accuracy neural network VAD.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Silero VAD is a state-of-the-art voice activity detector that uses a lightweight
/// neural network architecture to achieve high accuracy with low latency. It can:
/// <list type="bullet">
/// <item>Detect speech with very high accuracy (better than energy-based methods)</item>
/// <item>Handle noisy environments well</item>
/// <item>Run in real-time on CPU</item>
/// <item>Work across multiple languages</item>
/// </list>
/// </para>
/// <para><b>For Beginners:</b> Silero VAD tells you when someone is speaking vs silence.
/// Unlike simple energy-based VAD, it uses a neural network that has learned what
/// speech "looks like" from millions of examples.
///
/// Why use neural network VAD?
/// - Much more accurate than energy/threshold-based methods
/// - Handles background noise better (music, crowd noise, etc.)
/// - Detects speech even when quiet
/// - Doesn't false-trigger on non-speech sounds
///
/// Two ways to use this class:
/// 1. ONNX Mode: Load pretrained Silero model for fast inference
/// 2. Native Mode: Train your own VAD model from scratch
///
/// ONNX Mode Example (recommended):
/// <code>
/// var vad = new SileroVad&lt;float&gt;(
///     architecture,
///     modelPath: "path/to/silero_vad.onnx");
/// var (isSpeech, probability) = vad.ProcessChunk(audioFrame);
/// if (isSpeech)
///     Console.WriteLine($"Speech detected! Confidence: {probability:P0}");
/// </code>
///
/// Training Mode Example:
/// <code>
/// var vad = new SileroVad&lt;float&gt;(architecture);
/// for (int epoch = 0; epoch &lt; 100; epoch++)
/// {
///     foreach (var (audio, labels) in trainingData)
///     {
///         vad.Train(audio, labels);
///     }
/// }
/// </code>
/// </para>
/// </remarks>
public class SileroVad<T> : AudioNeuralNetworkBase<T>, IVoiceActivityDetector<T>
{
    private readonly SileroVadOptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    #region Execution Mode

    /// <summary>
    /// Indicates whether this network uses native layers (true) or ONNX model (false).
    /// </summary>
    private readonly bool _useNativeMode;

    #endregion

    #region ONNX Mode Fields

    /// <summary>
    /// Path to the ONNX model file.
    /// </summary>
    private readonly string? _modelPath;

    #endregion

    #region Native Mode Fields

    /// <summary>
    /// Convolutional feature extraction layers.
    /// </summary>
    private readonly List<ILayer<T>> _convLayers = [];

    /// <summary>
    /// LSTM layers for temporal modeling.
    /// </summary>
    private readonly List<ILayer<T>> _lstmLayers = [];

    /// <summary>
    /// Output classification layer.
    /// </summary>
    private ILayer<T>? _outputLayer;

    #endregion

    #region Configuration

    /// <summary>
    /// Loss function for training.
    /// </summary>
    private readonly ILossFunction<T> _lossFunction;

    /// <summary>
    /// Detection threshold (0-1).
    /// </summary>
    private readonly double _threshold;

    /// <summary>
    /// Frame size in samples.
    /// </summary>
    private readonly int _frameSize;

    /// <summary>
    /// Minimum speech duration in milliseconds.
    /// </summary>
    private readonly int _minSpeechDurationMs;

    /// <summary>
    /// Minimum silence duration in milliseconds.
    /// </summary>
    private readonly int _minSilenceDurationMs;

    /// <summary>
    /// Number of convolutional filters.
    /// </summary>
    private readonly int _convFilters;

    /// <summary>
    /// LSTM hidden dimension.
    /// </summary>
    private readonly int _lstmHiddenDim;

    /// <summary>
    /// Number of LSTM layers.
    /// </summary>
    private readonly int _numLstmLayers;

    #endregion

    #region Streaming State

    /// <summary>
    /// Number of consecutive speech frames.
    /// </summary>
    private int _speechFrameCount;

    /// <summary>
    /// Number of consecutive silence frames.
    /// </summary>
    private int _silenceFrameCount;

    /// <summary>
    /// Current speech state.
    /// </summary>
    private bool _inSpeech;

    #endregion

    /// <summary>
    /// Disposed flag.
    /// </summary>
    private bool _disposed;

    #region IVoiceActivityDetector Properties

    /// <inheritdoc/>
    public int FrameSize => _frameSize;

    /// <inheritdoc/>
    public double Threshold
    {
        get => _threshold;
        set { }  // Threshold is readonly for Silero VAD
    }

    /// <inheritdoc/>
    public int MinSpeechDurationMs
    {
        get => _minSpeechDurationMs;
        set { }  // Read-only
    }

    /// <inheritdoc/>
    public int MinSilenceDurationMs
    {
        get => _minSilenceDurationMs;
        set { }  // Read-only
    }

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a Silero VAD in ONNX inference mode with a pretrained model.
    /// </summary>
    /// <param name="architecture">The neural network architecture.</param>
    /// <param name="modelPath">Path to the Silero VAD ONNX model.</param>
    /// <param name="sampleRate">Expected sample rate (default: 16000 Hz).</param>
    /// <param name="frameSize">Frame size in samples (default: 512).</param>
    /// <param name="threshold">Detection threshold 0-1 (default: 0.5).</param>
    /// <param name="minSpeechDurationMs">Minimum speech duration in ms (default: 250).</param>
    /// <param name="minSilenceDurationMs">Minimum silence duration in ms (default: 100).</param>
    public SileroVad(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        int sampleRate = 16000,
        int frameSize = 512,
        double threshold = 0.5,
        int minSpeechDurationMs = 250,
        int minSilenceDurationMs = 100,
        SileroVadOptions? options = null)
        : base(architecture, new BinaryCrossEntropyLoss<T>())
    {
        _options = options ?? new SileroVadOptions();
        Options = _options;
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("Model path cannot be null or empty.", nameof(modelPath));

        _useNativeMode = false;
        _modelPath = modelPath;
        _lossFunction = new BinaryCrossEntropyLoss<T>();

        SampleRate = sampleRate;
        _frameSize = frameSize;
        _threshold = threshold;
        _minSpeechDurationMs = minSpeechDurationMs;
        _minSilenceDurationMs = minSilenceDurationMs;

        // Default architecture parameters (not used in ONNX mode)
        _convFilters = 64;
        _lstmHiddenDim = 64;
        _numLstmLayers = 2;

        // Load ONNX model
        OnnxModel = new OnnxModel<T>(modelPath);

        ResetVadState();
    }

    /// <summary>
    /// Creates a Silero VAD in native training mode for training from scratch.
    /// </summary>
    /// <param name="architecture">The neural network architecture.</param>
    /// <param name="sampleRate">Expected sample rate (default: 16000 Hz).</param>
    /// <param name="frameSize">Frame size in samples (default: 512).</param>
    /// <param name="threshold">Detection threshold 0-1 (default: 0.5).</param>
    /// <param name="minSpeechDurationMs">Minimum speech duration in ms (default: 250).</param>
    /// <param name="minSilenceDurationMs">Minimum silence duration in ms (default: 100).</param>
    /// <param name="convFilters">Number of convolutional filters (default: 64).</param>
    /// <param name="lstmHiddenDim">LSTM hidden dimension (default: 64).</param>
    /// <param name="numLstmLayers">Number of LSTM layers (default: 2).</param>
    public SileroVad(
        NeuralNetworkArchitecture<T> architecture,
        int sampleRate = 16000,
        int frameSize = 512,
        double threshold = 0.5,
        int minSpeechDurationMs = 250,
        int minSilenceDurationMs = 100,
        int convFilters = 64,
        int lstmHiddenDim = 64,
        int numLstmLayers = 2,
        SileroVadOptions? options = null)
        : base(architecture, new BinaryCrossEntropyLoss<T>())
    {
        _options = options ?? new SileroVadOptions();
        Options = _options;
        _useNativeMode = true;
        _modelPath = null;
        _lossFunction = new BinaryCrossEntropyLoss<T>();

        SampleRate = sampleRate;
        _frameSize = frameSize;
        _threshold = threshold;
        _minSpeechDurationMs = minSpeechDurationMs;
        _minSilenceDurationMs = minSilenceDurationMs;

        _convFilters = convFilters;
        _lstmHiddenDim = lstmHiddenDim;
        _numLstmLayers = numLstmLayers;

        InitializeLayers();
        ResetVadState();
    }

    #endregion

    #region Layer Initialization

    /// <inheritdoc/>
    protected override void InitializeLayers()
    {
        if (!_useNativeMode)
            return;

        // Silero VAD architecture:
        // 1. Conv feature extraction (2D conv with 1x1 height to simulate 1D)
        // 2. LSTM layers for temporal modeling
        // 3. Dense output layer with sigmoid for probability

        // Calculate expected sequence length after convolutions
        // Using stride 4, 2, 2 on frameSize samples
        int seqLen1 = (_frameSize + 2 * 2 - 8) / 4 + 1;  // After first conv
        int seqLen2 = (seqLen1 + 2 * 1 - 4) / 2 + 1;     // After second conv
        int seqLen3 = (seqLen2 + 2 * 1 - 4) / 2 + 1;     // After third conv

        // Convolutional feature extraction using 2D conv (height=1 simulates 1D)
        // First conv layer: raw audio -> features
        // ConvolutionalLayer(inputDepth, inputHeight, inputWidth, outputDepth, kernelSize, stride, padding, activation)
        _convLayers.Add(new ConvolutionalLayer<T>(
            inputDepth: 1,
            inputHeight: 1,
            inputWidth: _frameSize,
            outputDepth: _convFilters,
            kernelSize: 8,
            stride: 4,
            padding: 2,
            activationFunction: new LeakyReLUActivation<T>()));

        // Second conv layer
        _convLayers.Add(new ConvolutionalLayer<T>(
            inputDepth: _convFilters,
            inputHeight: 1,
            inputWidth: seqLen1,
            outputDepth: _convFilters,
            kernelSize: 4,
            stride: 2,
            padding: 1,
            activationFunction: new LeakyReLUActivation<T>()));

        // Third conv layer
        _convLayers.Add(new ConvolutionalLayer<T>(
            inputDepth: _convFilters,
            inputHeight: 1,
            inputWidth: seqLen2,
            outputDepth: _convFilters,
            kernelSize: 4,
            stride: 2,
            padding: 1,
            activationFunction: new LeakyReLUActivation<T>()));

        // LSTM layers for temporal modeling
        for (int i = 0; i < _numLstmLayers; i++)
        {
            int inputSize = i == 0 ? _convFilters : _lstmHiddenDim;
            // Explicitly specify IActivationFunction<T> to disambiguate constructor
            _lstmLayers.Add(new LSTMLayer<T>(
                inputSize: inputSize,
                hiddenSize: _lstmHiddenDim,
                inputShape: [1, seqLen3, inputSize],
                activation: (IActivationFunction<T>?)null,
                recurrentActivation: (IActivationFunction<T>?)null));
        }

        // Output layer: probability of speech
        _outputLayer = new DenseLayer<T>(
            inputSize: _lstmHiddenDim,
            outputSize: 1,
            activationFunction: new SigmoidActivation<T>());

        // Add all layers to base class Layers collection
        foreach (var layer in _convLayers)
            Layers.Add(layer);
        foreach (var layer in _lstmLayers)
            Layers.Add(layer);
        if (_outputLayer is not null)
            Layers.Add(_outputLayer);
    }

    #endregion

    #region IVoiceActivityDetector Implementation

    /// <inheritdoc/>
    public bool DetectSpeech(Tensor<T> audioFrame)
    {
        var prob = GetSpeechProbability(audioFrame);
        return NumOps.ToDouble(prob) >= _threshold;
    }

    /// <inheritdoc/>
    public T GetSpeechProbability(Tensor<T> audioFrame)
    {
        var preprocessed = PreprocessAudio(audioFrame);
        Tensor<T> output;

        if (IsOnnxMode)
        {
            output = RunOnnxInference(preprocessed);
        }
        else
        {
            output = Forward(preprocessed);
        }

        // Get the probability value
        var probabilities = output.ToVector().ToArray();
        return probabilities.Length > 0 ? probabilities[0] : NumOps.FromDouble(0);
    }

    /// <inheritdoc/>
    public IReadOnlyList<(int StartSample, int EndSample)> DetectSpeechSegments(Tensor<T> audio)
    {
        var samples = audio.ToVector().ToArray();
        var segments = new List<(int, int)>();

        int minSpeechFrames = (_minSpeechDurationMs * SampleRate) / (1000 * _frameSize);
        int minSilenceFrames = (_minSilenceDurationMs * SampleRate) / (1000 * _frameSize);

        int? segmentStart = null;
        int speechCount = 0;
        int silenceCount = 0;
        bool inSpeech = false;

        // Reset state for fresh analysis
        ResetVadState();

        for (int i = 0; i + _frameSize <= samples.Length; i += _frameSize)
        {
            var frame = new T[_frameSize];
            Array.Copy(samples, i, frame, 0, _frameSize);

            // Create tensor from frame data
            var frameTensor = new Tensor<T>([_frameSize]);
            var frameVector = frameTensor.ToVector();
            for (int j = 0; j < _frameSize; j++)
            {
                frameVector[j] = frame[j];
            }

            var prob = GetSpeechProbability(frameTensor);
            bool isSpeech = NumOps.ToDouble(prob) >= _threshold;

            if (isSpeech)
            {
                speechCount++;
                silenceCount = 0;

                if (!inSpeech && speechCount >= minSpeechFrames)
                {
                    inSpeech = true;
                    segmentStart = i - (speechCount - 1) * _frameSize;
                }
            }
            else
            {
                silenceCount++;
                speechCount = 0;

                if (inSpeech && silenceCount >= minSilenceFrames)
                {
                    inSpeech = false;
                    if (segmentStart.HasValue)
                    {
                        segments.Add((segmentStart.Value, i - (silenceCount - 1) * _frameSize));
                        segmentStart = null;
                    }
                }
            }
        }

        // Handle segment at end
        if (inSpeech && segmentStart.HasValue)
        {
            segments.Add((segmentStart.Value, samples.Length));
        }

        return segments;
    }

    /// <inheritdoc/>
    public T[] GetFrameProbabilities(Tensor<T> audio)
    {
        var samples = audio.ToVector().ToArray();
        int numFrames = samples.Length / _frameSize;
        var probabilities = new T[numFrames];

        // Reset state for fresh analysis
        ResetVadState();

        for (int i = 0; i < numFrames; i++)
        {
            var frame = new T[_frameSize];
            Array.Copy(samples, i * _frameSize, frame, 0, _frameSize);

            // Create tensor from frame data
            var frameTensor = new Tensor<T>([_frameSize]);
            var frameVector = frameTensor.ToVector();
            for (int j = 0; j < _frameSize; j++)
            {
                frameVector[j] = frame[j];
            }

            probabilities[i] = GetSpeechProbability(frameTensor);
        }

        return probabilities;
    }

    /// <inheritdoc/>
    public (bool IsSpeech, T Probability) ProcessChunk(Tensor<T> audioChunk)
    {
        var prob = GetSpeechProbability(audioChunk);
        var isSpeech = NumOps.ToDouble(prob) >= _threshold;

        // Apply hangover logic
        if (isSpeech)
        {
            _speechFrameCount++;
            _silenceFrameCount = 0;
        }
        else
        {
            _silenceFrameCount++;
            _speechFrameCount = 0;
        }

        int minSpeechFrames = (_minSpeechDurationMs * SampleRate) / (1000 * _frameSize);
        int minSilenceFrames = (_minSilenceDurationMs * SampleRate) / (1000 * _frameSize);

        if (!_inSpeech && _speechFrameCount >= minSpeechFrames)
        {
            _inSpeech = true;
        }
        else if (_inSpeech && _silenceFrameCount >= minSilenceFrames)
        {
            _inSpeech = false;
        }

        return (_inSpeech, prob);
    }

    /// <summary>
    /// Resets the VAD streaming state (implements IVoiceActivityDetector).
    /// </summary>
    void IVoiceActivityDetector<T>.ResetState()
    {
        ResetVadState();
    }

    #endregion

    #region AudioNeuralNetworkBase Implementation

    /// <inheritdoc/>
    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio)
    {
        // Normalize audio to [-1, 1] range
        var samples = rawAudio.ToVector().ToArray();
        double maxAbs = 0;

        for (int i = 0; i < samples.Length; i++)
        {
            double absVal = Math.Abs(NumOps.ToDouble(samples[i]));
            if (absVal > maxAbs) maxAbs = absVal;
        }

        var normalizedSamples = new T[samples.Length];
        if (maxAbs > 0)
        {
            for (int i = 0; i < samples.Length; i++)
            {
                double normalized = NumOps.ToDouble(samples[i]) / maxAbs;
                normalizedSamples[i] = NumOps.FromDouble(normalized);
            }
        }
        else
        {
            Array.Copy(samples, normalizedSamples, samples.Length);
        }

        // Reshape to [batch, channels, samples] for Conv
        var result = new Tensor<T>([1, 1, samples.Length]);
        var resultVector = result.ToVector();
        for (int i = 0; i < normalizedSamples.Length; i++)
        {
            resultVector[i] = normalizedSamples[i];
        }

        return result;
    }

    /// <inheritdoc/>
    protected override Tensor<T> PostprocessOutput(Tensor<T> modelOutput)
    {
        // Output is already a probability [0, 1] from sigmoid
        return modelOutput;
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
    protected override Tensor<T> Forward(Tensor<T> input)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException("Forward pass only available in native mode.");

        var output = input;

        // Pass through conv layers
        foreach (var layer in _convLayers)
        {
            output = layer.Forward(output);
        }

        // Pass through LSTM layers
        foreach (var layer in _lstmLayers)
        {
            output = layer.Forward(output);
        }

        // Take the last timestep output and pass through dense layer
        if (_outputLayer is not null)
        {
            // Get last timestep: shape [batch, hidden] from [batch, seq, hidden]
            var lastTimestep = ExtractLastTimestep(output);
            output = _outputLayer.Forward(lastTimestep);
        }

        return output;
    }

    /// <summary>
    /// Extracts the last timestep from a sequence tensor.
    /// </summary>
    private Tensor<T> ExtractLastTimestep(Tensor<T> sequenceOutput)
    {
        var shape = sequenceOutput.Shape;
        if (shape.Length < 2) return sequenceOutput;

        // Assuming shape [batch, seq, hidden] or [batch, hidden]
        var data = sequenceOutput.ToVector().ToArray();
        int lastDim = shape[^1];
        int resultSize = shape[0] * lastDim;

        // Extract last timestep values
        var result = new Tensor<T>([shape[0], lastDim]);
        var resultVector = result.ToVector();
        int offset = data.Length - resultSize;
        for (int i = 0; i < resultSize && offset + i < data.Length; i++)
        {
            resultVector[i] = data[offset + i];
        }

        return result;
    }

    /// <inheritdoc/>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (!SupportsTraining)
            throw new InvalidOperationException("Training not supported in ONNX mode.");

        SetTrainingMode(true);

        // Forward pass
        var preprocessed = PreprocessAudio(input);
        var output = Forward(preprocessed);

        // Compute loss
        var outputVector = output.ToVector();
        var expectedVector = expectedOutput.ToVector();
        var loss = _lossFunction.CalculateLoss(outputVector, expectedVector);

        // Backward pass
        var gradients = _lossFunction.CalculateDerivative(outputVector, expectedVector);
        var gradientTensor = Tensor<T>.FromVector(gradients);

        // Backpropagate through output layer
        if (_outputLayer is not null)
        {
            gradientTensor = _outputLayer.Backward(gradientTensor);
        }

        // Backpropagate through LSTM layers (reverse order)
        for (int i = _lstmLayers.Count - 1; i >= 0; i--)
        {
            gradientTensor = _lstmLayers[i].Backward(gradientTensor);
        }

        // Backpropagate through conv layers (reverse order)
        for (int i = _convLayers.Count - 1; i >= 0; i--)
        {
            gradientTensor = _convLayers[i].Backward(gradientTensor);
        }

        SetTrainingMode(false);
    }

    /// <inheritdoc/>
    public override void UpdateParameters(Vector<T> parameters)
    {
        // Apply gradient descent updates to all layers
        var learningRate = NumOps.FromDouble(0.001);

        foreach (var layer in _convLayers)
        {
            layer.UpdateParameters(learningRate);
        }

        foreach (var layer in _lstmLayers)
        {
            layer.UpdateParameters(learningRate);
        }

        _outputLayer?.UpdateParameters(learningRate);
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = "SileroVad",
            Version = "1.0",
            ModelType = ModelType.NeuralNetwork,
            Description = "Silero Voice Activity Detection neural network"
        };

        metadata.SetProperty("SampleRate", SampleRate);
        metadata.SetProperty("FrameSize", _frameSize);
        metadata.SetProperty("Threshold", _threshold);
        metadata.SetProperty("ConvFilters", _convFilters);
        metadata.SetProperty("LstmHiddenDim", _lstmHiddenDim);
        metadata.SetProperty("NumLstmLayers", _numLstmLayers);
        metadata.SetProperty("UseNativeMode", _useNativeMode);

        return metadata;
    }

    /// <inheritdoc/>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_useNativeMode);
        writer.Write(SampleRate);
        writer.Write(_frameSize);
        writer.Write(_threshold);
        writer.Write(_minSpeechDurationMs);
        writer.Write(_minSilenceDurationMs);
        writer.Write(_convFilters);
        writer.Write(_lstmHiddenDim);
        writer.Write(_numLstmLayers);
    }

    /// <inheritdoc/>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        // Read saved values (but don't reassign readonly fields)
        _ = reader.ReadBoolean(); // _useNativeMode
        SampleRate = reader.ReadInt32();
        _ = reader.ReadInt32(); // _frameSize
        _ = reader.ReadDouble(); // _threshold
        _ = reader.ReadInt32(); // _minSpeechDurationMs
        _ = reader.ReadInt32(); // _minSilenceDurationMs
        _ = reader.ReadInt32(); // _convFilters
        _ = reader.ReadInt32(); // _lstmHiddenDim
        _ = reader.ReadInt32(); // _numLstmLayers
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new SileroVad<T>(
            Architecture,
            SampleRate,
            _frameSize,
            _threshold,
            _minSpeechDurationMs,
            _minSilenceDurationMs,
            _convFilters,
            _lstmHiddenDim,
            _numLstmLayers);
    }

    #endregion

    #region State Management

    /// <summary>
    /// Resets the VAD streaming state.
    /// </summary>
    private void ResetVadState()
    {
        _speechFrameCount = 0;
        _silenceFrameCount = 0;
        _inSpeech = false;

        // Reset LSTM layer states
        foreach (var layer in _lstmLayers)
        {
            if (layer is LSTMLayer<T> lstmLayer)
            {
                lstmLayer.ResetState();
            }
        }
    }

    #endregion

    #region Dispose

    /// <inheritdoc/>
    protected override void Dispose(bool disposing)
    {
        if (!_disposed)
        {
            if (disposing)
            {
                OnnxModel?.Dispose();

                foreach (var layer in _convLayers)
                {
                    if (layer is IDisposable disposable)
                        disposable.Dispose();
                }

                foreach (var layer in _lstmLayers)
                {
                    if (layer is IDisposable disposable)
                        disposable.Dispose();
                }

                if (_outputLayer is IDisposable disposableOutput)
                    disposableOutput.Dispose();
            }
            _disposed = true;
        }
        base.Dispose(disposing);
    }

    #endregion
}
