using AiDotNet.ActivationFunctions;
using AiDotNet.Audio.Classification;
using AiDotNet.Diffusion.Audio;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Onnx;

namespace AiDotNet.Audio.Emotion;

/// <summary>
/// Neural network-based speech emotion recognition model that classifies emotional states from audio.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// This model uses deep learning to detect emotions from speech audio. It supports two operation modes:
/// <list type="bullet">
/// <item><description><b>ONNX Mode:</b> Load pre-trained models for fast inference</description></item>
/// <item><description><b>Native Mode:</b> Train models from scratch with full customization</description></item>
/// </list>
/// </para>
/// <para>
/// <b>For Beginners:</b> This is like teaching a computer to "hear" emotions in someone's voice!
///
/// How it works:
/// 1. Audio is converted to a mel spectrogram (a visual representation of sound frequencies over time)
/// 2. A neural network analyzes patterns in the spectrogram
/// 3. The network outputs probabilities for each emotion (happy, sad, angry, etc.)
///
/// Key features detected:
/// - Pitch patterns (high pitch often = excitement, low pitch often = sadness)
/// - Speaking rate (fast = excited/angry, slow = sad/calm)
/// - Volume dynamics (loud = angry, soft = sad/fearful)
/// - Voice quality (breathy, tense, relaxed)
///
/// Common applications:
/// - Call centers: Detect frustrated customers for priority handling
/// - Mental health: Monitor patient emotional well-being
/// - Voice assistants: Respond appropriately to user mood
/// - Gaming: Adapt gameplay to player emotional state
/// - Market research: Analyze focus group reactions
///
/// Default emotions supported (based on industry standards):
/// - Neutral, Happy, Sad, Angry, Fearful, Disgusted, Surprised
///
/// You can also measure:
/// - Arousal: How activated/calm the speaker is (-1 to +1)
/// - Valence: How positive/negative the emotion is (-1 to +1)
/// </para>
/// </remarks>
public class SpeechEmotionRecognizer<T> : AudioClassifierBase<T>, IEmotionRecognizer<T>
{
    #region Execution Mode

    /// <summary>
    /// Indicates whether this model is running in ONNX inference mode.
    /// </summary>
    private readonly bool _isOnnxMode;

    #endregion

    #region ONNX Mode Fields

    /// <summary>
    /// Path to the ONNX emotion recognition model.
    /// </summary>
    private readonly string? _modelPath;

    #endregion

    #region Native Mode Fields

    /// <summary>
    /// Convolutional feature extraction layers.
    /// </summary>
    private List<ILayer<T>> _convLayers = [];

    /// <summary>
    /// Dense classification layers.
    /// </summary>
    private List<ILayer<T>> _denseLayers = [];

    /// <summary>
    /// Output layer for emotion classification.
    /// </summary>
    private ILayer<T>? _outputLayer;

    /// <summary>
    /// Number of convolutional blocks in the feature extractor.
    /// </summary>
    private int _numConvBlocks;

    /// <summary>
    /// Number of filters in the first convolutional layer (doubles with each block).
    /// </summary>
    private int _baseFilters;

    /// <summary>
    /// Hidden dimension for dense layers.
    /// </summary>
    private int _hiddenDim;

    /// <summary>
    /// Dropout rate for regularization.
    /// </summary>
    private double _dropoutRate;

    #endregion

    #region Audio Configuration

    /// <summary>
    /// FFT window size for spectrogram computation.
    /// </summary>
    private int _nFft;

    /// <summary>
    /// Hop length between FFT frames.
    /// </summary>
    private int _hopLength;

    /// <summary>
    /// Expected input duration in seconds.
    /// </summary>
    private double _inputDurationSeconds;

    /// <summary>
    /// Mel spectrogram extractor.
    /// </summary>
    private MelSpectrogram<T>? _melSpec;

    /// <summary>
    /// Whether to include arousal/valence prediction.
    /// </summary>
    private bool _includeArousalValence;

    #endregion

    #region Emotion Classes

    /// <summary>
    /// Standard emotions supported by this model.
    /// </summary>
    private static readonly string[] DefaultEmotions =
    [
        "neutral",
        "happy",
        "sad",
        "angry",
        "fearful",
        "disgusted",
        "surprised"
    ];

    /// <summary>
    /// Custom emotion labels if provided.
    /// </summary>
    private string[] _emotionLabels;

    #endregion

    #region IEmotionRecognizer Properties

    /// <inheritdoc/>
    public IReadOnlyList<string> SupportedEmotions => _emotionLabels;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a speech emotion recognizer in ONNX inference mode with a pre-trained model.
    /// </summary>
    /// <param name="architecture">The neural network architecture provided by the user.</param>
    /// <param name="modelPath">Path to the ONNX emotion recognition model.</param>
    /// <param name="sampleRate">Audio sample rate in Hz. Default: 16000 (standard for speech).</param>
    /// <param name="numMels">Number of mel spectrogram bands. Default: 80 (industry standard).</param>
    /// <param name="nFft">FFT window size. Default: 1024 samples.</param>
    /// <param name="hopLength">Hop length between FFT frames. Default: 256 samples.</param>
    /// <param name="emotionLabels">Custom emotion labels. If null, uses standard 7 emotions.</param>
    /// <param name="includeArousalValence">Whether to include arousal/valence prediction. Default: true.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this constructor to load a pre-trained model.
    /// Pre-trained models are ready to use immediately without training.
    ///
    /// Example:
    /// <code>
    /// var architecture = new NeuralNetworkArchitecture&lt;float&gt;(...);
    /// var recognizer = new SpeechEmotionRecognizer&lt;float&gt;(
    ///     architecture,
    ///     "emotion_model.onnx");
    ///
    /// var result = recognizer.RecognizeEmotion(audioTensor);
    /// Console.WriteLine($"Emotion: {result.Emotion}, Confidence: {result.Confidence}");
    /// </code>
    /// </para>
    /// </remarks>
    public SpeechEmotionRecognizer(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        int sampleRate = 16000,
        int numMels = 80,
        int nFft = 1024,
        int hopLength = 256,
        string[]? emotionLabels = null,
        bool includeArousalValence = true)
        : base(architecture)
    {
        if (architecture is null)
            throw new ArgumentNullException(nameof(architecture));
        if (string.IsNullOrWhiteSpace(modelPath))
            throw new ArgumentException("Model path cannot be null or whitespace", nameof(modelPath));

        _isOnnxMode = true;
        _modelPath = modelPath;

        // Audio configuration
        SampleRate = sampleRate;
        NumMels = numMels;
        _nFft = nFft;
        _hopLength = hopLength;
        _inputDurationSeconds = 3.0;
        _includeArousalValence = includeArousalValence;

        // Emotion labels
        _emotionLabels = emotionLabels ?? DefaultEmotions;
        ClassLabels = _emotionLabels;

        // Initialize native mode fields with defaults (not used in ONNX mode)
        _numConvBlocks = 4;
        _baseFilters = 32;
        _hiddenDim = 256;
        _dropoutRate = 0.3;

        // Create mel spectrogram extractor
        _melSpec = CreateMelSpectrogram(sampleRate, numMels, nFft, hopLength);

        // Load ONNX model
        OnnxModel = new OnnxModel<T>(modelPath);
    }

    /// <summary>
    /// Creates a speech emotion recognizer in native training mode.
    /// </summary>
    /// <param name="architecture">The neural network architecture provided by the user.</param>
    /// <param name="sampleRate">Audio sample rate in Hz. Default: 16000 (standard for speech).</param>
    /// <param name="numMels">Number of mel spectrogram bands. Default: 80.</param>
    /// <param name="nFft">FFT window size. Default: 1024 samples.</param>
    /// <param name="hopLength">Hop length between FFT frames. Default: 256 samples.</param>
    /// <param name="inputDurationSeconds">Expected input audio duration. Default: 3.0 seconds.</param>
    /// <param name="numConvBlocks">Number of convolutional feature extraction blocks. Default: 4.</param>
    /// <param name="baseFilters">Filters in first conv layer (doubles per block). Default: 32.</param>
    /// <param name="hiddenDim">Hidden dimension for dense layers. Default: 256.</param>
    /// <param name="dropoutRate">Dropout rate for regularization. Default: 0.3.</param>
    /// <param name="emotionLabels">Custom emotion labels. If null, uses standard 7 emotions.</param>
    /// <param name="includeArousalValence">Whether to include arousal/valence prediction. Default: true.</param>
    /// <param name="lossFunction">Loss function for training. Default: CrossEntropyLoss.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this constructor to train a new model from scratch.
    /// You can customize every aspect of the model architecture.
    ///
    /// Example:
    /// <code>
    /// var architecture = new NeuralNetworkArchitecture&lt;float&gt;(...);
    /// var recognizer = new SpeechEmotionRecognizer&lt;float&gt;(
    ///     architecture,
    ///     sampleRate: 16000,
    ///     numConvBlocks: 4,
    ///     hiddenDim: 256);
    ///
    /// // Train the model
    /// recognizer.Train(audioTensor, emotionLabels);
    /// </code>
    /// </para>
    /// </remarks>
    public SpeechEmotionRecognizer(
        NeuralNetworkArchitecture<T> architecture,
        int sampleRate = 16000,
        int numMels = 80,
        int nFft = 1024,
        int hopLength = 256,
        double inputDurationSeconds = 3.0,
        int numConvBlocks = 4,
        int baseFilters = 32,
        int hiddenDim = 256,
        double dropoutRate = 0.3,
        string[]? emotionLabels = null,
        bool includeArousalValence = true,
        ILossFunction<T>? lossFunction = null)
        : base(architecture)
    {
        if (architecture is null)
            throw new ArgumentNullException(nameof(architecture));

        _isOnnxMode = false;
        _modelPath = null;

        // Audio configuration
        SampleRate = sampleRate;
        NumMels = numMels;
        _nFft = nFft;
        _hopLength = hopLength;
        _inputDurationSeconds = inputDurationSeconds;
        _includeArousalValence = includeArousalValence;

        // Architecture configuration
        _numConvBlocks = numConvBlocks;
        _baseFilters = baseFilters;
        _hiddenDim = hiddenDim;
        _dropoutRate = dropoutRate;

        // Emotion labels
        _emotionLabels = emotionLabels ?? DefaultEmotions;
        ClassLabels = _emotionLabels;

        // Set loss function
        if (lossFunction is not null)
        {
            LossFunction = lossFunction;
        }
        else
        {
            LossFunction = new CrossEntropyLoss<T>();
        }

        // Create mel spectrogram extractor
        _melSpec = CreateMelSpectrogram(sampleRate, numMels, nFft, hopLength);

        // Initialize layers
        InitializeLayers();
    }

    #endregion

    #region Layer Initialization

    /// <summary>
    /// Initializes the neural network layers for native training mode.
    /// </summary>
    protected override void InitializeLayers()
    {
        if (_isOnnxMode)
        {
            return;
        }

        // Calculate input dimensions
        int numFrames = (int)((_inputDurationSeconds * SampleRate - _nFft) / _hopLength) + 1;
        int currentFilters = _baseFilters;
        int currentHeight = NumMels;
        int currentWidth = numFrames;

        // Convolutional feature extraction blocks
        for (int block = 0; block < _numConvBlocks; block++)
        {
            int inputDepth = block == 0 ? 1 : currentFilters / 2;
            int outputDepth = currentFilters;

            // Conv + ReLU
            _convLayers.Add(new ConvolutionalLayer<T>(
                inputDepth: inputDepth,
                inputHeight: currentHeight,
                inputWidth: currentWidth,
                outputDepth: outputDepth,
                kernelSize: 3,
                stride: 1,
                padding: 1,
                activationFunction: new ReLUActivation<T>()));

            // BatchNorm
            _convLayers.Add(new BatchNormalizationLayer<T>(outputDepth * currentHeight * currentWidth));

            // Max pooling (reduce by 2 in frequency dimension)
            if (block < _numConvBlocks - 1)
            {
                _convLayers.Add(new MaxPoolingLayer<T>(
                    [outputDepth, currentHeight, currentWidth],
                    poolSize: 2,
                    stride: 2));
                currentHeight /= 2;
                currentWidth /= 2;
            }

            currentFilters *= 2;
        }

        // Flatten layer
        int flattenedSize = (currentFilters / 2) * currentHeight * currentWidth;
        _convLayers.Add(new FlattenLayer<T>([(currentFilters / 2), currentHeight, currentWidth]));

        // Dense layers
        _denseLayers.Add(new DenseLayer<T>(
            inputSize: flattenedSize,
            outputSize: _hiddenDim,
            activationFunction: new ReLUActivation<T>()));

        if (_dropoutRate > 0)
        {
            _denseLayers.Add(new DropoutLayer<T>(_dropoutRate));
        }

        _denseLayers.Add(new DenseLayer<T>(
            inputSize: _hiddenDim,
            outputSize: _hiddenDim / 2,
            activationFunction: new ReLUActivation<T>()));

        if (_dropoutRate > 0)
        {
            _denseLayers.Add(new DropoutLayer<T>(_dropoutRate));
        }

        // Output layer - outputs logits (softmax is applied in GetEmotionProbabilities)
        _outputLayer = new DenseLayer<T>(
            inputSize: _hiddenDim / 2,
            outputSize: _emotionLabels.Length);
    }

    #endregion

    #region Audio Preprocessing

    /// <inheritdoc/>
    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio)
    {
        if (_melSpec is null)
        {
            throw new InvalidOperationException("Mel spectrogram extractor is not initialized.");
        }

        // Convert audio to mel spectrogram
        var melSpectrogram = _melSpec.Forward(rawAudio);

        // Normalize (mean and variance normalization)
        var normalized = NormalizeMelSpectrogram(melSpectrogram);

        return normalized;
    }

    /// <summary>
    /// Normalizes the mel spectrogram for neural network input.
    /// </summary>
    private Tensor<T> NormalizeMelSpectrogram(Tensor<T> melSpec)
    {
        var inputVector = melSpec.ToVector();
        int length = inputVector.Length;

        // Compute mean
        T sum = NumOps.Zero;
        for (int i = 0; i < length; i++)
        {
            sum = NumOps.Add(sum, inputVector[i]);
        }
        T mean = NumOps.Divide(sum, NumOps.FromDouble(length));

        // Compute variance
        T varSum = NumOps.Zero;
        for (int i = 0; i < length; i++)
        {
            T diff = NumOps.Subtract(inputVector[i], mean);
            varSum = NumOps.Add(varSum, NumOps.Multiply(diff, diff));
        }
        T variance = NumOps.Divide(varSum, NumOps.FromDouble(length));
        T stdDev = NumOps.Sqrt(NumOps.Add(variance, NumOps.FromDouble(1e-8)));

        // Normalize - create result vector, fill it, then create tensor from it
        var resultVector = new Vector<T>(length);
        for (int i = 0; i < length; i++)
        {
            resultVector[i] = NumOps.Divide(NumOps.Subtract(inputVector[i], mean), stdDev);
        }

        // Create tensor from the normalized vector with the original shape
        return Tensor<T>.FromVector(resultVector, melSpec.Shape);
    }

    /// <inheritdoc/>
    protected override Tensor<T> PostprocessOutput(Tensor<T> modelOutput)
    {
        // Output is already softmax probabilities
        return modelOutput;
    }

    #endregion

    #region Forward Pass

    /// <inheritdoc/>
    protected override Tensor<T> Forward(Tensor<T> input)
    {
        if (_isOnnxMode)
        {
            throw new InvalidOperationException("Forward pass only available in native mode.");
        }

        var output = input;

        // Convolutional layers
        foreach (var layer in _convLayers)
        {
            output = layer.Forward(output);
        }

        // Dense layers
        foreach (var layer in _denseLayers)
        {
            output = layer.Forward(output);
        }

        // Output layer
        if (_outputLayer is not null)
        {
            output = _outputLayer.Forward(output);
        }

        return output;
    }

    #endregion

    #region IEmotionRecognizer Implementation

    /// <inheritdoc/>
    public EmotionResult<T> RecognizeEmotion(Tensor<T> audio)
    {
        var probabilities = GetEmotionProbabilities(audio);
        var probDict = new Dictionary<string, T>();
        foreach (var kvp in probabilities)
        {
            probDict[kvp.Key] = kvp.Value;
        }
        var (emotion, confidence) = GetPrediction(probDict);

        // Get secondary emotion
        string? secondaryEmotion = null;
        T secondaryConfidence = NumOps.Zero;
        foreach (var (label, prob) in probabilities)
        {
            if (label != emotion && NumOps.GreaterThan(prob, secondaryConfidence))
            {
                secondaryEmotion = label;
                secondaryConfidence = prob;
            }
        }

        // Only include secondary if significant
        if (NumOps.ToDouble(secondaryConfidence) < 0.15)
        {
            secondaryEmotion = null;
        }

        // Get arousal and valence from already-computed probabilities (avoid redundant inference)
        T arousal = _includeArousalValence ? ComputeArousalFromProbabilities(probabilities) : NumOps.Zero;
        T valence = _includeArousalValence ? ComputeValenceFromProbabilities(probabilities) : NumOps.Zero;

        return new EmotionResult<T>
        {
            Emotion = emotion,
            Confidence = confidence,
            SecondaryEmotion = secondaryEmotion,
            Arousal = arousal,
            Valence = valence
        };
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, T> GetEmotionProbabilities(Tensor<T> audio)
    {
        var preprocessed = PreprocessAudio(audio);

        Tensor<T> output;
        if (_isOnnxMode)
        {
            output = RunOnnxInference(preprocessed);
        }
        else
        {
            output = Forward(preprocessed);
        }

        return ApplySoftmax(output);
    }

    /// <inheritdoc/>
    public IReadOnlyList<TimedEmotionResult<T>> RecognizeEmotionTimeSeries(
        Tensor<T> audio,
        int windowSizeMs = 1000,
        int hopSizeMs = 500)
    {
        var results = new List<TimedEmotionResult<T>>();

        int windowSamples = windowSizeMs * SampleRate / 1000;
        int hopSamples = hopSizeMs * SampleRate / 1000;

        var audioVector = audio.ToVector();
        int totalSamples = audioVector.Length;

        for (int startSample = 0; startSample + windowSamples <= totalSamples; startSample += hopSamples)
        {
            // Extract window
            var windowTensor = new Tensor<T>([windowSamples]);
            var windowVector = windowTensor.ToVector();
            for (int i = 0; i < windowSamples; i++)
            {
                windowVector[i] = audioVector[startSample + i];
            }

            // Recognize emotion for this window
            var result = RecognizeEmotion(windowTensor);

            double startTime = (double)startSample / SampleRate;
            double endTime = (double)(startSample + windowSamples) / SampleRate;

            results.Add(new TimedEmotionResult<T>
            {
                Emotion = result.Emotion,
                Confidence = result.Confidence,
                SecondaryEmotion = result.SecondaryEmotion,
                Arousal = result.Arousal,
                Valence = result.Valence,
                StartTime = startTime,
                EndTime = endTime
            });
        }

        return results;
    }

    /// <inheritdoc/>
    public T GetArousal(Tensor<T> audio)
    {
        var probs = GetEmotionProbabilities(audio);

        // Arousal mapping (based on circumplex model of affect)
        var arousalWeights = new Dictionary<string, double>
        {
            { "neutral", 0.0 },
            { "happy", 0.5 },
            { "sad", -0.7 },
            { "angry", 0.9 },
            { "fearful", 0.6 },
            { "disgusted", 0.2 },
            { "surprised", 0.7 }
        };

        double arousal = 0;
        foreach (var (emotion, prob) in probs)
        {
            if (arousalWeights.TryGetValue(emotion.ToLowerInvariant(), out double weight))
            {
                arousal += NumOps.ToDouble(prob) * weight;
            }
        }

        return NumOps.FromDouble(Math.Max(-1.0, Math.Min(1.0, arousal)));
    }

    /// <inheritdoc/>
    public T GetValence(Tensor<T> audio)
    {
        var probs = GetEmotionProbabilities(audio);

        // Valence mapping (based on circumplex model of affect)
        var valenceWeights = new Dictionary<string, double>
        {
            { "neutral", 0.0 },
            { "happy", 0.9 },
            { "sad", -0.8 },
            { "angry", -0.6 },
            { "fearful", -0.7 },
            { "disgusted", -0.5 },
            { "surprised", 0.3 }
        };

        double valence = 0;
        foreach (var (emotion, prob) in probs)
        {
            if (valenceWeights.TryGetValue(emotion.ToLowerInvariant(), out double weight))
            {
                valence += NumOps.ToDouble(prob) * weight;
            }
        }

        return NumOps.FromDouble(Math.Max(-1.0, Math.Min(1.0, valence)));
    }

    /// <summary>
    /// Computes arousal from already-computed emotion probabilities.
    /// </summary>
    private T ComputeArousalFromProbabilities(IReadOnlyDictionary<string, T> probs)
    {
        // Arousal mapping (based on circumplex model of affect)
        var arousalWeights = new Dictionary<string, double>
        {
            { "neutral", 0.0 },
            { "happy", 0.5 },
            { "sad", -0.7 },
            { "angry", 0.9 },
            { "fearful", 0.6 },
            { "disgusted", 0.2 },
            { "surprised", 0.7 }
        };

        double arousal = 0;
        foreach (var (emotion, prob) in probs)
        {
            if (arousalWeights.TryGetValue(emotion.ToLowerInvariant(), out double weight))
            {
                arousal += NumOps.ToDouble(prob) * weight;
            }
        }

        return NumOps.FromDouble(Math.Max(-1.0, Math.Min(1.0, arousal)));
    }

    /// <summary>
    /// Computes valence from already-computed emotion probabilities.
    /// </summary>
    private T ComputeValenceFromProbabilities(IReadOnlyDictionary<string, T> probs)
    {
        // Valence mapping (based on circumplex model of affect)
        var valenceWeights = new Dictionary<string, double>
        {
            { "neutral", 0.0 },
            { "happy", 0.9 },
            { "sad", -0.8 },
            { "angry", -0.6 },
            { "fearful", -0.7 },
            { "disgusted", -0.5 },
            { "surprised", 0.3 }
        };

        double valence = 0;
        foreach (var (emotion, prob) in probs)
        {
            if (valenceWeights.TryGetValue(emotion.ToLowerInvariant(), out double weight))
            {
                valence += NumOps.ToDouble(prob) * weight;
            }
        }

        return NumOps.FromDouble(Math.Max(-1.0, Math.Min(1.0, valence)));
    }

    /// <inheritdoc/>
    public Vector<T> ExtractEmotionFeatures(Tensor<T> audio)
    {
        var preprocessed = PreprocessAudio(audio);

        if (_isOnnxMode)
        {
            var output = RunOnnxInference(preprocessed);
            return output.ToVector();
        }

        // For native mode, get embeddings before final classification layer
        Tensor<T> features = preprocessed;

        foreach (var layer in _convLayers)
        {
            features = layer.Forward(features);
        }

        foreach (var layer in _denseLayers)
        {
            features = layer.Forward(features);
        }

        // Return before output layer
        return features.ToVector();
    }

    #endregion

    #region Training

    /// <inheritdoc/>
    public override void Train(Tensor<T> input, Tensor<T> expected)
    {
        if (_isOnnxMode)
        {
            throw new InvalidOperationException(
                "Cannot train in ONNX mode. Create a new model with the native mode constructor for training.");
        }

        // Preprocess audio
        var preprocessed = PreprocessAudio(input);

        // Forward pass
        SetTrainingMode(true);
        var predictions = Forward(preprocessed);

        // Compute loss derivative
        var predVector = predictions.ToVector();
        var expVector = expected.ToVector();
        var gradients = LossFunction.CalculateDerivative(predVector, expVector);

        var gradientTensor = new Tensor<T>([gradients.Length]);
        var gradientVector = gradientTensor.ToVector();
        for (int i = 0; i < gradients.Length; i++)
        {
            gradientVector[i] = gradients[i];
        }

        // Backpropagate through output layer
        if (_outputLayer is not null)
        {
            gradientTensor = _outputLayer.Backward(gradientTensor);
        }

        // Backpropagate through dense layers
        for (int i = _denseLayers.Count - 1; i >= 0; i--)
        {
            gradientTensor = _denseLayers[i].Backward(gradientTensor);
        }

        // Backpropagate through conv layers
        for (int i = _convLayers.Count - 1; i >= 0; i--)
        {
            gradientTensor = _convLayers[i].Backward(gradientTensor);
        }

        SetTrainingMode(false);
    }

    /// <inheritdoc/>
    public override void UpdateParameters(Vector<T> parameters)
    {
        if (_isOnnxMode)
        {
            throw new InvalidOperationException("Cannot update parameters in ONNX mode.");
        }

        int offset = 0;

        // Update conv layers
        foreach (var layer in _convLayers)
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

        // Update dense layers
        foreach (var layer in _denseLayers)
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

    #endregion

    #region Model Serialization

    /// <inheritdoc/>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        var probabilities = GetEmotionProbabilities(input);

        var result = new Tensor<T>([_emotionLabels.Length]);
        var resultVector = result.ToVector();
        for (int i = 0; i < _emotionLabels.Length; i++)
        {
            if (probabilities.TryGetValue(_emotionLabels[i], out var prob))
            {
                resultVector[i] = prob;
            }
        }

        return result;
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        if (_isOnnxMode && _modelPath is not null)
        {
            return new SpeechEmotionRecognizer<T>(
                Architecture,
                _modelPath,
                SampleRate,
                NumMels,
                _nFft,
                _hopLength,
                _emotionLabels,
                _includeArousalValence);
        }

        return new SpeechEmotionRecognizer<T>(
            Architecture,
            SampleRate,
            NumMels,
            _nFft,
            _hopLength,
            _inputDurationSeconds,
            _numConvBlocks,
            _baseFilters,
            _hiddenDim,
            _dropoutRate,
            _emotionLabels,
            _includeArousalValence);
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = "SpeechEmotionRecognizer",
            Version = "1.0",
            ModelType = ModelType.NeuralNetwork,
            AdditionalInfo = new Dictionary<string, object>
            {
                { "SampleRate", SampleRate },
                { "NumMels", NumMels },
                { "FFTSize", _nFft },
                { "HopLength", _hopLength },
                { "EmotionLabels", _emotionLabels },
                { "IncludeArousalValence", _includeArousalValence },
                { "IsOnnxMode", _isOnnxMode }
            }
        };

        return metadata;
    }

    /// <inheritdoc/>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_isOnnxMode);
        writer.Write(SampleRate);
        writer.Write(NumMels);
        writer.Write(_nFft);
        writer.Write(_hopLength);
        writer.Write(_inputDurationSeconds);
        writer.Write(_numConvBlocks);
        writer.Write(_baseFilters);
        writer.Write(_hiddenDim);
        writer.Write(_dropoutRate);
        writer.Write(_includeArousalValence);
        writer.Write(_emotionLabels.Length);
        foreach (var label in _emotionLabels)
        {
            writer.Write(label);
        }
    }

    /// <inheritdoc/>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        // Note: _isOnnxMode is readonly and set at construction, but we read to advance stream position
        // The deserialized model will always be in native mode since ONNX models need model files
        _ = reader.ReadBoolean(); // _isOnnxMode (read but not assigned - mode is set at construction)

        // Restore audio configuration
        SampleRate = reader.ReadInt32();
        NumMels = reader.ReadInt32();
        _nFft = reader.ReadInt32();
        _hopLength = reader.ReadInt32();
        _inputDurationSeconds = reader.ReadDouble();

        // Restore architecture configuration
        _numConvBlocks = reader.ReadInt32();
        _baseFilters = reader.ReadInt32();
        _hiddenDim = reader.ReadInt32();
        _dropoutRate = reader.ReadDouble();
        _includeArousalValence = reader.ReadBoolean();

        // Restore emotion labels
        int labelCount = reader.ReadInt32();
        _emotionLabels = new string[labelCount];
        for (int i = 0; i < labelCount; i++)
        {
            _emotionLabels[i] = reader.ReadString();
        }
        ClassLabels = _emotionLabels;

        // Reinitialize mel spectrogram extractor with restored parameters
        _melSpec = CreateMelSpectrogram(SampleRate, NumMels, _nFft, _hopLength);

        // Reinitialize layers if needed (native mode)
        if (_convLayers.Count == 0)
        {
            InitializeLayers();
        }
    }

    #endregion
}
