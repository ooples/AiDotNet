using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Audio.Classification;
using AiDotNet.Diffusion.Audio;
using AiDotNet.Enums;
using AiDotNet.Helpers;
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
/// <example>
/// <code>
/// // Create a speech emotion recognizer using a pre-trained ONNX model
/// var architecture = new NeuralNetworkArchitecture&lt;float&gt;(
///     inputType: InputType.OneDimensional,
///     taskType: NeuralNetworkTaskType.AudioProcessing,
///     inputSize: 80,
///     outputSize: 7);
///
/// var recognizer = new SpeechEmotionRecognizer&lt;float&gt;(
///     architecture: architecture,
///     modelPath: "emotion_model.onnx",
///     sampleRate: 16000);
///
/// // Classify emotions from audio
/// var result = recognizer.RecognizeEmotion(audioTensor);
/// // result.Emotion: "Happy", "Sad", "Angry", etc.
/// </code>
/// </example>
[ModelDomain(ModelDomain.Audio)]
[ModelDomain(ModelDomain.Language)]
[ModelCategory(ModelCategory.ConvolutionalNetwork)]
[ModelCategory(ModelCategory.RecurrentNetwork)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Classification)]
[ModelTask(ModelTask.FeatureExtraction)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
// Citation replaced: it named the WRONG KIND of source, not merely a wrong URL. The recorded title
// was a REVIEW ("Speech Emotion Recognition Using Deep Learning Techniques: A Review"), which defines
// no architecture and so cannot be what a model implements, and its arXiv id 2101.06572 resolves to
// "Tracial smooth functions of non-commuting variables and the free Wasserstein manifold" — unrelated
// to speech, emotion or learning.
//
// Identified the paper this class actually implements by matching the architecture: Badshah et al.
// build "three convolutional layers and three fully connected layers" over SPECTROGRAMS to predict
// SEVEN emotions. That is exactly this model — CreateSpeechEmotionRecognizerLayers emits 3 conv blocks
// (conv + BatchNorm + pool) then 3 dense layers (hiddenDim, hiddenDim/2, numEmotions), with
// numEmotions defaulting to 7 and mel-spectrogram input. Published at PlatCon 2017 and not on arXiv,
// so the canonical DOI is used.
[ResearchPaper("Speech Emotion Recognition from Spectrograms with Deep Convolutional Neural Network",
    "https://doi.org/10.1109/PlatCon.2017.7883728",
    Year = 2017,
    Authors = "Abdul Malik Badshah, Jamil Ahmad, Nasir Rahim, Sung Wook Baik")]
public partial class SpeechEmotionRecognizer<T> : AudioClassifierBase<T>, IEmotionRecognizer<T>
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
    /// <summary>
    /// Default learning rate for the tape trainer's optimizer.
    /// </summary>
    /// <remarks>
    /// A library default, not a paper value: Badshah et al. is an IEEE PlatCon publication whose
    /// optimizer settings are not available in an accessible source. Chosen an order of magnitude below
    /// Adam's own 1e-3 default because that default, combined with the absence of gradient clipping,
    /// measurably drove the memorization probe's loss UP over 100 steps. Overridable per instance.
    /// </remarks>
    private const double DefaultLearningRate = 1e-4;

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
    /// // Result is available in the returned value
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
        // Badshah et al. specify THREE convolutional layers; this defaulted to 4.
        int numConvBlocks = 3,
        int baseFilters = 32,
        int hiddenDim = 256,
        double dropoutRate = 0.3,
        string[]? emotionLabels = null,
        bool includeArousalValence = true,
        ILossFunction<T>? lossFunction = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        double learningRate = DefaultLearningRate)
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
            LossFunction = new CrossEntropyWithLogitsLoss<T>();
        }

        // Publish an optimizer to the tape trainer.
        //
        // This model previously configured NO optimizer at all, so training silently fell back to the
        // base class's lazily-created Adam at its own 1e-3 default with no gradient clipping. The
        // measured consequence was that the 100-step memorization probe ended HIGHER than it started
        // (1.386729 -> 1.475667): a spectrogram CNN whose dense head sees a large flattened feature
        // vector produces big early gradients, and without clipping the first steps overshoot into a
        // region the run does not recover from. Every other invariant passed, which is why this looked
        // like a converging model.
        //
        // Clipping is a stability measure rather than a claim about the paper: Badshah et al. is an
        // IEEE PlatCon publication whose optimizer settings are not available in an accessible source,
        // so the learning rate is a documented library default and both it and the optimizer itself are
        // caller-overridable.
        SetBaseTrainOptimizer(optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(
            this,
            new AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = learningRate,
                EnableGradientClipping = true,
                MaxGradientNorm = 1.0
            }));

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

        int numFrames = (int)((_inputDurationSeconds * SampleRate - _nFft) / _hopLength) + 1;

        var layers = (Architecture.Layers != null && Architecture.Layers.Count > 0)
            ? Architecture.Layers.ToList()
            : LayerHelper<T>.CreateSpeechEmotionRecognizerLayers(
                numMels: NumMels, numFrames: numFrames, baseFilters: _baseFilters,
                numConvBlocks: _numConvBlocks, hiddenDim: _hiddenDim,
                dropoutRate: _dropoutRate, numEmotions: _emotionLabels.Length).ToList();

        Layers.Clear();
        _convLayers.Clear();
        _denseLayers.Clear();
        Layers.AddRange(layers);

        // Assign internal references for forward pass
        // Conv section: numConvBlocks * 2 (conv+bn) + (numConvBlocks-1) pool + 1 flatten
        int convLayerCount = _numConvBlocks * 2 + (_numConvBlocks - 1) + 1;
        for (int i = 0; i < convLayerCount && i < layers.Count; i++)
            _convLayers.Add(layers[i]);

        // Dense section: dense + optional dropout + dense + optional dropout
        int denseStart = convLayerCount;
        int denseCount = _dropoutRate > 0 ? 4 : 2;
        for (int i = 0; i < denseCount && denseStart + i < layers.Count; i++)
            _denseLayers.Add(layers[denseStart + i]);

        // Output layer is the last layer
        if (layers.Count > 0)
            _outputLayer = layers[^1];
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
        return Tensor<T>.FromVector(resultVector, melSpec._shape);
    }

    /// <inheritdoc/>
    protected override Tensor<T> PostprocessOutput(Tensor<T> modelOutput)
    {
        // Output is already softmax probabilities
        return modelOutput;
    }

    #endregion

    #region Forward Pass

    /// <summary>
    /// Shapes a mel spectrogram into the rank-4 <c>[B, C, H, W]</c> form the conv stack needs,
    /// turning a rank-2 <c>[numFrames, numMels]</c> spectrogram into <c>[1, 1, numFrames, numMels]</c>.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <see cref="PreprocessAudio"/> returns the mel spectrogram as a RANK-2 tensor, but the conv
    /// stack built by <c>CreateSpeechEmotionRecognizerLayers</c> accepts only rank-3 <c>[C,H,W]</c>
    /// or rank-4 <c>[B,C,H,W]</c>. Every native-mode inference path — <see cref="Forward"/> via
    /// <see cref="GetEmotionProbabilities"/>, and <see cref="ExtractEmotionFeatures"/> — therefore
    /// threw "ConvolutionalLayer expects rank-3 [C,H,W] or rank-4 [B,C,H,W] input; got rank 2"
    /// before reaching the first layer, so the model could not run in native mode at all.
    /// </para>
    /// <para>
    /// The batch axis is what matters, and rank-3 is NOT sufficient. With a rank-3 <c>[C,H,W]</c>
    /// tensor the trailing <see cref="FlattenLayer{T}"/> reads dim 0 as a BATCH dimension, so a
    /// measured <c>[16,4,32]</c> feature map flattened to <c>[16,128]</c> and the dense head emitted
    /// <c>[16,4]</c> — 16 rows of 4 logits. <c>ApplySoftmax</c> then reported "Expected 4 logits but
    /// got 64" (64 = 16 channels x 4 classes). Prepending an explicit singleton batch axis makes the
    /// flatten collapse C*H*W into one row, so the head emits exactly <c>[1, numEmotions]</c>.
    /// </para>
    /// <para>
    /// Reshaping goes through <c>Engine</c> so the operation is recorded on the autodiff tape:
    /// a bare <c>tensor.Reshape</c> here would sever gradient flow for every training invariant.
    /// Rank-4 input (already batched) passes through untouched; a rank-3 <c>[C,H,W]</c> input gets
    /// the batch axis it is missing.
    /// </para>
    /// </remarks>
    private Tensor<T> AddChannelAxisIfNeeded(Tensor<T> input)
    {
        int rank = input.Shape.Length;

        if (rank == 2)
            return Engine.Reshape(input, new[] { 1, 1, input.Shape[0], input.Shape[1] });

        if (rank == 3)
            return Engine.Reshape(input, new[] { 1, input.Shape[0], input.Shape[1], input.Shape[2] });

        return input;
    }

    /// <inheritdoc/>
    protected override Tensor<T> Forward(Tensor<T> input)
    {
        if (_isOnnxMode)
        {
            throw new InvalidOperationException("Forward pass only available in native mode.");
        }

        var output = AddChannelAxisIfNeeded(input);

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
        if (NumOps.LessThan(secondaryConfidence, NumOps.FromDouble(0.15)))
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
        // Use the private helper to avoid redundant code
        var probs = GetEmotionProbabilities(audio);
        return ComputeArousalFromProbabilities(probs);
    }

    /// <inheritdoc/>
    public T GetValence(Tensor<T> audio)
    {
        // Use the private helper to avoid redundant code
        var probs = GetEmotionProbabilities(audio);
        return ComputeValenceFromProbabilities(probs);
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
        Tensor<T> features = AddChannelAxisIfNeeded(preprocessed);

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

        SetTrainingMode(true);
        try
        {
            TrainWithTape(input, expected);
        }
        finally
        {
            SetTrainingMode(false);
        }
    }

    // UpdateParameters restated the base verbatim; ModelBase routes it to SetParameters.
    #endregion

    #region Model Serialization

    /// <inheritdoc/>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        var probabilities = GetEmotionProbabilities(input);

        // Write DIRECTLY into the result tensor. Tensor<T>.ToVector() materializes a COPY, not a
        // view over the tensor's storage, so the previous `result.ToVector()[i] = prob` populated a
        // throwaway vector and returned the freshly-allocated (all-zero) tensor: Predict emitted
        // [0, 0, 0, 0] for EVERY input. That made all input-sensitivity invariants compare zeros to
        // zeros — DifferentInputs saw L2 = 0 and read it as a collapsed network, while the network
        // itself was healthy (measured L2 = 8.9e-1 between two probes at the output layer).
        var result = new Tensor<T>([_emotionLabels.Length]);
        for (int i = 0; i < _emotionLabels.Length; i++)
        {
            if (probabilities.TryGetValue(_emotionLabels[i], out var prob))
            {
                result[i] = prob;
            }
        }

        return result;
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = "SpeechEmotionRecognizer",
            Version = "1.0",
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


    /// <inheritdoc/>


    #endregion
}
