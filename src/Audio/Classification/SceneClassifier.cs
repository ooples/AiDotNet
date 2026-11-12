using AiDotNet.ActivationFunctions;
using AiDotNet.Attributes;
using AiDotNet.Audio.Features;
using AiDotNet.Diffusion.Audio;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Audio.Classification;

/// <summary>
/// Acoustic scene classification model for identifying recording environments.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Classifies audio recordings by their acoustic environment or scene context.
/// Based on DCASE (Detection and Classification of Acoustic Scenes and Events) challenge.
/// </para>
/// <para><b>For Beginners:</b> Scene classification answers "Where was this recorded?":
/// <list type="bullet">
/// <item>Indoor: office, home, shopping mall, restaurant, library</item>
/// <item>Outdoor: street, park, beach, forest, construction site</item>
/// <item>Transportation: bus, train, metro, airport, car</item>
/// </list>
///
/// Usage with ONNX model:
/// <code>
/// var classifier = new SceneClassifier&lt;float&gt;("model.onnx");
/// var result = classifier.Classify(audioTensor);
/// // Result is available in the returned value
/// </code>
///
/// Usage for training:
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;float&gt;(inputFeatures: 60, outputSize: 30);
/// var classifier = new SceneClassifier&lt;float&gt;(architecture);
/// classifier.Train(features, labels);
/// var result = classifier.Classify(newAudio);
/// </code>
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Classifier)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("A Large-Scale Evaluation of Acoustic and Subjective Music-Similarity Measures", "https://doi.org/10.1016/j.csl.2017.01.007", Year = 2017, Authors = "Annamaria Mesaros, Toni Heittola, Tuomas Virtanen")]
public partial class SceneClassifier<T> : AudioClassifierBase<T>, ISceneClassifier<T>
{
    #region Fields

    private readonly SceneClassifierOptions _options;

    private readonly MelSpectrogram<T> _melSpectrogram;
    private readonly MfccExtractor<T> _mfccExtractor;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    /// <summary>Standard acoustic scene labels (DCASE-style).</summary>
    public static readonly string[] StandardScenes =
    [
        // Indoor scenes
        "airport", "shopping_mall", "metro_station", "street_pedestrian",
        "public_square", "street_traffic", "tram", "bus", "metro", "park",

        // Extended scenes
        "office", "home", "restaurant", "cafe", "library", "classroom",
        "hospital", "gym", "church", "theater",

        // Outdoor scenes
        "beach", "forest", "mountain", "countryside", "city_center",
        "residential_area", "construction_site", "industrial_area",

        // Transportation
        "car", "train", "airplane", "boat"
    ];

    /// <summary>Scene category mapping.</summary>
    private static readonly Dictionary<string, string> SceneCategories = new()
    {
        // Indoor
        { "office", "indoor" }, { "home", "indoor" }, { "restaurant", "indoor" },
        { "cafe", "indoor" }, { "library", "indoor" }, { "classroom", "indoor" },
        { "hospital", "indoor" }, { "gym", "indoor" }, { "church", "indoor" },
        { "theater", "indoor" }, { "shopping_mall", "indoor" },

        // Outdoor urban
        { "street_pedestrian", "outdoor_urban" }, { "street_traffic", "outdoor_urban" },
        { "public_square", "outdoor_urban" }, { "city_center", "outdoor_urban" },
        { "residential_area", "outdoor_urban" }, { "construction_site", "outdoor_urban" },
        { "industrial_area", "outdoor_urban" },

        // Outdoor nature
        { "park", "outdoor_nature" }, { "beach", "outdoor_nature" },
        { "forest", "outdoor_nature" }, { "mountain", "outdoor_nature" },
        { "countryside", "outdoor_nature" },

        // Transportation
        { "airport", "transportation" }, { "metro_station", "transportation" },
        { "bus", "transportation" }, { "tram", "transportation" },
        { "metro", "transportation" }, { "car", "transportation" },
        { "train", "transportation" }, { "airplane", "transportation" },
        { "boat", "transportation" }
    };

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a SceneClassifier for ONNX inference mode.
    /// </summary>
    /// <param name="modelPath">Path to the ONNX model file.</param>
    /// <param name="options">Optional configuration options.</param>
    public SceneClassifier(string modelPath, SceneClassifierOptions? options = null)
        : base(CreateMinimalArchitecture(options))
    {
        _options = options ?? new SceneClassifierOptions();
        Options = _options;
        _useNativeMode = false;
        _options.ModelPath = modelPath;

        // Set base class properties
        base.SampleRate = _options.SampleRate;
        ClassLabels = _options.CustomScenes ?? StandardScenes;

        // Initialize ONNX model
        OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);

        // Initialize feature extractors
        _melSpectrogram = CreateMelSpectrogram();
        _mfccExtractor = CreateMfccExtractor();
    }

    /// <summary>
    /// Creates a SceneClassifier for native training mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture.</param>
    /// <param name="options">Optional configuration options.</param>
    /// <param name="optimizer">Optional custom optimizer.</param>
    public SceneClassifier(
        NeuralNetworkArchitecture<T> architecture,
        SceneClassifierOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new SceneClassifierOptions();
        Options = _options;
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);

        // Set base class properties
        base.SampleRate = _options.SampleRate;
        ClassLabels = _options.CustomScenes ?? StandardScenes;

        // Initialize feature extractors
        _melSpectrogram = CreateMelSpectrogram();
        _mfccExtractor = CreateMfccExtractor();

        // Initialize layers
        InitializeLayers();
    }

    /// <summary>
    /// Creates a SceneClassifier with default options for basic classification.
    /// </summary>
    /// <param name="options">Optional configuration options.</param>
    public SceneClassifier(SceneClassifierOptions? options = null)
        : base(CreateMinimalArchitecture(options))
    {
        _options = options ?? new SceneClassifierOptions();
        Options = _options;

        // Set base class properties
        base.SampleRate = _options.SampleRate;
        ClassLabels = _options.CustomScenes ?? StandardScenes;

        // Initialize feature extractors
        _melSpectrogram = CreateMelSpectrogram();
        _mfccExtractor = CreateMfccExtractor();

        if (_options.ModelPath is string modelPath && !string.IsNullOrEmpty(modelPath))
        {
            // An ONNX model file was supplied → ONNX inference mode.
            _useNativeMode = false;
            OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        }
        else
        {
            // No ONNX model supplied → NATIVE trainable mode. Previously this ctor left the model in a
            // broken ONNX state (no encoder AND no trainable layers), so Train() threw "not supported in
            // ONNX inference mode" and the model could neither train nor infer. Build the trainable stack
            // (matching the native architecture ctor) so a from-options SceneClassifier can be trained.
            _useNativeMode = true;
            _optimizer = new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
            InitializeLayers();
        }
    }

    private static NeuralNetworkArchitecture<T> CreateMinimalArchitecture(SceneClassifierOptions? options)
    {
        var opts = options ?? new SceneClassifierOptions();
        var scenes = opts.CustomScenes ?? StandardScenes;
        int numMfccs = opts.NumMfccs;
        // Features: MFCC mean + std + delta + spectral features + band energies
        int inputFeatures = numMfccs * 3 + 7 + 6;
        return new NeuralNetworkArchitecture<T>(inputFeatures: inputFeatures, outputSize: scenes.Length);
    }

    private MelSpectrogram<T> CreateMelSpectrogram()
    {
        return new MelSpectrogram<T>(
            sampleRate: _options.SampleRate,
            nMels: _options.NumMels,
            nFft: _options.FftSize,
            hopLength: _options.HopLength,
            fMin: 0,
            fMax: _options.SampleRate / 2,
            logMel: true);
    }

    private MfccExtractor<T> CreateMfccExtractor()
    {
        return new MfccExtractor<T>(new MfccOptions
        {
            SampleRate = _options.SampleRate,
            NumCoefficients = _options.NumMfccs,
            FftSize = _options.FftSize,
            HopLength = _options.HopLength
        });
    }

    #endregion

    #region Static Factory Methods

    /// <summary>
    /// Creates a SceneClassifier asynchronously with model download.
    /// </summary>
    internal static async Task<SceneClassifier<T>> CreateAsync(
        SceneClassifierOptions? options = null,
        IProgress<double>? progress = null,
        CancellationToken cancellationToken = default)
    {
        options ??= new SceneClassifierOptions();
        string modelPath = options.ModelPath ?? string.Empty;

        if (string.IsNullOrEmpty(modelPath))
        {
            var downloader = new OnnxModelDownloader();
            modelPath = await downloader.DownloadAsync(
                "acoustic-scene-classifier",
                "model.onnx",
                progress: progress,
                cancellationToken);
            options.ModelPath = modelPath;
        }

        return new SceneClassifier<T>(modelPath, options);
    }

    #endregion

    #region ISceneClassifier Properties

    /// <summary>
    /// Gets the list of scenes this model can classify.
    /// </summary>
    public IReadOnlyList<string> SupportedScenes => ClassLabels;

    /// <summary>
    /// Gets the scenes (alias for SupportedScenes for legacy API compatibility).
    /// </summary>
    public IReadOnlyList<string> Scenes => ClassLabels;

    /// <summary>
    /// Gets the minimum audio duration required for reliable classification.
    /// </summary>
    public double MinimumDurationSeconds => 1.0;

    #endregion

    #region Legacy API

    /// <summary>
    /// Classifies audio and returns category with confidence (legacy API compatibility).
    /// </summary>
    /// <param name="audio">Audio waveform tensor.</param>
    /// <returns>Tuple of (category, confidence).</returns>
    public (string Category, T Confidence) ClassifyCategory(Tensor<T> audio)
    {
        var result = Classify(audio);
        return (result.Category, result.Confidence);
    }

    #endregion

    #region Layer Initialization

    /// <summary>
    /// Initializes the neural network layers.
    /// </summary>
    protected override void InitializeLayers()
    {
        // Use architecture layers if provided
        if (Architecture.Layers is not null && Architecture.Layers.Any())
        {
            foreach (var layer in Architecture.Layers)
            {
                Layers.Add(layer);
            }
            return;
        }

        // Create default scene classification layers
        var layers = LayerHelper<T>.CreateDefaultLayers(Architecture);
        foreach (var layer in layers)
        {
            Layers.Add(layer);
        }

        // NOTE: no in-graph softmax layer is appended. The classification head emits LINEAR logits and
        // every probability path (Classify / GetSceneProbabilities / PostprocessOutput) applies softmax
        // exactly once, so PredictCore returns a consistent logits tensor in BOTH native and ONNX modes
        // (the ONNX graph likewise has no trailing softmax). A prior revision added a SoftmaxActivation
        // layer here, which double-softmaxed the native path — softmax(softmax(x)) flattens the
        // distribution toward uniform and can shift the top scene (#1789 review). This mirrors the
        // sibling GenreClassifier, which keeps the single softmax in the consumer paths.
    }

    #endregion

    #region ISceneClassifier Methods

    /// <summary>
    /// Classifies the acoustic scene of an audio recording.
    /// </summary>
    public SceneClassificationResult<T> Classify(Tensor<T> audio)
    {
        ThrowIfDisposed();

        // Extract features
        var featureStruct = ExtractFeaturesInternal(audio);
        var featureTensor = CreateFeatureTensor(featureStruct);

        // Get predictions
        // Feed LOGITS (not the softmaxed Predict output) to ApplySoftmax so scene probabilities are
        // softmaxed exactly once (PredictCore now returns probabilities for the inference contract).
        var output = ForwardLogits(featureTensor);
        var probabilities = ApplySoftmax(output);

        // Find best prediction
        var (predictedScene, confidence) = GetPrediction(probabilities);
        string category = GetCategory(predictedScene);

        // Create all scene predictions
        var allScenes = probabilities
            .OrderByDescending(p => NumOps.ToDouble(p.Value))
            .Select((p, index) => new ScenePrediction<T>
            {
                Scene = p.Key,
                Category = GetCategory(p.Key),
                Probability = p.Value,
                Rank = index + 1
            })
            .ToList();

        // Convert extracted features to generic SceneFeatures<T>
        // Note: SceneClassifier doesn't compute tempo (beat detection not implemented)
        var sceneFeatures = new SceneFeatures<T>
        {
            MfccMean = featureStruct.MfccMean.Select(v => NumOps.FromDouble(v)).ToArray(),
            MfccStd = featureStruct.MfccStd.Select(v => NumOps.FromDouble(v)).ToArray(),
            BandEnergies = featureStruct.BandEnergies.Select(v => NumOps.FromDouble(v)).ToArray(),
            Tempo = NumOps.Zero // Tempo detection not implemented
        };

        return new SceneClassificationResult<T>
        {
            PredictedScene = predictedScene,
            Category = category,
            Confidence = confidence,
            AllScenes = allScenes,
            Characteristics = ExtractAcousticCharacteristics(featureStruct),
            Features = sceneFeatures
        };
    }

    /// <summary>
    /// Classifies the acoustic scene asynchronously.
    /// </summary>
    public Task<SceneClassificationResult<T>> ClassifyAsync(
        Tensor<T> audio,
        CancellationToken cancellationToken = default)
    {
        return Task.Run(() => Classify(audio), cancellationToken);
    }

    /// <summary>
    /// Gets scene probabilities for all supported scenes.
    /// </summary>
    public IReadOnlyDictionary<string, T> GetSceneProbabilities(Tensor<T> audio)
    {
        ThrowIfDisposed();

        var featureStruct = ExtractFeaturesInternal(audio);
        var featureTensor = CreateFeatureTensor(featureStruct);
        // Feed LOGITS (not the softmaxed Predict output) to ApplySoftmax so scene probabilities are
        // softmaxed exactly once (PredictCore now returns probabilities for the inference contract).
        var output = ForwardLogits(featureTensor);
        var probabilities = ApplySoftmax(output);

        return probabilities;
    }

    /// <summary>
    /// Gets top-K scene predictions.
    /// </summary>
    public IReadOnlyList<ScenePrediction<T>> GetTopScenes(Tensor<T> audio, int k = 5)
    {
        var result = Classify(audio);
        return result.AllScenes.Take(k).ToList();
    }

    /// <summary>
    /// Tracks scene changes over time in longer audio.
    /// </summary>
    public SceneTrackingResult<T> TrackSceneChanges(Tensor<T> audio, double segmentDuration = 10.0)
    {
        ThrowIfDisposed();

        // Validate parameters to prevent infinite loop
        if (segmentDuration <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(segmentDuration),
                "Segment duration must be positive.");
        }

        int totalSamples = audio.Length;
        int segmentSamples = (int)(segmentDuration * SampleRate);

        // Ensure segmentSamples is at least 1 to prevent infinite loop
        if (segmentSamples < 1)
        {
            throw new ArgumentException(
                $"Segment duration ({segmentDuration}s) is too short for sample rate ({SampleRate}Hz). " +
                "Minimum duration: " + (1.0 / SampleRate).ToString("F6") + "s",
                nameof(segmentDuration));
        }

        int hopSamples = Math.Max(1, segmentSamples / 2); // 50% overlap, minimum 1 sample to prevent infinite loop

        var segments = new List<SceneSegment<T>>();
        var transitions = new List<SceneTransition<T>>();
        var sceneCounts = new Dictionary<string, int>();

        string? previousScene = null;
        int position = 0;

        var flatAudio = AudioEngineOps.Flatten(Engine, audio);
        while (position + segmentSamples <= totalSamples)
        {
            // Extract segment
            var segment = Engine.TensorSlice(flatAudio, [position], [segmentSamples]);

            // Classify segment
            var result = Classify(segment);

            double startTime = (double)position / SampleRate;
            double endTime = (double)(position + segmentSamples) / SampleRate;

            segments.Add(new SceneSegment<T>
            {
                StartTime = startTime,
                EndTime = endTime,
                Scene = result.PredictedScene,
                Confidence = result.Confidence
            });

            // Track scene counts
            if (!sceneCounts.TryGetValue(result.PredictedScene, out int count))
            {
                count = 0;
            }
            sceneCounts[result.PredictedScene] = count + 1;

            // Detect transitions
            if (previousScene is not null && previousScene != result.PredictedScene)
            {
                transitions.Add(new SceneTransition<T>
                {
                    Time = startTime,
                    FromScene = previousScene,
                    ToScene = result.PredictedScene,
                    Confidence = result.Confidence
                });
            }

            previousScene = result.PredictedScene;
            position += hopSamples;
        }

        // Calculate scene distribution
        int totalSegments = segments.Count;
        var distribution = sceneCounts.ToDictionary(
            kvp => kvp.Key,
            kvp => (double)kvp.Value / totalSegments);

        // Find dominant scene
        string dominantScene = sceneCounts.Count > 0
            ? sceneCounts.OrderByDescending(kvp => kvp.Value).First().Key
            : "unknown";

        return new SceneTrackingResult<T>
        {
            DominantScene = dominantScene,
            Segments = segments,
            Transitions = transitions,
            HasSceneChanges = transitions.Count > 0,
            SceneDistribution = distribution
        };
    }

    /// <summary>
    /// Extracts acoustic features used for scene classification.
    /// </summary>
    public Tensor<T> ExtractAcousticFeatures(Tensor<T> audio)
    {
        var features = ExtractFeaturesInternal(audio);
        return CreateFeatureTensor(features);
    }

    #endregion

    #region NeuralNetworkBase Overrides

    /// <summary>
    /// Predicts scene probabilities from audio features.
    /// </summary>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        // Classifier inference contract: Predict returns a NON-NEGATIVE class distribution
        // (probabilities), per AudioClassifierTestBase.ClassOutput_ShouldBeNonNegative and the standard
        // PyTorch convention (the forward/loss path uses raw logits; inference applies softmax). Training
        // is unaffected — the tape forward (base ForwardForTraining) walks the layers directly and never
        // calls PredictCore, so softmaxing here does NOT double-softmax the CrossEntropyWithLogits loss.
        // The internal probability paths (Classify / GetSceneProbabilities) call ForwardLogits + a single
        // ApplySoftmax, so they still softmax exactly once, and the ONNX logits become probabilities too,
        // keeping all three paths consistent.
        return PostprocessOutput(ForwardLogits(input));
    }

    /// <summary>
    /// Native forward to per-scene LOGITS (or the ONNX graph's logits). Without a loaded ONNX
    /// encoder or native layers there is no model to score with, so it throws. The linear classifier head is intentionally not soft-maxed here: <see cref="PredictCore"/>
    /// softmaxes for inference and the training tape (base ForwardForTraining) consumes these logits
    /// directly via CrossEntropyWithLogitsLoss.
    /// </summary>
    private Tensor<T> ForwardLogits(Tensor<T> input)
    {
        if (IsOnnxMode && OnnxEncoder is not null)
        {
            return OnnxEncoder.Run(input);
        }

        // Native mode - use layers
        if (!_useNativeMode || Layers.Count == 0)
        {
            // No model to score with: report it instead of returning uniform scores that would still
            // name a scene.
            throw new InvalidOperationException(
                "Scene classification requires a loaded ONNX model or initialized native layers.");
        }

        var current = input;
        foreach (var layer in Layers)
        {
            current = layer.Forward(current);
        }
        return current;
    }

    /// <inheritdoc />
    /// <remarks>In this mode the weights belong to the loaded graph. The base refuses the
    /// write on every parameter surface, so the guard is stated once here instead of being
    /// repeated -- and cannot be applied to one surface and forgotten on another.</remarks>
    protected override bool SupportsParameterMutation => _useNativeMode;
    /// <summary>
    /// Preprocesses raw audio into model input format.
    /// </summary>
    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio)
    {
        // Extract features from raw audio
        var features = ExtractFeaturesInternal(rawAudio);
        return CreateFeatureTensor(features);
    }

    /// <summary>
    /// Postprocesses model output into final predictions.
    /// </summary>
    protected override Tensor<T> PostprocessOutput(Tensor<T> modelOutput)
    {
        // Apply softmax for scene classification (single-label)
        var result = new Tensor<T>(modelOutput._shape);

        // Find max for numerical stability
        double maxVal = double.MinValue;
        for (int i = 0; i < modelOutput.Length; i++)
        {
            double val = NumOps.ToDouble(modelOutput[i]);
            if (val > maxVal) maxVal = val;
        }

        // Compute softmax
        double sum = 0;
        var expValues = new double[modelOutput.Length];
        for (int i = 0; i < modelOutput.Length; i++)
        {
            expValues[i] = Math.Exp(NumOps.ToDouble(modelOutput[i]) - maxVal);
            sum += expValues[i];
        }

        for (int i = 0; i < modelOutput.Length; i++)
        {
            result[i] = NumOps.FromDouble(expValues[i] / sum);
        }

        return result;
    }

    /// <summary>
    /// Gets model metadata for serialization.
    /// </summary>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "SceneClassifier-Native" : "SceneClassifier-ONNX",
            Description = "Acoustic scene classification model (DCASE-style)",
            FeatureCount = _options.NumMfccs,  // Input feature count, not output class count
            Complexity = 1
        };
        metadata.AdditionalInfo["SampleRate"] = _options.SampleRate.ToString();
        metadata.AdditionalInfo["NumMels"] = _options.NumMels.ToString();
        metadata.AdditionalInfo["NumScenes"] = ClassLabels.Count.ToString();
        return metadata;
    }

    /// <summary>
    /// Serializes network-specific data.
    /// </summary>


    /// <summary>
    /// Deserializes network-specific data.
    /// </summary>


    #endregion

    #region Feature Extraction

    private SceneFeatures ExtractFeaturesInternal(Tensor<T> audio)
    {
        // Extract mel spectrogram
        var melSpec = _melSpectrogram.Forward(audio);

        // Extract MFCCs
        var mfccs = _mfccExtractor.Extract(audio);

        // Compute statistics
        int numFrames = mfccs.Shape[0];
        int numCoeffs = mfccs.Shape[1];

        // Guard against empty input - return default features
        if (numFrames == 0 || numCoeffs == 0)
        {
            int defaultCoeffs = _options.NumMfccs;
            return new SceneFeatures
            {
                MfccMean = new double[defaultCoeffs],
                MfccStd = new double[defaultCoeffs],
                MfccDelta = new double[defaultCoeffs],
                SpectralCentroid = 0.0,
                SpectralBandwidth = 0.0,
                SpectralFlatness = 0.0,
                SpectralContrast = 0.0,
                RmsEnergy = 0.0,
                ZeroCrossingRate = 0.0,
                EnergyVariance = 0.0,
                BandEnergies = new double[6]  // Standard 6 frequency bands
            };
        }

        // Per-coefficient population mean / std over frames and mean absolute frame-to-frame delta.
        // SceneFeatures is double-typed, so each statistic vector is converted once at the end.
        var (meanTensor, stdTensor) = AudioEngineOps.ColumnMeanStd(Engine, mfccs);
        var mfccMean = meanTensor.Cast<double>().ToArray();
        var mfccStd = stdTensor.Cast<double>().ToArray();
        var mfccDelta = numFrames > 1
            ? Engine.TensorDivideScalar(AudioEngineOps.ColumnAbsDiffSum(Engine, mfccs), NumOps.FromDouble(numFrames - 1)).Cast<double>().ToArray()
            : new double[numCoeffs];

        // Compute spectral features
        double spectralCentroid = ComputeSpectralCentroid(melSpec);
        double spectralBandwidth = ComputeSpectralBandwidth(melSpec, spectralCentroid);
        double spectralFlatness = ComputeSpectralFlatness(melSpec);
        double spectralContrast = ComputeSpectralContrast(melSpec);

        // Compute temporal features
        double rmsEnergy = ComputeRmsEnergy(audio);
        double zeroCrossingRate = ComputeZeroCrossingRate(audio);
        double energyVariance = ComputeEnergyVariance(melSpec);

        // Compute band energies
        var bandEnergies = ComputeBandEnergies(melSpec);

        return new SceneFeatures
        {
            MfccMean = mfccMean,
            MfccStd = mfccStd,
            MfccDelta = mfccDelta,
            SpectralCentroid = spectralCentroid,
            SpectralBandwidth = spectralBandwidth,
            SpectralFlatness = spectralFlatness,
            SpectralContrast = spectralContrast,
            RmsEnergy = rmsEnergy,
            ZeroCrossingRate = zeroCrossingRate,
            EnergyVariance = energyVariance,
            BandEnergies = bandEnergies
        };
    }

    private Tensor<T> CreateFeatureTensor(SceneFeatures features)
    {
        int numFeatures = features.MfccMean.Length * 3 + 7 + features.BandEnergies.Length;
        var input = new Tensor<T>([1, numFeatures]);

        int idx = 0;
        foreach (var val in features.MfccMean) input[0, idx++] = NumOps.FromDouble(val);
        foreach (var val in features.MfccStd) input[0, idx++] = NumOps.FromDouble(val);
        foreach (var val in features.MfccDelta) input[0, idx++] = NumOps.FromDouble(val);
        input[0, idx++] = NumOps.FromDouble(features.SpectralCentroid / 100);
        input[0, idx++] = NumOps.FromDouble(features.SpectralBandwidth / 100);
        input[0, idx++] = NumOps.FromDouble(features.SpectralFlatness);
        input[0, idx++] = NumOps.FromDouble(features.SpectralContrast);
        input[0, idx++] = NumOps.FromDouble(features.RmsEnergy * 10);
        input[0, idx++] = NumOps.FromDouble(features.ZeroCrossingRate * 100);
        input[0, idx++] = NumOps.FromDouble(features.EnergyVariance);
        foreach (var val in features.BandEnergies) input[0, idx++] = NumOps.FromDouble(val);

        return input;
    }

    private AcousticCharacteristics<T> ExtractAcousticCharacteristics(SceneFeatures features)
    {
        // Estimate indoor/outdoor based on reverberation and energy patterns
        bool isIndoor = features.SpectralFlatness < 0.3 && features.EnergyVariance < 0.2;
        bool hasTraffic = features.BandEnergies[0] > 0.15 && features.EnergyVariance > 0.2;
        bool hasNature = features.SpectralFlatness > 0.3 && features.BandEnergies[3] > features.BandEnergies[0];

        // Estimate crowd density
        string crowdDensity = "none";
        if (features.ZeroCrossingRate > 0.08)
            crowdDensity = "high";
        else if (features.ZeroCrossingRate > 0.05)
            crowdDensity = "medium";
        else if (features.ZeroCrossingRate > 0.02)
            crowdDensity = "low";

        return new AcousticCharacteristics<T>
        {
            ReverberationLevel = NumOps.FromDouble(1.0 - features.SpectralFlatness),
            BackgroundNoiseLevel = NumOps.FromDouble(features.RmsEnergy),
            IsIndoor = isIndoor,
            CrowdDensity = crowdDensity,
            HasTrafficSounds = hasTraffic,
            HasNatureSounds = hasNature
        };
    }

    #endregion

    #region Spectral Feature Computation

    private double ComputeSpectralCentroid(Tensor<T> melSpec)
        => AudioEngineOps.SpectralCentroidBins(Engine, melSpec);

    private double ComputeSpectralBandwidth(Tensor<T> melSpec, double centroid)
    {
        int bins = melSpec.Shape[1];
        if (melSpec.Shape[0] == 0 || bins == 0) return 0;

        // sum_f (f - centroid)^2 * sum_t mag[t, f], over the total magnitude.
        var binTotals = Engine.ReduceSum(melSpec, [0], keepDims: false);
        var offsets = Engine.TensorSubtractScalar(Engine.TensorArange<T>(0, bins), NumOps.FromDouble(centroid));
        double sum = NumOps.ToDouble(Engine.TensorSum(Engine.TensorMultiply(Engine.TensorSquare(offsets), binTotals)));
        double totalMag = NumOps.ToDouble(Engine.TensorSum(binTotals));

        return totalMag > 0 ? Math.Sqrt(sum / totalMag) : 0;
    }

    private double ComputeSpectralFlatness(Tensor<T> melSpec)
        => AudioEngineOps.SpectralFlatness(Engine, melSpec);

    private double ComputeSpectralContrast(Tensor<T> melSpec)
    {
        int numBins = melSpec.Shape[1];
        int numBands = 6;

        // Guard against division by zero
        if (numBins == 0 || melSpec.Shape[0] == 0) return 0;

        int bandSize = Math.Max(1, numBins / numBands);
        int frames = melSpec.Shape[0];

        // Per frame and band: peak minus valley over the band's bins, summed over frames and bands.
        // Bands that start past the last bin are empty and contribute nothing.
        var totalContrast = new Tensor<T>([1]);
        for (int band = 0; band < numBands; band++)
        {
            int startBin = band * bandSize;
            int endBin = Math.Min((band + 1) * bandSize, numBins);
            if (endBin <= startBin) continue;

            var bandBins = Engine.TensorSlice(melSpec, [0, startBin], [frames, endBin - startBin]);
            var range = Engine.TensorSubtract(
                Engine.ReduceMax(bandBins, [1], keepDims: false),
                Engine.TensorAmin(bandBins, [1], keepDims: false));
            totalContrast = Engine.TensorAdd(totalContrast, Engine.ReduceSum(range, [0], keepDims: true));
        }

        int denominator = frames * numBands;
        return denominator > 0 ? NumOps.ToDouble(Engine.TensorSum(totalContrast)) / denominator : 0;
    }

    private double ComputeRmsEnergy(Tensor<T> audio)
    {
        // Guard against division by zero for empty audio
        if (audio.Length == 0) return 0;

        double sum = NumOps.ToDouble(AudioEngineOps.SumOfSquares(Engine, audio));
        return Math.Sqrt(sum / audio.Length);
    }

    private double ComputeZeroCrossingRate(Tensor<T> audio)
    {
        // Guard against division by zero for empty audio
        if (audio.Length == 0) return 0;

        int crossings = AudioEngineOps.ZeroCrossingCount(Engine, audio);
        return (double)crossings / audio.Length;
    }

    private double ComputeEnergyVariance(Tensor<T> melSpec)
    {
        // Guard against division by zero for empty input
        if (melSpec.Shape[0] == 0) return 0;

        // Population variance of the per-frame energies (sum over bins).
        var frameEnergies = Engine.ReduceSum(melSpec, [1], keepDims: false);
        var deviation = Engine.TensorSubtractScalar(frameEnergies, Engine.TensorMean(frameEnergies));
        return NumOps.ToDouble(Engine.TensorMean(Engine.TensorSquare(deviation)));
    }

    private double[] ComputeBandEnergies(Tensor<T> melSpec)
    {
        int numBands = 6;
        int numBins = melSpec.Shape[1];
        int bandSize = numBins / numBands;
        int frames = melSpec.Shape[0];

        // Mean magnitude of each band's bins over all frames (0 for an empty band); SceneFeatures is
        // double-typed, so the six means are converted together.
        var bandMeans = new Tensor<T>[numBands];
        for (int band = 0; band < numBands; band++)
        {
            int startBin = band * bandSize;
            int endBin = Math.Min((band + 1) * bandSize, numBins);
            bandMeans[band] = frames > 0 && endBin > startBin
                ? Engine.Reshape(Engine.ReduceMean(Engine.TensorSlice(melSpec, [0, startBin], [frames, endBin - startBin]), [0, 1], keepDims: false), [1])
                : new Tensor<T>([1]);
        }

        return Engine.Concat(bandMeans, 0).Cast<double>().ToArray();
    }

    #endregion

    #region Scene Categories

    private static string GetCategory(string scene)
    {
        if (SceneCategories.TryGetValue(scene.ToLowerInvariant(), out var category))
        {
            return category;
        }
        return "unknown";
    }

    #endregion

    #region Disposal

    private void ThrowIfDisposed()
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(GetType().FullName ?? nameof(SceneClassifier<T>));
        }
    }

    /// <summary>
    /// Disposes of managed resources.
    /// </summary>
    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;

        if (disposing)
        {
            OnnxEncoder?.Dispose();
        }

        _disposed = true;
        base.Dispose(disposing);
    }

    #endregion
}
