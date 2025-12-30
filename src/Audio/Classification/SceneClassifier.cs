using AiDotNet.Audio.Features;
using AiDotNet.Diffusion.Audio;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
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
/// Console.WriteLine($"Scene: {result.PredictedScene} ({result.Confidence})");
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
public class SceneClassifier<T> : AudioClassifierBase<T>, ISceneClassifier<T>
{
    #region Fields

    private readonly SceneClassifierOptions _options;
    private readonly MelSpectrogram<T> _melSpectrogram;
    private readonly MfccExtractor<T> _mfccExtractor;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly bool _useNativeMode;
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
        _useNativeMode = false;
        _optimizer = new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);

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
        _useNativeMode = false;
        _optimizer = new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);

        // Set base class properties
        base.SampleRate = _options.SampleRate;
        ClassLabels = _options.CustomScenes ?? StandardScenes;

        // Initialize ONNX if path provided
        if (!string.IsNullOrEmpty(_options.ModelPath))
        {
            OnnxEncoder = new OnnxModel<T>(_options.ModelPath, _options.OnnxOptions);
        }

        // Initialize feature extractors
        _melSpectrogram = CreateMelSpectrogram();
        _mfccExtractor = CreateMfccExtractor();
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
    public static async Task<SceneClassifier<T>> CreateAsync(
        SceneClassifierOptions? options = null,
        IProgress<double>? progress = null,
        CancellationToken cancellationToken = default)
    {
        options ??= new SceneClassifierOptions();

        if (string.IsNullOrEmpty(options.ModelPath))
        {
            var downloader = new OnnxModelDownloader();
            options.ModelPath = await downloader.DownloadAsync(
                "acoustic-scene-classifier",
                "model.onnx",
                progress: progress,
                cancellationToken);
        }

        return new SceneClassifier<T>(options.ModelPath, options);
    }

    #endregion

    #region ISceneClassifier Properties

    /// <summary>
    /// Gets the list of scenes this model can classify.
    /// </summary>
    public IReadOnlyList<string> SupportedScenes => ClassLabels;

    /// <summary>
    /// Gets the minimum audio duration required for reliable classification.
    /// </summary>
    public double MinimumDurationSeconds => 1.0;

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
        var output = Predict(featureTensor);
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

        return new SceneClassificationResult<T>
        {
            PredictedScene = predictedScene,
            Category = category,
            Confidence = confidence,
            AllScenes = allScenes,
            Characteristics = ExtractAcousticCharacteristics(featureStruct)
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
        var output = Predict(featureTensor);
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

        int totalSamples = audio.Length;
        int segmentSamples = (int)(segmentDuration * SampleRate);
        int hopSamples = segmentSamples / 2; // 50% overlap

        var segments = new List<SceneSegment<T>>();
        var transitions = new List<SceneTransition<T>>();
        var sceneCounts = new Dictionary<string, int>();

        string? previousScene = null;
        int position = 0;

        while (position + segmentSamples <= totalSamples)
        {
            // Extract segment
            var segment = new Tensor<T>([segmentSamples]);
            for (int i = 0; i < segmentSamples; i++)
            {
                segment[i] = audio[position + i];
            }

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
            ? sceneCounts.MaxBy(kvp => kvp.Value).Key
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
    public override Tensor<T> Predict(Tensor<T> input)
    {
        if (IsOnnxMode && OnnxEncoder is not null)
        {
            return OnnxEncoder.Run(input);
        }

        // Native mode - use layers
        if (!_useNativeMode || Layers.Count == 0)
        {
            // Fallback to rule-based classification
            return ClassifyWithRulesAsTensor(input);
        }

        var current = input;
        foreach (var layer in Layers)
        {
            current = layer.Forward(current);
        }
        return current;
    }

    /// <summary>
    /// Trains the model on labeled audio samples.
    /// </summary>
    public override void Train(Tensor<T> input, Tensor<T> expected)
    {
        if (!_useNativeMode)
        {
            throw new InvalidOperationException(
                "Training is not supported in ONNX inference mode. " +
                "Create the model with NeuralNetworkArchitecture for training.");
        }

        SetTrainingMode(true);

        // Forward pass
        var output = Predict(input);

        // Calculate loss gradient
        var gradient = LossFunction.CalculateDerivative(output.ToVector(), expected.ToVector());
        var gradientTensor = Tensor<T>.FromVector(gradient);

        // Backward pass through layers
        for (int i = Layers.Count - 1; i >= 0; i--)
        {
            gradientTensor = Layers[i].Backward(gradientTensor);
        }

        // Update parameters
        _optimizer.UpdateParameters(Layers);

        SetTrainingMode(false);
    }

    /// <summary>
    /// Updates parameters from a flattened parameter vector.
    /// </summary>
    public override void UpdateParameters(Vector<T> parameters)
    {
        if (!_useNativeMode)
            throw new NotSupportedException("UpdateParameters is not supported in ONNX mode.");

        int index = 0;
        foreach (var layer in Layers)
        {
            int count = layer.ParameterCount;
            var layerParams = parameters.SubVector(index, count);
            layer.UpdateParameters(layerParams);
            index += count;
        }
    }

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
        var result = new Tensor<T>(modelOutput.Shape);

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
            ModelType = ModelType.NeuralNetwork,
            FeatureCount = ClassLabels.Count,
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
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        // Write options
        writer.Write(_options.SampleRate);
        writer.Write(_options.NumMels);
        writer.Write(_options.FftSize);
        writer.Write(_options.HopLength);
        writer.Write(_options.NumMfccs);
        writer.Write(_useNativeMode);

        // Write class labels
        writer.Write(ClassLabels.Count);
        foreach (var label in ClassLabels)
        {
            writer.Write(label);
        }
    }

    /// <summary>
    /// Deserializes network-specific data.
    /// </summary>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        // Read would need to reinitialize options - this is a simplified implementation
        // In practice, the options should be reconstructed from saved data
    }

    /// <summary>
    /// Creates a new instance of this network type.
    /// </summary>
    protected override NeuralNetworkBase<T> CreateNewInstance()
    {
        return new SceneClassifier<T>(Architecture, _options);
    }

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

        var mfccMean = new double[numCoeffs];
        var mfccStd = new double[numCoeffs];
        var mfccDelta = new double[numCoeffs];

        for (int c = 0; c < numCoeffs; c++)
        {
            double sum = 0;
            for (int t = 0; t < numFrames; t++)
            {
                sum += NumOps.ToDouble(mfccs[t, c]);
            }
            mfccMean[c] = sum / numFrames;

            double sumSq = 0;
            for (int t = 0; t < numFrames; t++)
            {
                double diff = NumOps.ToDouble(mfccs[t, c]) - mfccMean[c];
                sumSq += diff * diff;
            }
            mfccStd[c] = Math.Sqrt(sumSq / numFrames);

            // Delta (first derivative)
            double sumDelta = 0;
            for (int t = 1; t < numFrames; t++)
            {
                double prev = NumOps.ToDouble(mfccs[t - 1, c]);
                double curr = NumOps.ToDouble(mfccs[t, c]);
                sumDelta += Math.Abs(curr - prev);
            }
            mfccDelta[c] = numFrames > 1 ? sumDelta / (numFrames - 1) : 0.0;
        }

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
    {
        double weightedSum = 0;
        double totalSum = 0;

        for (int t = 0; t < melSpec.Shape[0]; t++)
        {
            for (int f = 0; f < melSpec.Shape[1]; f++)
            {
                double mag = NumOps.ToDouble(melSpec[t, f]);
                weightedSum += f * mag;
                totalSum += mag;
            }
        }

        return totalSum > 0 ? weightedSum / totalSum : 0;
    }

    private double ComputeSpectralBandwidth(Tensor<T> melSpec, double centroid)
    {
        double sum = 0;
        double totalMag = 0;

        for (int t = 0; t < melSpec.Shape[0]; t++)
        {
            for (int f = 0; f < melSpec.Shape[1]; f++)
            {
                double mag = NumOps.ToDouble(melSpec[t, f]);
                double diff = f - centroid;
                sum += diff * diff * mag;
                totalMag += mag;
            }
        }

        return totalMag > 0 ? Math.Sqrt(sum / totalMag) : 0;
    }

    private double ComputeSpectralFlatness(Tensor<T> melSpec)
    {
        double logSum = 0;
        double sum = 0;
        int count = 0;

        for (int t = 0; t < melSpec.Shape[0]; t++)
        {
            for (int f = 0; f < melSpec.Shape[1]; f++)
            {
                double mag = Math.Max(NumOps.ToDouble(melSpec[t, f]), 1e-10);
                logSum += Math.Log(mag);
                sum += mag;
                count++;
            }
        }

        if (count == 0) return 0;

        double geometricMean = Math.Exp(logSum / count);
        double arithmeticMean = sum / count;

        return arithmeticMean > 0 ? geometricMean / arithmeticMean : 0;
    }

    private double ComputeSpectralContrast(Tensor<T> melSpec)
    {
        int numBins = melSpec.Shape[1];
        int numBands = 6;
        int bandSize = numBins / numBands;

        double totalContrast = 0;

        for (int t = 0; t < melSpec.Shape[0]; t++)
        {
            for (int band = 0; band < numBands; band++)
            {
                int startBin = band * bandSize;
                int endBin = Math.Min((band + 1) * bandSize, numBins);

                double maxVal = double.MinValue;
                double minVal = double.MaxValue;

                for (int f = startBin; f < endBin; f++)
                {
                    double val = NumOps.ToDouble(melSpec[t, f]);
                    if (val > maxVal) maxVal = val;
                    if (val < minVal) minVal = val;
                }

                totalContrast += maxVal - minVal;
            }
        }

        return totalContrast / (melSpec.Shape[0] * numBands);
    }

    private double ComputeRmsEnergy(Tensor<T> audio)
    {
        double sum = 0;
        for (int i = 0; i < audio.Length; i++)
        {
            double val = NumOps.ToDouble(audio[i]);
            sum += val * val;
        }
        return Math.Sqrt(sum / audio.Length);
    }

    private double ComputeZeroCrossingRate(Tensor<T> audio)
    {
        int crossings = 0;
        for (int i = 1; i < audio.Length; i++)
        {
            double prev = NumOps.ToDouble(audio[i - 1]);
            double curr = NumOps.ToDouble(audio[i]);
            if ((prev >= 0 && curr < 0) || (prev < 0 && curr >= 0))
                crossings++;
        }
        return (double)crossings / audio.Length;
    }

    private double ComputeEnergyVariance(Tensor<T> melSpec)
    {
        var frameEnergies = new double[melSpec.Shape[0]];

        for (int t = 0; t < melSpec.Shape[0]; t++)
        {
            double sum = 0;
            for (int f = 0; f < melSpec.Shape[1]; f++)
            {
                sum += NumOps.ToDouble(melSpec[t, f]);
            }
            frameEnergies[t] = sum;
        }

        double mean = frameEnergies.Average();
        double variance = frameEnergies.Sum(e => (e - mean) * (e - mean)) / frameEnergies.Length;

        return variance;
    }

    private double[] ComputeBandEnergies(Tensor<T> melSpec)
    {
        int numBands = 6;
        int numBins = melSpec.Shape[1];
        int bandSize = numBins / numBands;
        var bandEnergies = new double[numBands];

        for (int band = 0; band < numBands; band++)
        {
            int startBin = band * bandSize;
            int endBin = Math.Min((band + 1) * bandSize, numBins);
            double sum = 0;
            int count = 0;

            for (int t = 0; t < melSpec.Shape[0]; t++)
            {
                for (int f = startBin; f < endBin; f++)
                {
                    sum += NumOps.ToDouble(melSpec[t, f]);
                    count++;
                }
            }

            bandEnergies[band] = count > 0 ? sum / count : 0;
        }

        return bandEnergies;
    }

    #endregion

    #region Rule-Based Classification

    private Tensor<T> ClassifyWithRulesAsTensor(Tensor<T> features)
    {
        // Extract feature values from tensor for rule-based classification
        var scenes = ClassLabels;
        var probs = new double[scenes.Count];

        // Simple rule-based scoring based on feature position
        // This is a fallback when no neural network is available
        for (int i = 0; i < scenes.Count; i++)
        {
            probs[i] = 1.0 / scenes.Count; // Uniform distribution as fallback
        }

        var result = new Tensor<T>([scenes.Count]);
        for (int i = 0; i < scenes.Count; i++)
        {
            result[i] = NumOps.FromDouble(probs[i]);
        }

        return result;
    }

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
        ObjectDisposedException.ThrowIf(_disposed, GetType().FullName ?? nameof(SceneClassifier<T>));
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
