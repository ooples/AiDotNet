using AiDotNet.Audio.Features;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;

namespace AiDotNet.Audio.Classification;

/// <summary>
/// Music genre classification model.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Classifies audio into music genres using spectral features and neural network models.
/// Supports common genres: rock, pop, hip-hop, jazz, classical, electronic, country, R&amp;B, metal, folk.
/// </para>
/// <para>
/// This class supports both:
/// <list type="bullet">
/// <item><b>ONNX mode</b>: Load pre-trained models for fast inference</item>
/// <item><b>Native training mode</b>: Train from scratch using the layer architecture</item>
/// </list>
/// </para>
/// <para><b>For Beginners:</b> Genre classification analyzes audio characteristics:
/// <list type="bullet">
/// <item>Tempo and rhythm patterns (fast/slow, complex/simple beats)</item>
/// <item>Timbre and instrumentation (acoustic vs electronic sounds)</item>
/// <item>Harmonic content (simple vs complex chords)</item>
/// <item>Spectral features (brightness, warmth, texture)</item>
/// </list>
///
/// Usage:
/// <code>
/// // ONNX mode
/// var classifier = new GenreClassifier&lt;float&gt;(architecture, modelPath);
/// var result = classifier.Classify(audio);
///
/// // Native training mode
/// var classifier = new GenreClassifier&lt;float&gt;(architecture);
/// classifier.Train(features, labels);
/// </code>
/// </para>
/// </remarks>
public class GenreClassifier<T> : AudioClassifierBase<T>, IGenreClassifier<T>
{
    #region Fields

    private readonly GenreClassifierOptions _options;
    private readonly MfccExtractor<T> _mfccExtractor;
    private readonly SpectralFeatureExtractor<T> _spectralExtractor;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly bool _useNativeMode;
    private bool _disposed;

    #endregion

    #region Static Fields

    /// <summary>Standard music genres.</summary>
    public static readonly string[] StandardGenres =
    [
        "blues", "classical", "country", "disco", "hiphop",
        "jazz", "metal", "pop", "reggae", "rock"
    ];

    #endregion

    #region Properties

    /// <summary>
    /// Gets the sample rate.
    /// </summary>
    public new int SampleRate => _options.SampleRate;

    /// <summary>
    /// Gets the supported genre labels.
    /// </summary>
    public IReadOnlyList<string> SupportedGenres => ClassLabels;

    /// <summary>
    /// Gets the supported genre labels (legacy).
    /// </summary>
    public string[] Genres => ClassLabels.ToArray();

    /// <summary>
    /// Gets whether this model supports multi-label classification.
    /// </summary>
    public bool SupportsMultiLabel => false;

    /// <summary>
    /// Gets whether the model is operating in ONNX inference mode.
    /// </summary>
    public new bool IsOnnxMode => OnnxEncoder is not null;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a new genre classifier in ONNX inference mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture configuration.</param>
    /// <param name="modelPath">Path to the ONNX model file.</param>
    /// <param name="options">Classification options.</param>
    public GenreClassifier(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        GenreClassifierOptions? options = null)
        : base(architecture)
    {
        if (modelPath is null)
        {
            throw new ArgumentNullException(nameof(modelPath));
        }

        if (!File.Exists(modelPath))
        {
            throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        }

        _options = options ?? new GenreClassifierOptions();
        _useNativeMode = false;

        // Load ONNX model
        OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        base.SampleRate = _options.SampleRate;

        // Set class labels
        ClassLabels = _options.CustomGenres ?? StandardGenres;

        // Create feature extractors
        _mfccExtractor = CreateMfccExtractor();
        _spectralExtractor = CreateSpectralExtractor();

        // Optimizer not used in ONNX mode but required by interface
        _optimizer = new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);

        InitializeLayers();
    }

    /// <summary>
    /// Creates a new genre classifier in native training mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture configuration.</param>
    /// <param name="options">Classification options.</param>
    /// <param name="optimizer">Optional custom optimizer (defaults to AdamW).</param>
    public GenreClassifier(
        NeuralNetworkArchitecture<T> architecture,
        GenreClassifierOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new GenreClassifierOptions();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);

        base.SampleRate = _options.SampleRate;

        // Set class labels
        ClassLabels = _options.CustomGenres ?? StandardGenres;

        // Create feature extractors
        _mfccExtractor = CreateMfccExtractor();
        _spectralExtractor = CreateSpectralExtractor();

        InitializeLayers();
    }

    /// <summary>
    /// Creates a new genre classifier with legacy options only.
    /// </summary>
    /// <param name="options">Classification options.</param>
    public GenreClassifier(GenreClassifierOptions? options = null)
        : this(
            new NeuralNetworkArchitecture<T>(
                inputFeatures: (options ?? new GenreClassifierOptions()).NumMfccs * 2 + 5,
                outputSize: (options?.CustomGenres ?? StandardGenres).Length),
            options)
    {
    }

    #endregion

    #region Factory Methods

    /// <summary>
    /// Creates a GenreClassifier asynchronously with model download.
    /// </summary>
    public static async Task<GenreClassifier<T>> CreateAsync(
        GenreClassifierOptions? options = null,
        IProgress<double>? progress = null,
        CancellationToken cancellationToken = default)
    {
        options ??= new GenreClassifierOptions();

        if (options.ModelPath is null || options.ModelPath.Length == 0)
        {
            var downloader = new OnnxModelDownloader();
            options.ModelPath = await downloader.DownloadAsync(
                "music-genre-classifier",
                "model.onnx",
                progress: progress,
                cancellationToken);
        }

        var architecture = new NeuralNetworkArchitecture<T>(
            inputFeatures: options.NumMfccs * 2 + 5,
            outputSize: (options.CustomGenres ?? StandardGenres).Length);

        return new GenreClassifier<T>(architecture, options.ModelPath, options);
    }

    #endregion

    #region Initialization

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

    private SpectralFeatureExtractor<T> CreateSpectralExtractor()
    {
        return new SpectralFeatureExtractor<T>(new SpectralFeatureOptions
        {
            SampleRate = _options.SampleRate,
            FftSize = _options.FftSize,
            HopLength = _options.HopLength,
            FeatureTypes = SpectralFeatureType.Centroid | SpectralFeatureType.Rolloff
        });
    }

    /// <summary>
    /// Initializes the neural network layers.
    /// </summary>
    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;

        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
        }
        else
        {
            Layers.AddRange(LayerHelper<T>.CreateDefaultGenreClassifierLayers(
                numMels: 128,
                hiddenDim: 256,
                numClasses: ClassLabels.Count,
                maxFrames: 1000,
                dropoutRate: 0.1));
        }
    }

    #endregion

    #region IGenreClassifier Implementation

    /// <summary>
    /// Classifies the genre of an audio clip.
    /// </summary>
    /// <param name="audio">Audio waveform tensor.</param>
    /// <returns>Classification result with predicted genre and probabilities.</returns>
    public GenreClassificationResult<T> Classify(Tensor<T> audio)
    {
        ThrowIfDisposed();

        // Extract features
        var featuresTensor = ExtractFeatures(audio);

        // Get logits from model
        Tensor<T> logits;
        if (IsOnnxMode && OnnxEncoder is not null)
        {
            logits = OnnxEncoder.Run(featuresTensor);
        }
        else
        {
            logits = Predict(featuresTensor);
        }

        // Apply softmax to get probabilities
        var probabilities = ApplySoftmax(logits);

        // Get best prediction
        var (predictedGenre, confidence) = GetPrediction(probabilities);

        // Build result
        var allGenres = probabilities
            .OrderByDescending(p => NumOps.ToDouble(p.Value))
            .Select((p, index) => new GenrePrediction<T>
            {
                Genre = p.Key,
                Probability = p.Value,
                Rank = index + 1
            })
            .ToList();

        return new GenreClassificationResult<T>
        {
            PredictedGenre = predictedGenre,
            Confidence = confidence,
            AllGenres = allGenres,
            IsMultiLabel = false
        };
    }

    /// <summary>
    /// Classifies audio asynchronously.
    /// </summary>
    public Task<GenreClassificationResult<T>> ClassifyAsync(
        Tensor<T> audio,
        CancellationToken cancellationToken = default)
    {
        return Task.Run(() =>
        {
            cancellationToken.ThrowIfCancellationRequested();
            return Classify(audio);
        }, cancellationToken);
    }

    /// <summary>
    /// Gets genre probabilities for all supported genres.
    /// </summary>
    public IReadOnlyDictionary<string, T> GetGenreProbabilities(Tensor<T> audio)
    {
        ThrowIfDisposed();

        var featuresTensor = ExtractFeatures(audio);

        Tensor<T> logits;
        if (IsOnnxMode && OnnxEncoder is not null)
        {
            logits = OnnxEncoder.Run(featuresTensor);
        }
        else
        {
            logits = Predict(featuresTensor);
        }

        return ApplySoftmax(logits);
    }

    /// <summary>
    /// Gets top-K genre predictions.
    /// </summary>
    public IReadOnlyList<GenrePrediction<T>> GetTopGenres(Tensor<T> audio, int k = 5)
    {
        var probabilities = GetGenreProbabilities(audio);
        // Convert IReadOnlyDictionary to Dictionary for base class GetTopK method
        // Note: Must manually copy since net471 Dictionary constructor doesn't accept IReadOnlyDictionary
        var probabilitiesDict = new Dictionary<string, T>();
        foreach (var kvp in probabilities)
        {
            probabilitiesDict[kvp.Key] = kvp.Value;
        }
        var topK = GetTopK(probabilitiesDict, k);

        return topK.Select(t => new GenrePrediction<T>
        {
            Genre = t.Label,
            Probability = t.Probability,
            Rank = t.Rank
        }).ToList();
    }

    /// <summary>
    /// Tracks genre over time within a piece.
    /// </summary>
    public GenreTrackingResult<T> TrackGenreOverTime(Tensor<T> audio, double segmentDuration = 10.0)
    {
        ThrowIfDisposed();

        int samplesPerSegment = (int)(segmentDuration * _options.SampleRate);
        var segments = new List<GenreSegment<T>>();
        var genreCounts = new Dictionary<string, double>();

        for (int start = 0; start + samplesPerSegment <= audio.Length; start += samplesPerSegment)
        {
            var segment = new Tensor<T>([samplesPerSegment]);
            for (int i = 0; i < samplesPerSegment; i++)
            {
                segment[i] = audio[start + i];
            }

            var result = Classify(segment);
            double startTime = (double)start / _options.SampleRate;
            double endTime = (double)(start + samplesPerSegment) / _options.SampleRate;

            segments.Add(new GenreSegment<T>
            {
                StartTime = startTime,
                EndTime = endTime,
                Genre = result.PredictedGenre,
                Confidence = result.Confidence
            });

            if (!genreCounts.ContainsKey(result.PredictedGenre))
            {
                genreCounts[result.PredictedGenre] = 0;
            }
            genreCounts[result.PredictedGenre] += NumOps.ToDouble(result.Confidence);
        }

        // Find dominant genre
        string dominantGenre = genreCounts.Count > 0
            ? genreCounts.OrderByDescending(kv => kv.Value).First().Key
            : "unknown";

        // Compute distribution
        double totalWeight = genreCounts.Values.Sum();
        var distribution = genreCounts.ToDictionary(
            kv => kv.Key,
            kv => totalWeight > 0 ? kv.Value / totalWeight : 0.0);

        // Check for genre changes
        bool hasChanges = segments.Select(s => s.Genre).Distinct().Count() > 1;

        return new GenreTrackingResult<T>
        {
            DominantGenre = dominantGenre,
            Segments = segments,
            HasGenreChanges = hasChanges,
            GenreDistribution = distribution
        };
    }

    /// <summary>
    /// Extracts audio features used for classification.
    /// </summary>
    Tensor<T> IGenreClassifier<T>.ExtractFeatures(Tensor<T> audio)
    {
        return ExtractFeatures(audio);
    }

    #endregion

    #region Legacy API Support

    /// <summary>
    /// Classifies the genre of an audio clip (legacy API).
    /// </summary>
    public GenreClassificationResult ClassifyLegacy(Tensor<T> audio)
    {
        var result = Classify(audio);
        return ConvertToLegacyResult(result);
    }

    /// <summary>
    /// Classifies multiple audio segments in batch (legacy API).
    /// </summary>
    public List<GenreClassificationResult> ClassifyBatch(IEnumerable<Tensor<T>> audioSegments)
    {
        return audioSegments.Select(ClassifyLegacy).ToList();
    }

    private GenreClassificationResult ConvertToLegacyResult(GenreClassificationResult<T> result)
    {
        var allProbabilities = result.AllGenres.ToDictionary(
            g => g.Genre,
            g => NumOps.ToDouble(g.Probability));

        var topPredictions = result.AllGenres
            .Take(_options.TopK)
            .Select(g => (g.Genre, NumOps.ToDouble(g.Probability)))
            .ToList();

        return new GenreClassificationResult
        {
            PredictedGenre = result.PredictedGenre,
            Confidence = NumOps.ToDouble(result.Confidence),
            AllProbabilities = allProbabilities,
            TopPredictions = topPredictions,
            Features = new GenreFeatures { MfccMean = [], MfccStd = [] } // Default empty features for legacy API
        };
    }

    #endregion

    #region NeuralNetworkBase Implementation

    /// <summary>
    /// Predicts output for the given input.
    /// </summary>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        ThrowIfDisposed();

        if (IsOnnxMode && OnnxEncoder is not null)
        {
            return OnnxEncoder.Run(input);
        }

        // Native mode: forward pass through layers
        var current = input;
        foreach (var layer in Layers)
        {
            current = layer.Forward(current);
        }
        return current;
    }

    /// <summary>
    /// Updates model parameters.
    /// </summary>
    public override void UpdateParameters(Vector<T> parameters)
    {
        if (!_useNativeMode)
        {
            throw new NotSupportedException("UpdateParameters is not supported in ONNX mode.");
        }

        int index = 0;
        foreach (var layer in Layers)
        {
            int count = layer.ParameterCount;
            var layerParams = parameters.Slice(index, count);
            layer.UpdateParameters(layerParams);
            index += count;
        }
    }

    /// <summary>
    /// Trains the model on a single example.
    /// </summary>
    public override void Train(Tensor<T> input, Tensor<T> expected)
    {
        if (IsOnnxMode)
        {
            throw new NotSupportedException(
                "Training is not supported in ONNX mode. Create a new GenreClassifier " +
                "without modelPath parameter to train natively.");
        }

        // Set training mode
        SetTrainingMode(true);

        // Forward pass
        var output = Predict(input);

        // Compute loss and gradients
        var loss = LossFunction.CalculateLoss(output.ToVector(), expected.ToVector());
        var gradient = LossFunction.CalculateDerivative(output.ToVector(), expected.ToVector());

        // Backward pass
        var gradientTensor = Tensor<T>.FromVector(gradient);

        for (int i = Layers.Count - 1; i >= 0; i--)
        {
            gradientTensor = Layers[i].Backward(gradientTensor);
        }

        // Update parameters using optimizer
        _optimizer.UpdateParameters(Layers);

        // Set inference mode
        SetTrainingMode(false);
    }

    #endregion

    #region AudioNeuralNetworkBase Implementation

    /// <summary>
    /// Preprocesses raw audio for model input.
    /// </summary>
    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio)
    {
        return ExtractFeatures(rawAudio);
    }

    /// <summary>
    /// Postprocesses model output into the final result format.
    /// </summary>
    protected override Tensor<T> PostprocessOutput(Tensor<T> modelOutput)
    {
        // Apply softmax normalization
        var probs = ApplySoftmax(modelOutput);
        var result = new Tensor<T>([probs.Count]);
        int i = 0;
        foreach (var label in ClassLabels)
        {
            result[i++] = probs.TryGetValue(label, out var prob) ? prob : NumOps.Zero;
        }
        return result;
    }

    /// <summary>
    /// Gets metadata about the model.
    /// </summary>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "GenreClassifier-Native" : "GenreClassifier-ONNX",
            Description = "Music genre classification model",
            ModelType = ModelType.NeuralNetwork,
            FeatureCount = _options.NumMfccs * 2 + 5,
            Complexity = 1
        };
        metadata.AdditionalInfo["SampleRate"] = _options.SampleRate.ToString();
        metadata.AdditionalInfo["NumClasses"] = ClassLabels.Count.ToString();
        metadata.AdditionalInfo["Genres"] = string.Join(", ", ClassLabels);
        metadata.AdditionalInfo["Mode"] = _useNativeMode ? "Native Training" : "ONNX Inference";
        return metadata;
    }

    /// <summary>
    /// Serializes network-specific data.
    /// </summary>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(SampleRate);
        writer.Write(ClassLabels.Count);
        foreach (var label in ClassLabels)
        {
            writer.Write(label);
        }
        writer.Write(_options.NumMfccs);
        writer.Write(_options.FftSize);
        writer.Write(_options.HopLength);
    }

    /// <summary>
    /// Deserializes network-specific data.
    /// </summary>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        base.SampleRate = reader.ReadInt32();
        int numLabels = reader.ReadInt32();
        var labels = new string[numLabels];
        for (int i = 0; i < numLabels; i++)
        {
            labels[i] = reader.ReadString();
        }
        ClassLabels = labels;
        _options.NumMfccs = reader.ReadInt32();
        _options.FftSize = reader.ReadInt32();
        _options.HopLength = reader.ReadInt32();
    }

    /// <summary>
    /// Creates a new instance of this model for cloning.
    /// </summary>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        if (!_useNativeMode)
        {
            throw new NotSupportedException(
                "CreateNewInstance is not supported for ONNX models. " +
                "Create a new GenreClassifier with the model path instead.");
        }

        return new GenreClassifier<T>(
            Architecture,
            _options);
    }

    #endregion

    #region Private Methods - Feature Extraction

    private Tensor<T> ExtractFeatures(Tensor<T> audio)
    {
        // Extract MFCCs
        var mfccs = _mfccExtractor.Extract(audio);

        // Compute MFCC statistics (mean, std)
        int numFrames = mfccs.Shape[0];
        int numCoeffs = mfccs.Shape[1];

        var mfccMean = new double[numCoeffs];
        var mfccStd = new double[numCoeffs];

        for (int c = 0; c < numCoeffs; c++)
        {
            double sum = 0;
            for (int t = 0; t < numFrames; t++)
            {
                sum += NumOps.ToDouble(mfccs[t, c]);
            }
            // Guard against division by zero for empty MFCC frames
            mfccMean[c] = numFrames > 0 ? sum / numFrames : 0.0;

            double sumSq = 0;
            for (int t = 0; t < numFrames; t++)
            {
                double diff = NumOps.ToDouble(mfccs[t, c]) - mfccMean[c];
                sumSq += diff * diff;
            }
            // Guard against division by zero for empty MFCC frames
            mfccStd[c] = numFrames > 0 ? Math.Sqrt(sumSq / numFrames) : 0.0;
        }

        // Extract spectral features
        var spectralResult = _spectralExtractor.Extract(audio);
        int spectralFrames = spectralResult.Shape[0];

        var spectralCentroid = new Tensor<T>([spectralFrames]);
        var spectralRolloff = new Tensor<T>([spectralFrames]);
        for (int t = 0; t < spectralFrames; t++)
        {
            spectralCentroid[t] = spectralResult[t, 0];
            spectralRolloff[t] = spectralResult.Shape[1] > 1 ? spectralResult[t, 1] : NumOps.Zero;
        }

        // Compute temporal features
        double zeroCrossingRate = ComputeZeroCrossingRate(audio);
        double rmsEnergy = ComputeRmsEnergy(audio);
        double tempo = EstimateTempo(audio);

        // Flatten features to tensor
        int numFeatures = numCoeffs * 2 + 5;
        var features = new Tensor<T>([1, numFeatures]);

        int idx = 0;
        foreach (var val in mfccMean)
            features[0, idx++] = NumOps.FromDouble(val);
        foreach (var val in mfccStd)
            features[0, idx++] = NumOps.FromDouble(val);
        features[0, idx++] = NumOps.FromDouble(ComputeMean(spectralCentroid));
        features[0, idx++] = NumOps.FromDouble(ComputeStd(spectralCentroid));
        features[0, idx++] = NumOps.FromDouble(zeroCrossingRate);
        features[0, idx++] = NumOps.FromDouble(rmsEnergy);
        features[0, idx++] = NumOps.FromDouble(tempo / 200.0);

        return features;
    }

    private double ComputeMean(Tensor<T> tensor)
    {
        // Guard against division by zero for empty tensors
        if (tensor.Length == 0) return 0.0;

        double sum = 0;
        for (int i = 0; i < tensor.Length; i++)
        {
            sum += NumOps.ToDouble(tensor[i]);
        }
        return sum / tensor.Length;
    }

    private double ComputeStd(Tensor<T> tensor)
    {
        // Guard against division by zero for empty tensors
        if (tensor.Length == 0) return 0.0;

        double mean = ComputeMean(tensor);
        double sumSq = 0;
        for (int i = 0; i < tensor.Length; i++)
        {
            double diff = NumOps.ToDouble(tensor[i]) - mean;
            sumSq += diff * diff;
        }
        return Math.Sqrt(sumSq / tensor.Length);
    }

    private double ComputeZeroCrossingRate(Tensor<T> audio)
    {
        int crossings = 0;
        for (int i = 1; i < audio.Length; i++)
        {
            double prev = NumOps.ToDouble(audio[i - 1]);
            double curr = NumOps.ToDouble(audio[i]);
            if ((prev >= 0 && curr < 0) || (prev < 0 && curr >= 0))
            {
                crossings++;
            }
        }
        return (double)crossings / audio.Length;
    }

    private double ComputeRmsEnergy(Tensor<T> audio)
    {
        double sumSq = 0;
        for (int i = 0; i < audio.Length; i++)
        {
            double val = NumOps.ToDouble(audio[i]);
            sumSq += val * val;
        }
        return Math.Sqrt(sumSq / audio.Length);
    }

    private double EstimateTempo(Tensor<T> audio)
    {
        // Simple onset-based tempo estimation
        int frameSize = _options.HopLength;
        int numFrames = audio.Length / frameSize;
        if (numFrames < 2) return 120.0;

        var energies = new double[numFrames];
        for (int f = 0; f < numFrames; f++)
        {
            double sum = 0;
            for (int i = 0; i < frameSize && f * frameSize + i < audio.Length; i++)
            {
                double val = NumOps.ToDouble(audio[f * frameSize + i]);
                sum += val * val;
            }
            energies[f] = sum;
        }

        // Compute onset strength
        var onsets = new double[numFrames - 1];
        for (int f = 1; f < numFrames; f++)
        {
            onsets[f - 1] = Math.Max(0, energies[f] - energies[f - 1]);
        }

        // Find dominant periodicity using autocorrelation
        int minLag = (int)(0.5 * _options.SampleRate / frameSize);
        int maxLag = (int)(2.0 * _options.SampleRate / frameSize);
        maxLag = Math.Min(maxLag, onsets.Length - 1);

        double bestCorr = 0;
        int bestLag = minLag;

        for (int lag = minLag; lag <= maxLag; lag++)
        {
            double corr = 0;
            int count = 0;
            for (int i = 0; i < onsets.Length - lag; i++)
            {
                corr += onsets[i] * onsets[i + lag];
                count++;
            }
            corr /= count > 0 ? count : 1;

            if (corr > bestCorr)
            {
                bestCorr = corr;
                bestLag = lag;
            }
        }

        // Convert lag to BPM
        double lagSeconds = (double)bestLag * frameSize / _options.SampleRate;
        return 60.0 / lagSeconds;
    }

    #endregion

    #region IDisposable

    private void ThrowIfDisposed()
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(GetType().FullName ?? nameof(GenreClassifier<T>));
        }
    }

    /// <summary>
    /// Disposes managed resources.
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
