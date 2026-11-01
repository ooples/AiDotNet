using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;

namespace AiDotNet.Audio.Classification;

/// <summary>
/// Music genre classification after Tzanetakis and Cook 2002: a 30-dimensional "musical surface" feature vector per
/// recording, classified by one diagonal-covariance Gaussian mixture per genre.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// The features (section III) are 19 timbral texture features (means and variances over 1 s texture windows of the
/// spectral centroid, rolloff, flux and zero crossings of 23 ms analysis windows, the low-energy fraction, and the first
/// five MFCCs), 6 rhythmic content features from a wavelet beat histogram, and 5 pitch content features from multipitch
/// histograms. Each genre is a Gaussian mixture (three components by default, the paper's GMM(3)) fitted with EM from a
/// K-means start (section IV). The genres are the paper's ten.
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
/// var classifier2 = new GenreClassifier&lt;float&gt;(architecture);
/// classifier2.Train(features, labels);
/// </code>
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.Classifier)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Musical Genre Classification of Audio Signals", "https://doi.org/10.1109/TSA.2002.800560", Year = 2002, Authors = "George Tzanetakis, Perry Cook")]
public partial class GenreClassifier<T> : AudioClassifierBase<T>, IGenreClassifier<T>
{
    #region Fields

    private readonly GenreClassifierOptions _options;

    private readonly MusicalSurfaceFeatures _features;
    // Null for the paper's EM fit; an optimizer trains the mixtures discriminatively instead.
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;
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
        Options = _options;
        _useNativeMode = false;

        // Load ONNX model
        OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        base.SampleRate = _options.SampleRate;

        // Set class labels
        ClassLabels = _options.CustomGenres ?? StandardGenres;

        _features = CreateFeatureExtractor();

        _options.ModelPath = modelPath;

        InitializeLayers();
    }

    /// <summary>
    /// Creates a new genre classifier in native training mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture configuration.</param>
    /// <param name="options">Classification options.</param>
    /// <param name="optimizer">Optional optimizer. Without one (the default), <c>Train</c> fits each genre's mixture to
    /// its samples with EM, as the paper does; with one, <c>Train</c> takes gradient steps on the cross-entropy of the
    /// genre posteriors instead.</param>
    public GenreClassifier(
        NeuralNetworkArchitecture<T> architecture,
        GenreClassifierOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture, new AiDotNet.LossFunctions.CrossEntropyWithLogitsLoss<T>())
    {
        _options = options ?? new GenreClassifierOptions();
        Options = _options;
        _useNativeMode = true;
        _optimizer = optimizer;

        base.SampleRate = _options.SampleRate;

        // Set class labels
        ClassLabels = _options.CustomGenres ?? StandardGenres;

        _features = CreateFeatureExtractor();

        InitializeLayers();
    }

    /// <summary>
    /// Creates a new genre classifier with legacy options only.
    /// </summary>
    /// <param name="options">Classification options.</param>
    public GenreClassifier(GenreClassifierOptions? options = null)
        : this(
            new NeuralNetworkArchitecture<T>(
                inputFeatures: MusicalSurfaceFeatures.Length,
                outputSize: (options?.CustomGenres ?? StandardGenres).Length),
            options)
    {
    }

    #endregion

    #region Factory Methods

    /// <summary>
    /// Creates a GenreClassifier asynchronously with model download.
    /// </summary>
    internal static async Task<GenreClassifier<T>> CreateAsync(
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
            inputFeatures: MusicalSurfaceFeatures.Length,
            outputSize: (options.CustomGenres ?? StandardGenres).Length);

        return new GenreClassifier<T>(architecture, options.ModelPath, options);
    }

    #endregion

    #region Initialization

    private MusicalSurfaceFeatures CreateFeatureExtractor() => new(_options.SampleRate, _options.FftSize,
        _options.HopLength, _options.TextureWindowFrames, _options.BeatWindow, _options.BeatHop);

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
            // Section IV: one Gaussian mixture with diagonal covariance per genre over the 30 features.
            var mixtures = new AiDotNet.NeuralNetworks.Layers.GaussianMixtureClassifierLayer<T>(
                ClassLabels.Count, _options.MixtureComponents, MusicalSurfaceFeatures.Length);
            mixtures.InitializeMeans(_options.Seed is int seed
                ? AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(seed)
                : AiDotNet.Tensors.Helpers.RandomHelper.CreateSecureRandom());
            Layers.Add(mixtures);
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

        // Extract features with metadata
        var (featuresTensor, genreFeatures) = ExtractFeaturesWithMeta(audio);

        // Get logits from model
        Tensor<T> logits;
        if (IsOnnxMode && OnnxEncoder is not null)
        {
            logits = OnnxEncoder.Run(featuresTensor);
        }
        else
        {
            logits = ForwardToLogits(featuresTensor);
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
            IsMultiLabel = false,
            Features = genreFeatures
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
            logits = ForwardToLogits(featuresTensor);
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
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        ThrowIfDisposed();

        // Predict returns per-genre PROBABILITIES (a classifier's public output is a non-negative,
        // normalized class distribution — the AudioClassifier "class output must be non-negative"
        // invariant). Internal probability paths (Classify / GetGenreProbabilities) call
        // ForwardToLogits directly and softmax exactly ONCE, so they do NOT double-normalize
        // (the failure mode the previous logits-returning revision was guarding against — #1789).
        var logits = IsOnnxMode && OnnxEncoder is not null
            ? OnnxEncoder.Run(input)
            : ForwardToLogits(input);
        // Softmax the logits into a non-negative, shape-preserving probability tensor. Use the
        // tensor-softmax activation (Engine.Softmax) rather than ApplySoftmax/PostprocessOutput,
        // which return / rebuild a genre-keyed dictionary and would change the output tensor shape.
        return new AiDotNet.ActivationFunctions.SoftmaxActivation<T>().Activate(logits);
    }

    /// <summary>
    /// Native forward pass through the layers to per-genre LOGITS: each genre's log-likelihood under its mixture, so a
    /// softmax gives the posterior under equal priors. Callers that need probabilities apply ApplySoftmax exactly once;
    /// the tape-based training path (ForwardForTraining) also consumes these raw logits.
    /// </summary>
    private Tensor<T> ForwardToLogits(Tensor<T> input)
    {
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

    /// <inheritdoc />
    protected internal override IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? TrainingOptimizer => _optimizer;

    /// <inheritdoc />
    /// <exception cref="NotSupportedException">The paper's EM fit (no optimizer) computes no gradients.</exception>
    public override Vector<T> GetParameterGradients()
    {
        if (_useNativeMode && _optimizer is null)
            throw new NotSupportedException("GenreClassifier fits its mixtures with EM, which computes no gradients; " +
                "construct it with an optimizer for gradient training.");
        return base.GetParameterGradients();
    }

    /// <summary>
    /// The paper's training: every genre's mixture is fitted by EM to the feature vectors labelled with it (section IV).
    /// Pass the whole training set, <c>[samples, 30]</c> feature vectors (see <c>ExtractFeatures</c>) with one-hot genre
    /// targets <c>[samples, genres]</c>; each call refits the mixtures to the samples given. With an optimizer the
    /// standard gradient step runs instead.
    /// </summary>
    protected override bool TryApplyLearningRule(Tensor<T> input, Tensor<T> target)
    {
        if (!_useNativeMode)
            throw new InvalidOperationException("Training is only supported in native mode.");
        var layer = Layers.Count == 1 ? Layers[0] as AiDotNet.NeuralNetworks.Layers.GaussianMixtureClassifierLayer<T> : null;
        if (_optimizer is not null || layer is null) return false;

        int features = MusicalSurfaceFeatures.Length, genres = ClassLabels.Count;
        int samples = input.Length / features;
        if (samples * features != input.Length || target.Length != samples * genres)
            throw new ArgumentException($"Expected [samples, {features}] features and [samples, {genres}] one-hot targets, got " +
                $"[{string.Join(", ", input.Shape)}] and [{string.Join(", ", target.Shape)}].");
        var vectors = new List<double[]>(samples);
        var labels = new List<int>(samples);
        for (int i = 0; i < samples; i++)
        {
            var vector = new double[features];
            for (int j = 0; j < features; j++) vector[j] = NumOps.ToDouble(input.Data.Span[i * features + j]);
            int label = 0;
            for (int g = 1; g < genres; g++)
                if (NumOps.ToDouble(target.Data.Span[i * genres + g]) > NumOps.ToDouble(target.Data.Span[i * genres + label])) label = g;
            vectors.Add(vector);
            labels.Add(label);
        }
        layer.Fit(vectors, labels, _options.EmIterations, _options.Seed ?? 0);
        return true;
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
            Description = "Music genre classification (Tzanetakis and Cook 2002): musical surface features, a GMM per genre",
            FeatureCount = MusicalSurfaceFeatures.Length,
            Complexity = 1
        };
        metadata.AdditionalInfo["SampleRate"] = _options.SampleRate.ToString();
        metadata.AdditionalInfo["NumClasses"] = ClassLabels.Count.ToString();
        metadata.AdditionalInfo["Genres"] = string.Join(", ", ClassLabels);
        metadata.AdditionalInfo["Mode"] = _useNativeMode ? "Native Training" : "ONNX Inference";
        return metadata;
    }

    #endregion

    #region Private Methods - Feature Extraction

    private Tensor<T> ExtractFeatures(Tensor<T> audio)
    {
        // Delegate to ExtractFeaturesWithMeta to eliminate code duplication
        return ExtractFeaturesWithMeta(audio).Tensor;
    }

    private (Tensor<T> Tensor, GenreFeatures<T> Features) ExtractFeaturesWithMeta(Tensor<T> audio)
    {
        var samples = new double[audio.Length];
        for (int i = 0; i < samples.Length; i++) samples[i] = NumOps.ToDouble(audio.Data.Span[i]);
        var vector = _features.Extract(samples);

        var features = new Tensor<T>([1, MusicalSurfaceFeatures.Length]);
        for (int i = 0; i < vector.Length; i++) features[0, i] = NumOps.FromDouble(vector[i]);

        // Vector layout (MusicalSurfaceFeatures.Names): MFCC 1-5 means at 9-13, their variances at 14-18, and the beat
        // histogram's strongest periodicity P1, in beats per minute, at 22.
        var genreFeatures = new GenreFeatures<T>
        {
            MfccMean = Enumerable.Range(9, 5).Select(i => NumOps.FromDouble(vector[i])).ToArray(),
            MfccStd = Enumerable.Range(14, 5).Select(i => NumOps.FromDouble(Math.Sqrt(Math.Max(0, vector[i])))).ToArray(),
            Tempo = NumOps.FromDouble(vector[22])
        };

        return (features, genreFeatures);
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
