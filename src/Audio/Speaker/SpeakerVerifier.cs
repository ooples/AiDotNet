using System.Collections.Concurrent;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Audio.Speaker;

/// <summary>
/// Verifies speaker identity by comparing embeddings against enrolled speakers.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Speaker verification answers the question "Is this the person they claim to be?"
/// by comparing a test utterance against enrolled speaker embeddings.
/// </para>
/// <para><b>For Beginners:</b> Speaker verification is like voice-based password checking:
/// 1. First, you "enroll" a speaker by recording their voice samples
/// 2. Later, when someone claims to be that person, you record them and compare
/// 3. If the voices match closely enough, the identity is verified
///
/// Usage (ONNX Mode):
/// <code>
/// var verifier = new SpeakerVerifier&lt;float&gt;(
///     architecture,
///     embeddingModelPath: "speaker_model.onnx");
/// var result = verifier.Verify(audio, referenceEmbedding);
/// </code>
///
/// Usage (Native Training Mode):
/// <code>
/// var verifier = new SpeakerVerifier&lt;float&gt;(architecture);
/// verifier.Train(audioInput, expectedOutput);
/// </code>
/// </para>
/// </remarks>
public class SpeakerVerifier<T> : SpeakerRecognitionBase<T>, ISpeakerVerifier<T>
{
    #region Execution Mode

    /// <summary>
    /// Whether the model is operating in native training mode.
    /// </summary>
    private readonly bool _useNativeMode;

    #endregion

    #region ONNX Mode Fields

    /// <summary>
    /// Path to the speaker embedding model ONNX file.
    /// </summary>
    private readonly string? _embeddingModelPath;

    #endregion

    #region Shared Fields

    /// <summary>
    /// Speaker embedding extractor.
    /// </summary>
    private readonly SpeakerEmbeddingExtractor<T> _embeddingExtractor;

    /// <summary>
    /// Optimizer for training.
    /// </summary>
    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;

    /// <summary>
    /// Loss function for training.
    /// </summary>
    private ILossFunction<T> _lossFunction;

    /// <summary>
    /// Enrolled speakers dictionary.
    /// </summary>
    private readonly ConcurrentDictionary<string, SpeakerProfile<T>> _enrolledSpeakers;

    /// <summary>
    /// Whether the model has been disposed.
    /// </summary>
    private bool _disposed;

    #endregion

    #region Model Architecture Parameters

    /// <summary>
    /// Hidden dimension for encoder layers.
    /// </summary>
    private readonly int _hiddenDim;

    /// <summary>
    /// Number of encoder layers.
    /// </summary>
    private readonly int _numEncoderLayers;

    /// <summary>
    /// Number of attention heads.
    /// </summary>
    private readonly int _numHeads;

    #endregion

    #region Public Properties

    /// <summary>
    /// Gets the default verification threshold.
    /// </summary>
    public T DefaultThreshold { get; }

    /// <summary>
    /// Gets the underlying speaker embedding extractor.
    /// </summary>
    public ISpeakerEmbeddingExtractor<T> EmbeddingExtractor => _embeddingExtractor;

    /// <summary>
    /// Gets whether the model is in ONNX inference mode.
    /// </summary>
    public new bool IsOnnxMode => !_useNativeMode && _embeddingExtractor.IsOnnxMode;

    /// <summary>
    /// Gets the number of enrolled speakers.
    /// </summary>
    public int EnrolledCount => _enrolledSpeakers.Count;

    /// <summary>
    /// Gets the verification threshold.
    /// </summary>
    public double VerificationThreshold => NumOps.ToDouble(DefaultThreshold);

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a SpeakerVerifier with default settings for native training mode.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the simplest way to create a speaker verifier.
    /// It uses default settings suitable for most use cases.
    /// </para>
    /// </remarks>
    public SpeakerVerifier()
        : this(
            new NeuralNetworkArchitecture<T>(
                inputFeatures: 256,
                outputSize: 256),
            sampleRate: 16000,
            embeddingDimension: 256)
    {
    }

    /// <summary>
    /// Creates a SpeakerVerifier for ONNX inference with a pretrained model.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="embeddingModelPath">Required path to speaker embedding ONNX model.</param>
    /// <param name="sampleRate">Expected sample rate for input audio. Default is 16000.</param>
    /// <param name="embeddingDimension">Dimension of speaker embeddings. Default is 256.</param>
    /// <param name="defaultThreshold">Default verification threshold. Default is 0.6.</param>
    /// <param name="onnxOptions">ONNX runtime options.</param>
    public SpeakerVerifier(
        NeuralNetworkArchitecture<T> architecture,
        string embeddingModelPath,
        int sampleRate = 16000,
        int embeddingDimension = 256,
        double defaultThreshold = 0.6,
        OnnxModelOptions? onnxOptions = null)
        : base(architecture)
    {
        if (embeddingModelPath is null)
            throw new ArgumentNullException(nameof(embeddingModelPath));

        _useNativeMode = false;
        _embeddingModelPath = embeddingModelPath;

        // Store parameters
        SampleRate = sampleRate;
        EmbeddingDimension = embeddingDimension;
        DefaultThreshold = NumOps.FromDouble(defaultThreshold);
        _hiddenDim = 256;
        _numEncoderLayers = 3;
        _numHeads = 4;

        // Create embedding extractor
        _embeddingExtractor = new SpeakerEmbeddingExtractor<T>(
            architecture,
            embeddingModelPath,
            sampleRate,
            embeddingDimension,
            onnxOptions: onnxOptions);

        // Initialize enrolled speakers
        _enrolledSpeakers = new ConcurrentDictionary<string, SpeakerProfile<T>>();

        // Default loss function (MSE is standard for speaker verification)
        _lossFunction = new MeanSquaredErrorLoss<T>();

        // Initialize layers (empty for ONNX mode)
        InitializeLayers();
    }

    /// <summary>
    /// Creates a SpeakerVerifier for native training mode.
    /// </summary>
    /// <param name="architecture">The neural network architecture configuration.</param>
    /// <param name="sampleRate">Expected sample rate for input audio. Default is 16000.</param>
    /// <param name="embeddingDimension">Dimension of speaker embeddings. Default is 256.</param>
    /// <param name="defaultThreshold">Default verification threshold. Default is 0.6.</param>
    /// <param name="hiddenDim">Hidden dimension for encoder layers. Default is 256.</param>
    /// <param name="numEncoderLayers">Number of encoder layers. Default is 3.</param>
    /// <param name="numHeads">Number of attention heads. Default is 4.</param>
    /// <param name="optimizer">Optimizer for training. If null, AdamW is used.</param>
    /// <param name="lossFunction">Loss function for training. If null, MSE loss is used.</param>
    public SpeakerVerifier(
        NeuralNetworkArchitecture<T> architecture,
        int sampleRate = 16000,
        int embeddingDimension = 256,
        double defaultThreshold = 0.6,
        int hiddenDim = 256,
        int numEncoderLayers = 3,
        int numHeads = 4,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null)
        : base(architecture, lossFunction ?? new MeanSquaredErrorLoss<T>())
    {
        _useNativeMode = true;
        _embeddingModelPath = null;

        // Store parameters
        SampleRate = sampleRate;
        EmbeddingDimension = embeddingDimension;
        DefaultThreshold = NumOps.FromDouble(defaultThreshold);
        _hiddenDim = hiddenDim;
        _numEncoderLayers = numEncoderLayers;
        _numHeads = numHeads;

        // Create embedding extractor in native mode
        _embeddingExtractor = new SpeakerEmbeddingExtractor<T>(
            architecture,
            sampleRate,
            embeddingDimension,
            hiddenDim: hiddenDim,
            numEncoderLayers: numEncoderLayers,
            numHeads: numHeads);

        // Initialize enrolled speakers
        _enrolledSpeakers = new ConcurrentDictionary<string, SpeakerProfile<T>>();

        // Initialize optimizer and loss function
        _lossFunction = lossFunction ?? new MeanSquaredErrorLoss<T>();
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);

        // Initialize layers
        InitializeLayers();
    }

    #endregion

    #region Layer Initialization

    /// <summary>
    /// Initializes the layers for the speaker verifier.
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
            // Speaker verifier uses the embedding extractor layers
            // Add a simple verification head on top
            Layers.AddRange(LayerHelper<T>.CreateDefaultSpeakerEmbeddingLayers(
                numMels: 80,
                hiddenDim: _hiddenDim,
                embeddingDim: EmbeddingDimension,
                numLayers: _numEncoderLayers,
                maxFrames: 1000,
                dropoutRate: 0.0));
        }
    }

    #endregion

    #region ISpeakerVerifier Implementation

    /// <summary>
    /// Verifies if audio matches a reference speaker embedding.
    /// </summary>
    public SpeakerVerificationResult<T> Verify(Tensor<T> audio, Tensor<T> referenceEmbedding)
    {
        return Verify(audio, referenceEmbedding, DefaultThreshold);
    }

    /// <summary>
    /// Verifies if audio matches a reference speaker embedding with custom threshold.
    /// </summary>
    public SpeakerVerificationResult<T> Verify(Tensor<T> audio, Tensor<T> referenceEmbedding, T threshold)
    {
        ThrowIfDisposed();

        // Extract embedding from test audio
        var testEmbedding = _embeddingExtractor.ExtractEmbedding(audio);

        // Compute similarity
        var score = ComputeCosineSimilarity(testEmbedding, referenceEmbedding);

        // Make decision (score >= threshold)
        var isAccepted = NumOps.ToDouble(score) >= NumOps.ToDouble(threshold);

        // Compute confidence (distance from threshold)
        var scoreDiff = NumOps.Subtract(score, threshold);
        var confidence = NumOps.Abs(scoreDiff);

        return new SpeakerVerificationResult<T>
        {
            IsAccepted = isAccepted,
            Score = score,
            Threshold = threshold,
            Confidence = confidence
        };
    }

    /// <summary>
    /// Verifies if audio matches reference audio of a claimed speaker.
    /// </summary>
    public SpeakerVerificationResult<T> VerifyWithReferenceAudio(Tensor<T> audio, Tensor<T> referenceAudio)
    {
        ThrowIfDisposed();

        // Extract embedding from reference audio
        var referenceEmbedding = _embeddingExtractor.ExtractEmbedding(referenceAudio);

        // Verify against the extracted embedding
        return Verify(audio, referenceEmbedding);
    }

    /// <summary>
    /// Verifies if audio matches a reference speaker embedding asynchronously.
    /// </summary>
    public Task<SpeakerVerificationResult<T>> VerifyAsync(
        Tensor<T> audio,
        Tensor<T> referenceEmbedding,
        CancellationToken cancellationToken = default)
    {
        return Task.Run(() => Verify(audio, referenceEmbedding), cancellationToken);
    }

    /// <summary>
    /// Enrolls a speaker by creating a reference embedding from audio samples.
    /// </summary>
    public SpeakerProfile<T> Enroll(IReadOnlyList<Tensor<T>> enrollmentAudio)
    {
        ThrowIfDisposed();

        if (enrollmentAudio.Count == 0)
            throw new ArgumentException("At least one audio sample required for enrollment.");

        // Extract embeddings from all samples
        var embeddings = enrollmentAudio.Select(a => _embeddingExtractor.ExtractEmbedding(a)).ToList();

        // Aggregate embeddings
        var aggregatedEmbedding = ((ISpeakerEmbeddingExtractor<T>)_embeddingExtractor).AggregateEmbeddings(embeddings);

        // Compute total duration
        double totalDuration = enrollmentAudio.Sum(a => (double)a.Length / SampleRate);

        return new SpeakerProfile<T>
        {
            SpeakerId = Guid.NewGuid().ToString(),
            Embedding = aggregatedEmbedding,
            NumEnrollmentSamples = enrollmentAudio.Count,
            TotalEnrollmentDuration = totalDuration,
            CreatedAt = DateTime.UtcNow,
            UpdatedAt = DateTime.UtcNow
        };
    }

    /// <summary>
    /// Enrolls a speaker by creating a reference embedding from a single audio sample.
    /// </summary>
    public SpeakerProfile<T> Enroll(Tensor<T> enrollmentAudio)
    {
        return Enroll([enrollmentAudio]);
    }

    /// <summary>
    /// Updates an existing speaker profile with additional audio.
    /// </summary>
    public SpeakerProfile<T> UpdateProfile(SpeakerProfile<T> existingProfile, Tensor<T> newAudio)
    {
        ThrowIfDisposed();

        // Extract new embedding
        var newEmbedding = _embeddingExtractor.ExtractEmbedding(newAudio);

        // Aggregate with existing
        var aggregatedEmbedding = ((ISpeakerEmbeddingExtractor<T>)_embeddingExtractor).AggregateEmbeddings([existingProfile.Embedding, newEmbedding]);

        return new SpeakerProfile<T>
        {
            SpeakerId = existingProfile.SpeakerId,
            Embedding = aggregatedEmbedding,
            NumEnrollmentSamples = existingProfile.NumEnrollmentSamples + 1,
            TotalEnrollmentDuration = existingProfile.TotalEnrollmentDuration + ((double)newAudio.Length / SampleRate),
            CreatedAt = existingProfile.CreatedAt,
            UpdatedAt = DateTime.UtcNow
        };
    }

    /// <summary>
    /// Computes the verification score between audio and a reference.
    /// </summary>
    public T ComputeScore(Tensor<T> audio, Tensor<T> referenceEmbedding)
    {
        ThrowIfDisposed();

        var testEmbedding = _embeddingExtractor.ExtractEmbedding(audio);
        return ComputeCosineSimilarity(testEmbedding, referenceEmbedding);
    }

    /// <summary>
    /// Gets the recommended threshold for a target false accept rate.
    /// </summary>
    public T GetThresholdForFAR(double targetFAR)
    {
        // Approximate threshold based on typical speaker verification systems
        // Higher FAR tolerance = lower threshold
        double threshold = 0.85 - Math.Log10(1.0 / targetFAR) * 0.1;
        threshold = Math.Max(0.3, Math.Min(0.95, threshold));
        return NumOps.FromDouble(threshold);
    }

    #endregion

    #region AudioNeuralNetworkBase Implementation

    /// <summary>
    /// Preprocesses raw audio for model input.
    /// </summary>
    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio)
    {
        if (MfccExtractor is not null)
        {
            return MfccExtractor.Extract(rawAudio);
        }
        return rawAudio;
    }

    /// <summary>
    /// Postprocesses model output into the final result format.
    /// </summary>
    protected override Tensor<T> PostprocessOutput(Tensor<T> modelOutput)
    {
        return NormalizeEmbedding(modelOutput);
    }

    /// <summary>
    /// Makes a prediction using the model.
    /// </summary>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        // For speaker verifier, prediction returns the embedding
        return _embeddingExtractor.ExtractEmbedding(input);
    }

    /// <summary>
    /// Updates model parameters.
    /// </summary>
    public override void UpdateParameters(Vector<T> parameters)
    {
        if (!_useNativeMode)
        {
            throw new NotSupportedException("Cannot update parameters in ONNX inference mode.");
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
    /// Trains the model on input data.
    /// </summary>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        if (!_useNativeMode)
        {
            throw new NotSupportedException("Cannot train in ONNX inference mode. Use the native training constructor.");
        }

        // 1. Set training mode
        SetTrainingMode(true);

        // 2. Forward pass
        var prediction = Predict(input);

        // 3. Convert tensors to vectors for loss calculation
        var flatPrediction = prediction.ToVector();
        var flatExpected = expectedOutput.ToVector();

        // 4. Compute loss
        LastLoss = _lossFunction.CalculateLoss(flatPrediction, flatExpected);

        // 5. Compute gradients via backpropagation
        var lossGradient = _lossFunction.CalculateDerivative(flatPrediction, flatExpected);
        Backpropagate(Tensor<T>.FromVector(lossGradient));

        // 6. Update parameters using optimizer
        _optimizer?.UpdateParameters(Layers);

        // 7. Exit training mode
        SetTrainingMode(false);
    }

    /// <summary>
    /// Gets metadata about the model.
    /// </summary>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "SpeakerVerifier-Native" : "SpeakerVerifier-ONNX",
            Description = "Speaker verification model for voice authentication",
            ModelType = ModelType.NeuralNetwork,
            FeatureCount = EmbeddingDimension,
            Complexity = 1
        };
        metadata.AdditionalInfo["InputFormat"] = $"Audio ({SampleRate}Hz)";
        metadata.AdditionalInfo["OutputFormat"] = "Verification Decision";
        metadata.AdditionalInfo["Mode"] = _useNativeMode ? "Native Training" : "ONNX Inference";
        return metadata;
    }

    /// <summary>
    /// Serializes network-specific data.
    /// </summary>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(SampleRate);
        writer.Write(EmbeddingDimension);
        writer.Write(NumOps.ToDouble(DefaultThreshold));
        writer.Write(_useNativeMode);
        writer.Write(_hiddenDim);
        writer.Write(_numEncoderLayers);
        writer.Write(_numHeads);
    }

    /// <summary>
    /// Deserializes network-specific data.
    /// </summary>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        SampleRate = reader.ReadInt32();
        EmbeddingDimension = reader.ReadInt32();
        _ = reader.ReadDouble(); // DefaultThreshold
        _ = reader.ReadBoolean(); // useNativeMode
        _ = reader.ReadInt32(); // hiddenDim
        _ = reader.ReadInt32(); // numEncoderLayers
        _ = reader.ReadInt32(); // numHeads
    }

    /// <summary>
    /// Creates a new instance of this model for cloning.
    /// </summary>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        if (!_useNativeMode && _embeddingModelPath is not null)
        {
            return new SpeakerVerifier<T>(
                Architecture,
                _embeddingModelPath,
                SampleRate,
                EmbeddingDimension,
                NumOps.ToDouble(DefaultThreshold));
        }
        else
        {
            return new SpeakerVerifier<T>(
                Architecture,
                SampleRate,
                EmbeddingDimension,
                NumOps.ToDouble(DefaultThreshold),
                _hiddenDim,
                _numEncoderLayers,
                _numHeads,
                lossFunction: _lossFunction);
        }
    }

    #endregion

    #region Legacy API Support

    /// <summary>
    /// Enrolls a speaker with one or more embeddings (legacy API).
    /// </summary>
    public void Enroll(string speakerId, params SpeakerEmbedding<T>[] embeddings)
    {
        if (embeddings.Length == 0)
            throw new ArgumentException("At least one embedding required for enrollment.", nameof(embeddings));

        // Convert to tensor embeddings and aggregate
        var tensorEmbeddings = embeddings.Select(e =>
        {
            var tensor = new Tensor<T>([e.Vector.Length]);
            for (int i = 0; i < e.Vector.Length; i++)
            {
                tensor[i] = e.Vector[i];
            }
            return tensor;
        }).ToList();

        var aggregatedEmbedding = ((ISpeakerEmbeddingExtractor<T>)_embeddingExtractor).AggregateEmbeddings(tensorEmbeddings);

        var profile = new SpeakerProfile<T>
        {
            SpeakerId = speakerId,
            Embedding = aggregatedEmbedding,
            NumEnrollmentSamples = embeddings.Length,
            TotalEnrollmentDuration = embeddings.Sum(e => e.Duration),
            CreatedAt = DateTime.UtcNow,
            UpdatedAt = DateTime.UtcNow
        };

        _enrolledSpeakers[speakerId] = profile;
    }

    /// <summary>
    /// Verifies if a test embedding matches an enrolled speaker (legacy API).
    /// </summary>
    public VerificationResult Verify(string speakerId, SpeakerEmbedding<T> testEmbedding)
    {
        if (!_enrolledSpeakers.TryGetValue(speakerId, out var profile))
        {
            return new VerificationResult
            {
                ClaimedSpeakerId = speakerId,
                IsVerified = false,
                Score = 0,
                Threshold = NumOps.ToDouble(DefaultThreshold),
                ErrorMessage = "Speaker not enrolled"
            };
        }

        // Convert test embedding to tensor
        var testTensor = new Tensor<T>([testEmbedding.Vector.Length]);
        for (int i = 0; i < testEmbedding.Vector.Length; i++)
        {
            testTensor[i] = testEmbedding.Vector[i];
        }

        // Compute similarity
        double score = NumOps.ToDouble(ComputeCosineSimilarity(testTensor, profile.Embedding));

        return new VerificationResult
        {
            ClaimedSpeakerId = speakerId,
            IsVerified = score >= NumOps.ToDouble(DefaultThreshold),
            Score = score,
            Threshold = NumOps.ToDouble(DefaultThreshold)
        };
    }

    /// <summary>
    /// Identifies the most likely speaker from enrolled set (legacy API).
    /// </summary>
    public IdentificationResult Identify(SpeakerEmbedding<T> testEmbedding)
    {
        var scores = new List<(string speakerId, double score)>();

        // Convert test embedding to tensor
        var testTensor = new Tensor<T>([testEmbedding.Vector.Length]);
        for (int i = 0; i < testEmbedding.Vector.Length; i++)
        {
            testTensor[i] = testEmbedding.Vector[i];
        }

        foreach (var (speakerId, profile) in _enrolledSpeakers)
        {
            double score = NumOps.ToDouble(ComputeCosineSimilarity(testTensor, profile.Embedding));
            scores.Add((speakerId, score));
        }

        var ranked = scores.OrderByDescending(s => s.score).ToList();

        var result = new IdentificationResult
        {
            Matches = ranked.Select(s => new SpeakerMatch
            {
                SpeakerId = s.speakerId,
                Score = s.score
            }).ToList(),
            Threshold = NumOps.ToDouble(DefaultThreshold)
        };

        if (ranked.Count > 0 && ranked[0].score >= NumOps.ToDouble(DefaultThreshold))
        {
            result.IdentifiedSpeakerId = ranked[0].speakerId;
            result.TopScore = ranked[0].score;
        }

        return result;
    }

    /// <summary>
    /// Checks if a speaker is enrolled.
    /// </summary>
    public bool IsEnrolled(string speakerId) => _enrolledSpeakers.ContainsKey(speakerId);

    /// <summary>
    /// Gets all enrolled speaker IDs.
    /// </summary>
    public IReadOnlyList<string> GetEnrolledSpeakers() => _enrolledSpeakers.Keys.ToList();

    /// <summary>
    /// Removes a speaker's enrollment.
    /// </summary>
    public bool Unenroll(string speakerId)
    {
        return _enrolledSpeakers.TryRemove(speakerId, out _);
    }

    #endregion

    #region Private Methods

    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName);
    }

    #endregion

    #region IDisposable

    /// <summary>
    /// Disposes the model and releases resources.
    /// </summary>
    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;

        if (disposing)
        {
            _embeddingExtractor?.Dispose();
        }

        _disposed = true;
        base.Dispose(disposing);
    }

    #endregion
}

/// <summary>
/// Represents an enrolled speaker (legacy API).
/// </summary>
internal class EnrolledSpeaker<T>
{
    public string SpeakerId { get; set; } = string.Empty;
    public required SpeakerEmbedding<T> Centroid { get; set; }
    public List<SpeakerEmbedding<T>> Embeddings { get; set; } = [];
    public DateTime EnrollmentTime { get; set; }
}
