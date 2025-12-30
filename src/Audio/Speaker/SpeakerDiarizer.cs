using AiDotNet.Interfaces;
using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Audio.Speaker;

/// <summary>
/// Performs speaker diarization (who spoke when) on audio recordings.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Speaker diarization segments audio by speaker, answering "who spoke when?"
/// It uses embeddings from sliding windows and clustering to identify speaker turns.
/// </para>
/// <para>
/// This class supports both:
/// <list type="bullet">
/// <item><b>ONNX mode</b>: Load pre-trained models for fast inference</item>
/// <item><b>Native training mode</b>: Train from scratch using the layer architecture</item>
/// </list>
/// </para>
/// <para><b>For Beginners:</b> Diarization is like automatically labeling a meeting
/// recording with "Speaker A: 0:00-0:15, Speaker B: 0:15-0:45..."
///
/// The process:
/// 1. Split audio into short segments
/// 2. Extract speaker embeddings for each segment
/// 3. Cluster similar embeddings together
/// 4. Each cluster represents a different speaker
///
/// Common applications:
/// - Meeting transcription
/// - Call center analytics
/// - Podcast processing
///
/// Usage:
/// <code>
/// // ONNX mode (recommended for inference)
/// var diarizer = new SpeakerDiarizer&lt;float&gt;(architecture, modelPath);
/// var result = diarizer.Diarize(audioTensor);
///
/// // Native training mode
/// var diarizer = new SpeakerDiarizer&lt;float&gt;(architecture);
/// diarizer.Train(features, labels);
/// </code>
/// </para>
/// </remarks>
public class SpeakerDiarizer<T> : SpeakerRecognitionBase<T>, ISpeakerDiarizer<T>
{
    #region Fields

    private readonly SpeakerDiarizerOptions _options;
    private readonly SpeakerEmbeddingExtractor<T> _embeddingExtractor;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly bool _useNativeMode;
    private bool _disposed;

    #endregion

    #region Properties

    /// <summary>
    /// Gets the sample rate.
    /// </summary>
    public new int SampleRate => _options.SampleRate;

    /// <summary>
    /// Gets the minimum segment duration in seconds.
    /// </summary>
    public double MinSegmentDuration => _options.MinTurnDuration;

    /// <summary>
    /// Gets the minimum turn duration in seconds.
    /// </summary>
    /// <remarks>Legacy API - use MinSegmentDuration instead.</remarks>
    public double MinTurnDuration => _options.MinTurnDuration;

    /// <summary>
    /// Gets the clustering threshold.
    /// </summary>
    public double ClusteringThreshold => _options.ClusteringThreshold;

    /// <summary>
    /// Gets whether this model can detect overlapping speech.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Overlapping speech is when two or more people
    /// talk at the same time. This implementation currently does not support overlap detection.
    /// </para>
    /// </remarks>
    public bool SupportsOverlapDetection => false;

    /// <summary>
    /// Gets whether the model is operating in ONNX inference mode.
    /// </summary>
    public new bool IsOnnxMode => OnnxEncoder is not null;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a new speaker diarizer in ONNX inference mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture configuration.</param>
    /// <param name="modelPath">Path to the ONNX model file.</param>
    /// <param name="options">Diarization options.</param>
    /// <exception cref="ArgumentNullException">Thrown when modelPath is null.</exception>
    /// <exception cref="FileNotFoundException">Thrown when the model file doesn't exist.</exception>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this constructor for production inference with pre-trained models.
    /// ONNX models are optimized for fast execution on various hardware.
    /// </para>
    /// </remarks>
    public SpeakerDiarizer(
        NeuralNetworkArchitecture<T> architecture,
        string modelPath,
        SpeakerDiarizerOptions? options = null)
        : base(architecture)
    {
        ArgumentNullException.ThrowIfNull(modelPath, nameof(modelPath));

        if (!File.Exists(modelPath))
        {
            throw new FileNotFoundException($"ONNX model not found: {modelPath}", modelPath);
        }

        _options = options ?? new SpeakerDiarizerOptions();
        _useNativeMode = false;

        // Load ONNX model
        OnnxEncoder = new OnnxModel<T>(modelPath, new OnnxModelOptions());
        base.SampleRate = _options.SampleRate;
        EmbeddingDimension = _options.EmbeddingDimension;

        // Create embedding extractor in ONNX mode (using same model or a separate one)
        var extractorArch = new NeuralNetworkArchitecture<T>(
            inputFeatures: _options.EmbeddingDimension,
            outputSize: _options.EmbeddingDimension);

        if (_options.EmbeddingModelPath is not null && _options.EmbeddingModelPath.Length > 0)
        {
            _embeddingExtractor = new SpeakerEmbeddingExtractor<T>(
                extractorArch,
                modelPath: _options.EmbeddingModelPath,
                sampleRate: _options.SampleRate,
                embeddingDimension: _options.EmbeddingDimension);
        }
        else
        {
            // Use the main model path for embeddings if no separate embedding model
            _embeddingExtractor = new SpeakerEmbeddingExtractor<T>(
                extractorArch,
                modelPath: modelPath,
                sampleRate: _options.SampleRate,
                embeddingDimension: _options.EmbeddingDimension);
        }

        // Optimizer not used in ONNX mode but required by interface
        _optimizer = new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);

        InitializeLayers();
    }

    /// <summary>
    /// Creates a new speaker diarizer in native training mode.
    /// </summary>
    /// <param name="architecture">Neural network architecture configuration.</param>
    /// <param name="options">Diarization options.</param>
    /// <param name="optimizer">Optional custom optimizer (defaults to AdamW).</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Use this constructor when you want to train a model from scratch
    /// or fine-tune an existing model. Training requires labeled diarization data.
    /// </para>
    /// </remarks>
    public SpeakerDiarizer(
        NeuralNetworkArchitecture<T> architecture,
        SpeakerDiarizerOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new SpeakerDiarizerOptions();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);

        base.SampleRate = _options.SampleRate;
        EmbeddingDimension = _options.EmbeddingDimension;

        // Create embedding extractor in native mode
        var extractorArch = new NeuralNetworkArchitecture<T>(
            inputFeatures: _options.EmbeddingDimension,
            outputSize: _options.EmbeddingDimension);

        _embeddingExtractor = new SpeakerEmbeddingExtractor<T>(
            extractorArch,
            sampleRate: _options.SampleRate,
            embeddingDimension: _options.EmbeddingDimension);

        InitializeLayers();
    }

    /// <summary>
    /// Creates a new speaker diarizer with legacy options only.
    /// </summary>
    /// <param name="options">Diarization options.</param>
    /// <remarks>
    /// <para>
    /// <b>Legacy API:</b> Prefer the constructors with NeuralNetworkArchitecture parameter.
    /// This constructor creates a default architecture for backward compatibility.
    /// </para>
    /// </remarks>
    public SpeakerDiarizer(SpeakerDiarizerOptions? options = null)
        : this(
            new NeuralNetworkArchitecture<T>(
                inputFeatures: (options ?? new SpeakerDiarizerOptions()).EmbeddingDimension,
                outputSize: (options ?? new SpeakerDiarizerOptions()).EmbeddingDimension),
            options)
    {
    }

    #endregion

    #region Initialization

    /// <summary>
    /// Initializes the neural network layers.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This follows the golden standard pattern:
    /// 1. If in ONNX mode, layers are not needed (inference uses ONNX runtime)
    /// 2. If Architecture.Layers is provided, use those layers
    /// 3. Otherwise, fall back to LayerHelper.CreateDefaultSpeakerEmbeddingLayers()
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Layers are only initialized in native training mode.
    /// In ONNX mode, the model is already fully trained and ready for inference.
    /// </para>
    /// </remarks>
    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;

        if (Architecture.Layers is not null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
        }
        else
        {
            Layers.AddRange(LayerHelper<T>.CreateDefaultSpeakerEmbeddingLayers(
                numMels: 40, // MFCC coefficients
                hiddenDim: 512,
                embeddingDim: EmbeddingDimension,
                numLayers: 3,
                dropoutRate: 0.1));
        }
    }

    #endregion

    #region ISpeakerDiarizer Implementation

    /// <summary>
    /// Performs speaker diarization on audio.
    /// </summary>
    /// <param name="audio">Audio waveform tensor [samples].</param>
    /// <param name="numSpeakers">Expected number of speakers. Auto-detected if null.</param>
    /// <param name="minSpeakers">Minimum number of speakers (for auto-detection).</param>
    /// <param name="maxSpeakers">Maximum number of speakers (for auto-detection).</param>
    /// <returns>Diarization result with speaker segments.</returns>
    public DiarizationResult<T> Diarize(
        Tensor<T> audio,
        int? numSpeakers = null,
        int minSpeakers = 1,
        int maxSpeakers = 10)
    {
        ThrowIfDisposed();

        // Override options if provided
        int? effectiveMaxSpeakers = numSpeakers ?? _options.MaxSpeakers ?? maxSpeakers;

        // Segment audio into windows
        var segments = SegmentAudio(audio);

        // Extract embeddings for each segment
        var embeddings = ExtractSegmentEmbeddings(segments);

        // Cluster embeddings
        var labels = ClusterEmbeddings(embeddings, effectiveMaxSpeakers);

        // Merge consecutive same-speaker segments
        var speakerTurns = CreateSpeakerTurns(segments, labels);

        // Convert to DiarizationResult<T> format
        var speakerSegments = speakerTurns.Select(turn => new SpeakerSegment<T>
        {
            Speaker = turn.SpeakerId,
            StartTime = turn.StartTime,
            EndTime = turn.EndTime,
            Confidence = NumOps.FromDouble(1.0) // Clustering doesn't provide per-segment confidence
        }).ToList();

        var speakerLabels = speakerSegments.Select(s => s.Speaker).Distinct().ToList();
        var stats = ComputeSpeakerStatistics(speakerSegments, (double)audio.Length / _options.SampleRate);

        return new DiarizationResult<T>
        {
            Segments = speakerSegments,
            NumSpeakers = labels.Distinct().Count(),
            SpeakerLabels = speakerLabels,
            TotalDuration = (double)audio.Length / _options.SampleRate,
            OverlapRegions = Array.Empty<OverlapRegion<T>>(), // Overlap detection not supported yet
            SpeakerStats = stats
        };
    }

    /// <summary>
    /// Performs speaker diarization asynchronously.
    /// </summary>
    public Task<DiarizationResult<T>> DiarizeAsync(
        Tensor<T> audio,
        int? numSpeakers = null,
        int minSpeakers = 1,
        int maxSpeakers = 10,
        CancellationToken cancellationToken = default)
    {
        return Task.Run(() =>
        {
            cancellationToken.ThrowIfCancellationRequested();
            return Diarize(audio, numSpeakers, minSpeakers, maxSpeakers);
        }, cancellationToken);
    }

    /// <summary>
    /// Performs diarization with known speaker profiles.
    /// </summary>
    public DiarizationResult<T> DiarizeWithKnownSpeakers(
        Tensor<T> audio,
        IReadOnlyList<SpeakerProfile<T>> knownSpeakers,
        bool allowUnknownSpeakers = true)
    {
        ThrowIfDisposed();

        // First perform standard diarization
        var segments = SegmentAudio(audio);
        var embeddings = ExtractSegmentEmbeddings(segments);

        // Match each embedding to known speakers
        var labels = new string[embeddings.Count];
        int unknownCount = 0;

        for (int i = 0; i < embeddings.Count; i++)
        {
            var embedding = embeddings[i];
            string? bestMatch = null;
            double bestSimilarity = _options.ClusteringThreshold;

            foreach (var profile in knownSpeakers)
            {
                // Convert profile embedding to SpeakerEmbedding for comparison
                var profileEmbedding = new SpeakerEmbedding<T>
                {
                    Vector = profile.Embedding.ToArray()
                };
                double similarity = embedding.CosineSimilarity(profileEmbedding);
                if (similarity > bestSimilarity)
                {
                    bestSimilarity = similarity;
                    bestMatch = profile.SpeakerId;
                }
            }

            if (bestMatch is not null)
            {
                labels[i] = bestMatch;
            }
            else if (allowUnknownSpeakers)
            {
                labels[i] = $"Unknown_{unknownCount++}";
            }
            else
            {
                labels[i] = "Unknown";
            }
        }

        // Merge consecutive segments with same label
        var speakerSegments = MergeConsecutiveSegments(segments, labels);
        var speakerLabels = speakerSegments.Select(s => s.Speaker).Distinct().ToList();
        var stats = ComputeSpeakerStatistics(speakerSegments, (double)audio.Length / _options.SampleRate);

        return new DiarizationResult<T>
        {
            Segments = speakerSegments,
            NumSpeakers = speakerLabels.Count,
            SpeakerLabels = speakerLabels,
            TotalDuration = (double)audio.Length / _options.SampleRate,
            OverlapRegions = Array.Empty<OverlapRegion<T>>(),
            SpeakerStats = stats
        };
    }

    /// <summary>
    /// Gets speaker embeddings for each detected speaker.
    /// </summary>
    public IReadOnlyDictionary<string, Tensor<T>> ExtractSpeakerEmbeddings(
        Tensor<T> audio,
        DiarizationResult<T> diarizationResult)
    {
        ThrowIfDisposed();

        var result = new Dictionary<string, Tensor<T>>();

        foreach (var speakerLabel in diarizationResult.SpeakerLabels)
        {
            var speakerSegments = diarizationResult.Segments
                .Where(s => s.Speaker == speakerLabel)
                .ToList();

            if (speakerSegments.Count == 0) continue;

            var speakerEmbeddings = new List<Tensor<T>>();

            foreach (var segment in speakerSegments)
            {
                int startSample = (int)(segment.StartTime * _options.SampleRate);
                int endSample = Math.Min((int)(segment.EndTime * _options.SampleRate), audio.Length);
                int segmentLength = endSample - startSample;

                if (segmentLength < _options.SampleRate * 0.5) continue; // Skip very short segments

                var segmentAudio = new Tensor<T>([segmentLength]);
                for (int i = 0; i < segmentLength; i++)
                {
                    segmentAudio[i] = audio[startSample + i];
                }

                var embedding = _embeddingExtractor.Extract(segmentAudio);
                // Convert embedding Vector to Tensor
                var embeddingTensor = new Tensor<T>([embedding.Vector.Length]);
                for (int j = 0; j < embedding.Vector.Length; j++)
                {
                    embeddingTensor[j] = embedding.Vector[j];
                }
                speakerEmbeddings.Add(embeddingTensor);
            }

            if (speakerEmbeddings.Count > 0)
            {
                result[speakerLabel] = AggregateEmbeddings(speakerEmbeddings);
            }
        }

        return result;
    }

    /// <summary>
    /// Refines diarization result by re-segmenting with different parameters.
    /// </summary>
    public DiarizationResult<T> RefineDiarization(
        Tensor<T> audio,
        DiarizationResult<T> previousResult,
        T mergeThreshold)
    {
        ThrowIfDisposed();

        // Re-cluster with the new threshold
        var segments = SegmentAudio(audio);
        var embeddings = ExtractSegmentEmbeddings(segments);

        double thresholdValue = NumOps.ToDouble(mergeThreshold);
        var labels = ClusterEmbeddingsWithThreshold(embeddings, thresholdValue);

        // Merge consecutive same-speaker segments
        var speakerTurns = CreateSpeakerTurns(segments, labels);

        var speakerSegments = speakerTurns.Select(turn => new SpeakerSegment<T>
        {
            Speaker = turn.SpeakerId,
            StartTime = turn.StartTime,
            EndTime = turn.EndTime,
            Confidence = NumOps.FromDouble(1.0)
        }).ToList();

        var speakerLabels = speakerSegments.Select(s => s.Speaker).Distinct().ToList();
        var stats = ComputeSpeakerStatistics(speakerSegments, (double)audio.Length / _options.SampleRate);

        return new DiarizationResult<T>
        {
            Segments = speakerSegments,
            NumSpeakers = labels.Distinct().Count(),
            SpeakerLabels = speakerLabels,
            TotalDuration = (double)audio.Length / _options.SampleRate,
            OverlapRegions = Array.Empty<OverlapRegion<T>>(),
            SpeakerStats = stats
        };
    }

    #endregion

    #region Legacy API Support

    /// <summary>
    /// Performs diarization on audio (legacy API).
    /// </summary>
    /// <param name="audio">Audio samples as a tensor.</param>
    /// <returns>Legacy diarization result.</returns>
    /// <remarks>
    /// <b>Legacy API:</b> Prefer using <see cref="Diarize(Tensor{T}, int?, int, int)"/> instead.
    /// </remarks>
    public DiarizationResult DiarizeLegacy(Tensor<T> audio)
    {
        var result = Diarize(audio);
        return ConvertToLegacyResult(result);
    }

    /// <summary>
    /// Performs diarization on audio (legacy API).
    /// </summary>
    /// <param name="audio">Audio samples as a vector.</param>
    /// <returns>Legacy diarization result.</returns>
    public DiarizationResult DiarizeLegacy(Vector<T> audio)
    {
        var tensor = new Tensor<T>([audio.Length]);
        for (int i = 0; i < audio.Length; i++)
        {
            tensor[i] = audio[i];
        }
        return DiarizeLegacy(tensor);
    }

    private static DiarizationResult ConvertToLegacyResult(DiarizationResult<T> result)
    {
        var turns = result.Segments.Select((seg, idx) => new SpeakerTurn
        {
            SpeakerId = seg.Speaker,
            SpeakerIndex = result.SpeakerLabels.ToList().IndexOf(seg.Speaker),
            StartTime = seg.StartTime,
            EndTime = seg.EndTime
        }).ToList();

        return new DiarizationResult
        {
            Turns = turns,
            NumSpeakers = result.NumSpeakers,
            Duration = result.TotalDuration
        };
    }

    #endregion

    #region NeuralNetworkBase Implementation

    /// <summary>
    /// Predicts output for the given input.
    /// </summary>
    /// <param name="input">Input tensor (audio features).</param>
    /// <returns>Output tensor (speaker probabilities per frame).</returns>
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
    /// <param name="parameters">Parameter vector.</param>
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
    /// <param name="input">Input features.</param>
    /// <param name="expected">Expected output.</param>
    public override void Train(Tensor<T> input, Tensor<T> expected)
    {
        if (IsOnnxMode)
        {
            throw new NotSupportedException(
                "Training is not supported in ONNX mode. Create a new SpeakerDiarizer " +
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
        var gradientTensor = new Tensor<T>([gradient.Length]);
        for (int i = 0; i < gradient.Length; i++)
        {
            gradientTensor[i] = gradient[i];
        }

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

    #region Private Methods - Audio Processing

    private List<AudioSegment<T>> SegmentAudio(Tensor<T> audio)
    {
        var segments = new List<AudioSegment<T>>();
        int samplesPerWindow = (int)(_options.WindowDurationSeconds * _options.SampleRate);
        int hopSamples = (int)(_options.HopDurationSeconds * _options.SampleRate);

        for (int start = 0; start + samplesPerWindow <= audio.Length; start += hopSamples)
        {
            var segment = new Tensor<T>([samplesPerWindow]);
            for (int i = 0; i < samplesPerWindow; i++)
            {
                segment[i] = audio[start + i];
            }

            segments.Add(new AudioSegment<T>
            {
                Audio = segment,
                StartSample = start,
                StartTime = start / (double)_options.SampleRate,
                EndTime = (start + samplesPerWindow) / (double)_options.SampleRate
            });
        }

        return segments;
    }

    private List<SpeakerEmbedding<T>> ExtractSegmentEmbeddings(List<AudioSegment<T>> segments)
    {
        return segments.Select(s => _embeddingExtractor.Extract(s.Audio)).ToList();
    }

    #endregion

    #region Private Methods - Clustering

    private int[] ClusterEmbeddings(List<SpeakerEmbedding<T>> embeddings, int? maxSpeakers)
    {
        if (embeddings.Count == 0)
            return [];

        // Compute similarity matrix
        int n = embeddings.Count;
        var similarityMatrix = new double[n, n];

        for (int i = 0; i < n; i++)
        {
            for (int j = i; j < n; j++)
            {
                double sim = embeddings[i].CosineSimilarity(embeddings[j]);
                similarityMatrix[i, j] = sim;
                similarityMatrix[j, i] = sim;
            }
        }

        // Agglomerative clustering
        var labels = AgglomerativeCluster(similarityMatrix, _options.ClusteringThreshold);

        // Apply constraints (max speakers)
        int numClusters = labels.Distinct().Count();
        if (maxSpeakers.HasValue && numClusters > maxSpeakers.Value)
        {
            labels = MergeClusters(labels, similarityMatrix, maxSpeakers.Value);
        }

        return labels;
    }

    private int[] ClusterEmbeddingsWithThreshold(List<SpeakerEmbedding<T>> embeddings, double threshold)
    {
        if (embeddings.Count == 0)
            return [];

        int n = embeddings.Count;
        var similarityMatrix = new double[n, n];

        for (int i = 0; i < n; i++)
        {
            for (int j = i; j < n; j++)
            {
                double sim = embeddings[i].CosineSimilarity(embeddings[j]);
                similarityMatrix[i, j] = sim;
                similarityMatrix[j, i] = sim;
            }
        }

        return AgglomerativeCluster(similarityMatrix, threshold);
    }

    private static int[] AgglomerativeCluster(double[,] similarity, double threshold)
    {
        int n = similarity.GetLength(0);
        var labels = new int[n];
        var clusterMembers = new Dictionary<int, List<int>>();

        // Initialize: each point is its own cluster
        for (int i = 0; i < n; i++)
        {
            labels[i] = i;
            clusterMembers[i] = [i];
        }

        bool merged = true;
        while (merged)
        {
            merged = false;
            double bestSim = threshold;
            int best_i = -1, best_j = -1;

            // Find most similar pair of clusters
            var activeLabels = labels.Distinct().ToList();
            for (int li = 0; li < activeLabels.Count; li++)
            {
                for (int lj = li + 1; lj < activeLabels.Count; lj++)
                {
                    int label_i = activeLabels[li];
                    int label_j = activeLabels[lj];

                    // Average linkage
                    double avgSim = 0;
                    int count = 0;

                    foreach (int mi in clusterMembers[label_i])
                    {
                        foreach (int mj in clusterMembers[label_j])
                        {
                            avgSim += similarity[mi, mj];
                            count++;
                        }
                    }

                    if (count > 0)
                    {
                        avgSim /= count;
                        if (avgSim > bestSim)
                        {
                            bestSim = avgSim;
                            best_i = label_i;
                            best_j = label_j;
                        }
                    }
                }
            }

            // Merge best pair if above threshold
            if (best_i >= 0 && best_j >= 0)
            {
                // Merge j into i
                foreach (int member in clusterMembers[best_j])
                {
                    labels[member] = best_i;
                    clusterMembers[best_i].Add(member);
                }
                clusterMembers.Remove(best_j);
                merged = true;
            }
        }

        // Relabel to consecutive integers
        var labelMap = labels.Distinct().OrderBy(l => l)
            .Select((l, i) => (l, i))
            .ToDictionary(x => x.l, x => x.i);

        for (int i = 0; i < n; i++)
        {
            labels[i] = labelMap[labels[i]];
        }

        return labels;
    }

    private static int[] MergeClusters(int[] labels, double[,] similarity, int maxClusters)
    {
        while (labels.Distinct().Count() > maxClusters)
        {
            // Find two most similar clusters to merge
            var labelCounts = labels.GroupBy(l => l)
                .ToDictionary(g => g.Key, g => g.Count());
            var activeLabels = labelCounts.Keys.OrderBy(l => l).ToList();

            double bestSim = double.NegativeInfinity;
            int mergeFrom = -1, mergeTo = -1;

            for (int li = 0; li < activeLabels.Count; li++)
            {
                for (int lj = li + 1; lj < activeLabels.Count; lj++)
                {
                    int l1 = activeLabels[li];
                    int l2 = activeLabels[lj];

                    // Compute average similarity between clusters
                    double avgSim = 0;
                    int count = 0;

                    for (int i = 0; i < labels.Length; i++)
                    {
                        if (labels[i] != l1) continue;
                        for (int j = 0; j < labels.Length; j++)
                        {
                            if (labels[j] != l2) continue;
                            avgSim += similarity[i, j];
                            count++;
                        }
                    }

                    if (count > 0)
                    {
                        avgSim /= count;
                        if (avgSim > bestSim)
                        {
                            bestSim = avgSim;
                            mergeFrom = l2;
                            mergeTo = l1;
                        }
                    }
                }
            }

            if (mergeFrom >= 0 && mergeTo >= 0)
            {
                for (int i = 0; i < labels.Length; i++)
                {
                    if (labels[i] == mergeFrom)
                        labels[i] = mergeTo;
                }
            }
            else
            {
                break; // No more merges possible
            }
        }

        // Relabel to consecutive integers
        var labelMap = labels.Distinct().OrderBy(l => l)
            .Select((l, i) => (l, i))
            .ToDictionary(x => x.l, x => x.i);

        for (int i = 0; i < labels.Length; i++)
        {
            labels[i] = labelMap[labels[i]];
        }

        return labels;
    }

    #endregion

    #region Private Methods - Results Processing

    private List<SpeakerTurn> CreateSpeakerTurns(List<AudioSegment<T>> segments, int[] labels)
    {
        var turns = new List<SpeakerTurn>();
        if (segments.Count == 0) return turns;

        int currentLabel = labels[0];
        double turnStart = segments[0].StartTime;
        double turnEnd = segments[0].EndTime;

        for (int i = 1; i < segments.Count; i++)
        {
            if (labels[i] == currentLabel)
            {
                // Extend current turn
                turnEnd = segments[i].EndTime;
            }
            else
            {
                // Save current turn and start new one
                if (turnEnd - turnStart >= _options.MinTurnDuration)
                {
                    turns.Add(new SpeakerTurn
                    {
                        SpeakerId = $"Speaker_{currentLabel + 1}",
                        SpeakerIndex = currentLabel,
                        StartTime = turnStart,
                        EndTime = turnEnd
                    });
                }

                currentLabel = labels[i];
                turnStart = segments[i].StartTime;
                turnEnd = segments[i].EndTime;
            }
        }

        // Add final turn
        if (turnEnd - turnStart >= _options.MinTurnDuration)
        {
            turns.Add(new SpeakerTurn
            {
                SpeakerId = $"Speaker_{currentLabel + 1}",
                SpeakerIndex = currentLabel,
                StartTime = turnStart,
                EndTime = turnEnd
            });
        }

        return turns;
    }

    private List<SpeakerSegment<T>> MergeConsecutiveSegments(List<AudioSegment<T>> segments, string[] labels)
    {
        var result = new List<SpeakerSegment<T>>();
        if (segments.Count == 0) return result;

        string currentLabel = labels[0];
        double turnStart = segments[0].StartTime;
        double turnEnd = segments[0].EndTime;

        for (int i = 1; i < segments.Count; i++)
        {
            if (labels[i] == currentLabel)
            {
                turnEnd = segments[i].EndTime;
            }
            else
            {
                if (turnEnd - turnStart >= _options.MinTurnDuration)
                {
                    result.Add(new SpeakerSegment<T>
                    {
                        Speaker = currentLabel,
                        StartTime = turnStart,
                        EndTime = turnEnd,
                        Confidence = NumOps.FromDouble(1.0)
                    });
                }

                currentLabel = labels[i];
                turnStart = segments[i].StartTime;
                turnEnd = segments[i].EndTime;
            }
        }

        if (turnEnd - turnStart >= _options.MinTurnDuration)
        {
            result.Add(new SpeakerSegment<T>
            {
                Speaker = currentLabel,
                StartTime = turnStart,
                EndTime = turnEnd,
                Confidence = NumOps.FromDouble(1.0)
            });
        }

        return result;
    }

    private IReadOnlyDictionary<string, SpeakerStatistics<T>> ComputeSpeakerStatistics(
        List<SpeakerSegment<T>> segments, double totalDuration)
    {
        var stats = new Dictionary<string, SpeakerStatistics<T>>();

        var grouped = segments.GroupBy(s => s.Speaker);
        foreach (var group in grouped)
        {
            var speakerSegments = group.ToList();
            double totalSpeakingTime = speakerSegments.Sum(s => s.Duration);
            int numTurns = speakerSegments.Count;

            stats[group.Key] = new SpeakerStatistics<T>
            {
                TotalSpeakingTime = totalSpeakingTime,
                NumTurns = numTurns,
                AverageTurnDuration = numTurns > 0 ? totalSpeakingTime / numTurns : 0,
                SpeakingPercentage = totalDuration > 0 ? (totalSpeakingTime / totalDuration) * 100 : 0
            };
        }

        return stats;
    }

    #endregion

    #region AudioNeuralNetworkBase Implementation

    /// <summary>
    /// Preprocesses raw audio for model input.
    /// </summary>
    /// <param name="rawAudio">Raw audio waveform.</param>
    /// <returns>Preprocessed audio features.</returns>
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
    /// <param name="modelOutput">Model output tensor.</param>
    /// <returns>Postprocessed output.</returns>
    protected override Tensor<T> PostprocessOutput(Tensor<T> modelOutput)
    {
        return NormalizeEmbedding(modelOutput);
    }

    /// <summary>
    /// Gets metadata about the model.
    /// </summary>
    /// <returns>Model metadata.</returns>
    public override ModelMetadata<T> GetModelMetadata()
    {
        var metadata = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "SpeakerDiarizer-Native" : "SpeakerDiarizer-ONNX",
            Description = "Speaker diarization model for identifying who spoke when",
            ModelType = ModelType.NeuralNetwork,
            FeatureCount = EmbeddingDimension,
            Complexity = 1
        };
        metadata.AdditionalInfo["SampleRate"] = _options.SampleRate.ToString();
        metadata.AdditionalInfo["EmbeddingDimension"] = EmbeddingDimension.ToString();
        metadata.AdditionalInfo["ClusteringThreshold"] = _options.ClusteringThreshold.ToString();
        metadata.AdditionalInfo["MinTurnDuration"] = _options.MinTurnDuration.ToString();
        metadata.AdditionalInfo["Mode"] = _useNativeMode ? "Native Training" : "ONNX Inference";
        return metadata;
    }

    /// <summary>
    /// Serializes network-specific data.
    /// </summary>
    /// <param name="writer">Binary writer.</param>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(SampleRate);
        writer.Write(EmbeddingDimension);
        writer.Write(_options.ClusteringThreshold);
        writer.Write(_options.MinTurnDuration);
        writer.Write(_options.WindowDurationSeconds);
        writer.Write(_options.HopDurationSeconds);
    }

    /// <summary>
    /// Deserializes network-specific data.
    /// </summary>
    /// <param name="reader">Binary reader.</param>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        base.SampleRate = reader.ReadInt32();
        EmbeddingDimension = reader.ReadInt32();
        // Note: Options are readonly, values loaded for reference
        _ = reader.ReadDouble(); // ClusteringThreshold
        _ = reader.ReadDouble(); // MinTurnDuration
        _ = reader.ReadDouble(); // WindowDurationSeconds
        _ = reader.ReadDouble(); // HopDurationSeconds
    }

    /// <summary>
    /// Creates a new instance of this model for cloning.
    /// </summary>
    /// <returns>New model instance.</returns>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        if (!_useNativeMode)
        {
            throw new NotSupportedException(
                "CreateNewInstance is not supported for ONNX models. " +
                "Create a new SpeakerDiarizer with the model path instead.");
        }

        return new SpeakerDiarizer<T>(
            Architecture,
            _options);
    }

    #endregion

    #region IDisposable

    private void ThrowIfDisposed()
    {
        ObjectDisposedException.ThrowIf(_disposed, GetType().FullName ?? nameof(SpeakerDiarizer<T>));
    }

    /// <summary>
    /// Disposes resources.
    /// </summary>
    public new void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Disposes managed resources.
    /// </summary>
    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;

        if (disposing)
        {
            _embeddingExtractor.Dispose();
            OnnxEncoder?.Dispose();
        }

        _disposed = true;
        base.Dispose(disposing);
    }

    #endregion
}

/// <summary>
/// Represents an audio segment for processing.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
internal class AudioSegment<T>
{
    /// <summary>
    /// Gets or sets the audio tensor for this segment.
    /// </summary>
    public required Tensor<T> Audio { get; set; }

    /// <summary>
    /// Gets or sets the start sample index.
    /// </summary>
    public int StartSample { get; set; }

    /// <summary>
    /// Gets or sets the start time in seconds.
    /// </summary>
    public double StartTime { get; set; }

    /// <summary>
    /// Gets or sets the end time in seconds.
    /// </summary>
    public double EndTime { get; set; }
}
