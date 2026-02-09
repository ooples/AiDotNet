using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.NeuralNetworks.Options;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Neural network for audio-visual event localization - identifying WHEN and WHERE events occur
/// in video by jointly analyzing audio and visual streams with precise temporal boundaries.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class AudioVisualEventLocalizationNetwork<T> : NeuralNetworkBase<T>, IAudioVisualEventLocalizationModel<T>
{
    private readonly AudioVisualEventLocalizationOptions _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    #region Constants

    private const int DEFAULT_EMBEDDING_DIM = 512;
    private const int SPECTROGRAM_BINS = 128;
    private const double DEFAULT_TEMPORAL_RESOLUTION = 0.1; // 100ms resolution
    private const int DEFAULT_NUM_CATEGORIES = 100;

    #endregion

    #region Fields

    private readonly INumericOperations<T> _numOps;
    private readonly int _embeddingDimension;
    private readonly double _temporalResolution;
    private readonly int _numEncoderLayers;
    private readonly IReadOnlyList<string> _supportedCategories;
    private readonly Random _random;
    private readonly IOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly ILossFunction<T> _lossFunction;

    // Audio encoder
    private DenseLayer<T> _audioInputProjection;
    private MultiHeadAttentionLayer<T>[] _audioEncoderLayers;
    private DenseLayer<T> _audioOutputProjection;

    // Visual encoder
    private DenseLayer<T> _visualInputProjection;
    private MultiHeadAttentionLayer<T>[] _visualEncoderLayers;
    private DenseLayer<T> _visualOutputProjection;

    // Temporal modeling
    private MultiHeadAttentionLayer<T>[] _temporalAttentionLayers;
    private DenseLayer<T> _temporalProposalHead;

    // Cross-modal fusion for event detection
    private MultiHeadAttentionLayer<T>[] _crossModalAttentionLayers;

    // Task-specific heads
    private DenseLayer<T> _eventClassificationHead;
    private DenseLayer<T> _temporalBoundaryHead;
    private DenseLayer<T> _spatialLocalizationHead;
    private DenseLayer<T> _anomalyDetectionHead;

    #endregion

    #region Properties

    /// <inheritdoc/>
    public double TemporalResolution => _temporalResolution;

    /// <inheritdoc/>
    public IReadOnlyList<string> SupportedEventCategories => _supportedCategories;

    #endregion

    #region Constructors

    /// <summary>
    /// Initializes a new instance of the AudioVisualEventLocalizationNetwork.
    /// </summary>
    public AudioVisualEventLocalizationNetwork(
        NeuralNetworkArchitecture<T> architecture,
        int embeddingDimension = DEFAULT_EMBEDDING_DIM,
        double temporalResolution = DEFAULT_TEMPORAL_RESOLUTION,
        int numEncoderLayers = 6,
        IEnumerable<string>? eventCategories = null,
        IOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        int? seed = null,
        AudioVisualEventLocalizationOptions? options = null)
        : base(architecture, lossFunction ?? new CrossEntropyLoss<T>(), 1.0)
    {
        _options = options ?? new AudioVisualEventLocalizationOptions();
        Options = _options;

        _numOps = MathHelper.GetNumericOperations<T>();
        _embeddingDimension = embeddingDimension;
        _temporalResolution = temporalResolution;
        _numEncoderLayers = numEncoderLayers;
        _random = seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomHelper.CreateSeededRandom(42);
        _lossFunction = lossFunction ?? new CrossEntropyLoss<T>();
        _optimizer = optimizer ?? new Optimizers.AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

        // Default event categories
        _supportedCategories = (eventCategories?.ToList() ?? GetDefaultEventCategories()).AsReadOnly();

        // Initialize all layers
        _audioInputProjection = null!;
        _audioEncoderLayers = null!;
        _audioOutputProjection = null!;
        _visualInputProjection = null!;
        _visualEncoderLayers = null!;
        _visualOutputProjection = null!;
        _temporalAttentionLayers = null!;
        _temporalProposalHead = null!;
        _crossModalAttentionLayers = null!;
        _eventClassificationHead = null!;
        _temporalBoundaryHead = null!;
        _spatialLocalizationHead = null!;
        _anomalyDetectionHead = null!;

        InitializeLayers();
    }

    #endregion

    #region Initialization

    /// <inheritdoc/>
    protected override void InitializeLayers()
    {
        IActivationFunction<T>? nullActivation = null;
        var geluActivation = new GELUActivation<T>() as IActivationFunction<T>;

        // Audio encoder
        _audioInputProjection = new DenseLayer<T>(SPECTROGRAM_BINS, _embeddingDimension, nullActivation);
        _audioEncoderLayers = new MultiHeadAttentionLayer<T>[_numEncoderLayers];
        for (int i = 0; i < _numEncoderLayers; i++)
        {
            // Constructor: (sequenceLength, embeddingDimension, headCount, activation)
            // Use sequenceLength=1 as placeholder since attention works with variable-length sequences
            _audioEncoderLayers[i] = new MultiHeadAttentionLayer<T>(
                1, _embeddingDimension, 8, geluActivation);
        }
        _audioOutputProjection = new DenseLayer<T>(_embeddingDimension, _embeddingDimension, nullActivation);

        // Visual encoder
        _visualInputProjection = new DenseLayer<T>(768, _embeddingDimension, nullActivation); // ViT-style input
        _visualEncoderLayers = new MultiHeadAttentionLayer<T>[_numEncoderLayers];
        for (int i = 0; i < _numEncoderLayers; i++)
        {
            _visualEncoderLayers[i] = new MultiHeadAttentionLayer<T>(
                1, _embeddingDimension, 8, geluActivation);
        }
        _visualOutputProjection = new DenseLayer<T>(_embeddingDimension, _embeddingDimension, nullActivation);

        // Temporal modeling
        _temporalAttentionLayers = new MultiHeadAttentionLayer<T>[4];
        for (int i = 0; i < 4; i++)
        {
            _temporalAttentionLayers[i] = new MultiHeadAttentionLayer<T>(
                1, _embeddingDimension, 8, geluActivation);
        }
        _temporalProposalHead = new DenseLayer<T>(_embeddingDimension, 2, nullActivation); // start, end offsets

        // Cross-modal fusion
        _crossModalAttentionLayers = new MultiHeadAttentionLayer<T>[4];
        for (int i = 0; i < 4; i++)
        {
            _crossModalAttentionLayers[i] = new MultiHeadAttentionLayer<T>(
                1, _embeddingDimension * 2, 8, geluActivation);
        }

        // Task-specific heads
        _eventClassificationHead = new DenseLayer<T>(_embeddingDimension * 2, _supportedCategories.Count, nullActivation);
        _temporalBoundaryHead = new DenseLayer<T>(_embeddingDimension * 2, 3, nullActivation); // start, end, confidence
        _spatialLocalizationHead = new DenseLayer<T>(_embeddingDimension * 2, 4, nullActivation); // x, y, w, h
        _anomalyDetectionHead = new DenseLayer<T>(_embeddingDimension * 2, 1, nullActivation); // anomaly score
    }

    private static List<string> GetDefaultEventCategories()
    {
        return new List<string>
        {
            "speech", "music", "applause", "laughter", "crying", "shouting",
            "dog_bark", "cat_meow", "bird_chirp", "car_horn", "siren",
            "door_slam", "glass_break", "explosion", "gunshot", "thunder",
            "rain", "wind", "footsteps", "typing", "phone_ring",
            "cooking", "eating", "drinking", "coughing", "sneezing",
            "engine_start", "engine_idle", "tire_screech", "crash",
            "sports_crowd", "whistle", "ball_bounce", "ball_kick",
            "unknown"
        };
    }

    #endregion

    #region Audio/Visual Encoding

    private Vector<T> EncodeAudio(Tensor<T> audioWaveform)
    {
        // Convert waveform to spectrogram representation
        var spectrogram = ComputeSpectrogram(audioWaveform);

        // Project to embedding dimension
        var projected = _audioInputProjection.Forward(spectrogram);

        // Apply transformer encoder layers
        var encoded = projected;
        foreach (var layer in _audioEncoderLayers)
        {
            encoded = layer.Forward(encoded);
        }

        // Final projection and pooling
        var output = _audioOutputProjection.Forward(encoded);
        return GlobalAveragePool(output);
    }

    private Vector<T> EncodeVisual(IEnumerable<Tensor<T>> frames)
    {
        var frameList = frames.ToList();
        if (frameList.Count == 0)
        {
            return new Vector<T>(_embeddingDimension);
        }

        var frameEmbeddings = new List<Vector<T>>();

        foreach (var frame in frameList)
        {
            // Flatten frame to expected input size
            var flattened = FlattenFrame(frame, 768);
            var frameTensor = Tensor<T>.FromVector(flattened);

            // Project to embedding dimension
            var projected = _visualInputProjection.Forward(frameTensor);

            // Apply transformer encoder layers
            var encoded = projected;
            foreach (var layer in _visualEncoderLayers)
            {
                encoded = layer.Forward(encoded);
            }

            var output = _visualOutputProjection.Forward(encoded);
            frameEmbeddings.Add(GlobalAveragePool(output));
        }

        // Average across frames
        return AverageVectors(frameEmbeddings);
    }

    private Tensor<T> ComputeSpectrogram(Tensor<T> waveform)
    {
        var spectrogramData = new T[SPECTROGRAM_BINS];
        var waveData = waveform.ToVector();
        int samplesPerBin = Math.Max(1, waveData.Length / SPECTROGRAM_BINS);

        for (int i = 0; i < SPECTROGRAM_BINS; i++)
        {
            T sum = _numOps.Zero;
            int start = i * samplesPerBin;
            int end = Math.Min(start + samplesPerBin, waveData.Length);

            for (int j = start; j < end; j++)
            {
                var val = _numOps.Abs(waveData[j]);
                sum = _numOps.Add(sum, _numOps.Multiply(val, val));
            }

            spectrogramData[i] = _numOps.Sqrt(_numOps.Divide(sum, _numOps.FromDouble(end - start)));
        }

        var spectrogramVector = new Vector<T>(SPECTROGRAM_BINS);
        for (int i = 0; i < SPECTROGRAM_BINS; i++)
        {
            spectrogramVector[i] = spectrogramData[i];
        }
        return Tensor<T>.FromVector(spectrogramVector);
    }

    private Vector<T> FlattenFrame(Tensor<T> frame, int targetSize)
    {
        var result = new Vector<T>(targetSize);
        var frameData = frame.ToVector();

        if (frameData.Length >= targetSize)
        {
            for (int i = 0; i < targetSize; i++)
            {
                result[i] = frameData[i];
            }
        }
        else
        {
            for (int i = 0; i < frameData.Length; i++)
            {
                result[i] = frameData[i];
            }
            for (int i = frameData.Length; i < targetSize; i++)
            {
                result[i] = _numOps.Zero;
            }
        }

        return result;
    }

    private Vector<T> GlobalAveragePool(Tensor<T> tensor)
    {
        var data = tensor.ToVector();
        int outputSize = Math.Min(_embeddingDimension, data.Length);
        var result = new Vector<T>(outputSize);

        int elementsPerOutput = Math.Max(1, data.Length / outputSize);
        for (int i = 0; i < outputSize; i++)
        {
            T sum = _numOps.Zero;
            int start = i * elementsPerOutput;
            int end = Math.Min(start + elementsPerOutput, data.Length);

            for (int j = start; j < end; j++)
            {
                sum = _numOps.Add(sum, data[j]);
            }
            result[i] = _numOps.Divide(sum, _numOps.FromDouble(end - start));
        }

        return result;
    }

    private Vector<T> AverageVectors(List<Vector<T>> vectors)
    {
        if (vectors.Count == 0)
        {
            return new Vector<T>(_embeddingDimension);
        }

        var result = new Vector<T>(vectors[0].Length);
        foreach (var vec in vectors)
        {
            for (int i = 0; i < result.Length && i < vec.Length; i++)
            {
                result[i] = _numOps.Add(result[i], vec[i]);
            }
        }

        var divisor = _numOps.FromDouble(vectors.Count);
        for (int i = 0; i < result.Length; i++)
        {
            result[i] = _numOps.Divide(result[i], divisor);
        }

        return result;
    }

    #endregion

    #region IAudioVisualEventLocalizationModel Implementation

    /// <inheritdoc/>
    public IEnumerable<AudioVisualEvent> DetectEvents(
        Tensor<T> audioWaveform,
        IEnumerable<Tensor<T>> frames,
        double frameRate)
    {
        var frameList = frames.ToList();
        if (frameList.Count == 0)
        {
            return Enumerable.Empty<AudioVisualEvent>();
        }

        var events = new List<AudioVisualEvent>();
        var proposals = GenerateProposals(audioWaveform, frameList, frameRate).ToList();

        foreach (var proposal in proposals)
        {
            var eventnessScore = _numOps.ToDouble(proposal.EventnessScore);
            if (eventnessScore < 0.5)
            {
                continue;
            }

            // Extract segment
            var (audioSegment, frameSegment) = ExtractSegment(
                audioWaveform, frameList, proposal.StartTime, proposal.EndTime, frameRate);

            // Classify the event
            var classification = ClassifyEvent(audioSegment, frameSegment, _supportedCategories);
            var bestLabel = classification.OrderByDescending(kvp => _numOps.ToDouble(kvp.Value)).First();

            // Get spatial localization
            var bbox = LocalizeSpatially(audioSegment, frameSegment);

            events.Add(new AudioVisualEvent
            {
                StartTime = proposal.StartTime,
                EndTime = proposal.EndTime,
                Label = bestLabel.Key,
                Confidence = _numOps.ToDouble(bestLabel.Value),
                Modality = DetermineModality(audioSegment, frameSegment),
                BoundingBox = bbox
            });
        }

        return events.OrderBy(e => e.StartTime);
    }

    /// <inheritdoc/>
    public IEnumerable<AudioVisualEvent> DetectSpecificEvents(
        Tensor<T> audioWaveform,
        IEnumerable<Tensor<T>> frames,
        IEnumerable<string> targetCategories,
        double frameRate)
    {
        var allEvents = DetectEvents(audioWaveform, frames, frameRate);
        var targetSet = new HashSet<string>(targetCategories, StringComparer.OrdinalIgnoreCase);

        return allEvents.Where(e => targetSet.Contains(e.Label));
    }

    /// <inheritdoc/>
    public IEnumerable<(double StartTime, double EndTime, T Confidence)> LocalizeEventByDescription(
        Tensor<T> audioWaveform,
        IEnumerable<Tensor<T>> frames,
        string eventDescription,
        double frameRate)
    {
        var frameList = frames.ToList();
        var results = new List<(double, double, T)>();

        // Use temporal proposals and check similarity to description
        var proposals = GenerateProposals(audioWaveform, frameList, frameRate).ToList();

        foreach (var proposal in proposals)
        {
            var (audioSegment, frameSegment) = ExtractSegment(
                audioWaveform, frameList, proposal.StartTime, proposal.EndTime, frameRate);

            // Encode segment features
            var audioFeatures = EncodeAudio(audioSegment);
            var visualFeatures = EncodeVisual(frameSegment);
            var fusedFeatures = FuseFeatures(audioFeatures, visualFeatures);

            // Compute text-to-feature similarity
            var descriptionEmbedding = EncodeTextDescription(eventDescription);
            var similarity = ComputeCosineSimilarity(fusedFeatures, descriptionEmbedding);

            if (_numOps.ToDouble(similarity) > 0.3)
            {
                results.Add((proposal.StartTime, proposal.EndTime, similarity));
            }
        }

        return results.OrderByDescending(r => _numOps.ToDouble(r.Item3));
    }

    /// <inheritdoc/>
    public IEnumerable<(double StartTime, double EndTime, T EventnessScore)> GenerateProposals(
        Tensor<T> audioWaveform,
        IEnumerable<Tensor<T>> frames,
        double frameRate)
    {
        var frameList = frames.ToList();
        var proposals = new List<(double, double, T)>();

        if (frameList.Count == 0)
        {
            return proposals;
        }

        double videoDuration = frameList.Count / frameRate;
        int numSegments = (int)(videoDuration / _temporalResolution);

        // Compute per-segment features
        var segmentScores = new List<T>();

        for (int i = 0; i < numSegments; i++)
        {
            double segmentStart = i * _temporalResolution;
            double segmentEnd = (i + 1) * _temporalResolution;

            var (audioSegment, frameSegment) = ExtractSegment(
                audioWaveform, frameList, segmentStart, segmentEnd, frameRate);

            var audioFeatures = EncodeAudio(audioSegment);
            var visualFeatures = EncodeVisual(frameSegment);
            var fusedFeatures = FuseFeatures(audioFeatures, visualFeatures);

            // Apply temporal proposal head
            var proposalTensor = Tensor<T>.FromVector(fusedFeatures);
            var proposalOutput = _temporalProposalHead.Forward(proposalTensor);
            var proposalData = proposalOutput.ToVector();

            // Eventness score
            var score = proposalData.Length > 0 ? _numOps.Abs(proposalData[0]) : _numOps.Zero;
            segmentScores.Add(score);
        }

        // Find contiguous high-scoring regions
        int regionStart = -1;
        T threshold = _numOps.FromDouble(0.3);

        for (int i = 0; i < segmentScores.Count; i++)
        {
            bool isEvent = _numOps.Compare(segmentScores[i], threshold) > 0;

            if (isEvent && regionStart < 0)
            {
                regionStart = i;
            }
            else if (!isEvent && regionStart >= 0)
            {
                // End of region
                T avgScore = ComputeAverageScore(segmentScores, regionStart, i);
                proposals.Add((
                    regionStart * _temporalResolution,
                    i * _temporalResolution,
                    avgScore));
                regionStart = -1;
            }
        }

        // Handle region that extends to end
        if (regionStart >= 0)
        {
            T avgScore = ComputeAverageScore(segmentScores, regionStart, segmentScores.Count);
            proposals.Add((
                regionStart * _temporalResolution,
                segmentScores.Count * _temporalResolution,
                avgScore));
        }

        return proposals.OrderByDescending(p => _numOps.ToDouble(p.Item3));
    }

    /// <inheritdoc/>
    public Dictionary<string, T> ClassifyEvent(
        Tensor<T> audioSegment,
        IEnumerable<Tensor<T>> frameSegment,
        IEnumerable<string> candidateLabels)
    {
        var labelList = candidateLabels.ToList();
        var result = new Dictionary<string, T>();

        var audioFeatures = EncodeAudio(audioSegment);
        var visualFeatures = EncodeVisual(frameSegment);
        var fusedFeatures = FuseFeatures(audioFeatures, visualFeatures);

        // Apply classification head
        var fusedTensor = Tensor<T>.FromVector(fusedFeatures);
        var logits = _eventClassificationHead.Forward(fusedTensor);
        var logitsData = logits.ToVector();

        // Apply softmax
        var probs = Softmax(logitsData);

        // Map to labels
        for (int i = 0; i < labelList.Count && i < probs.Length; i++)
        {
            result[labelList[i]] = probs[i];
        }

        // Handle remaining labels
        for (int i = probs.Length; i < labelList.Count; i++)
        {
            result[labelList[i]] = _numOps.Zero;
        }

        return result;
    }

    /// <inheritdoc/>
    public IEnumerable<AudioVisualEvent> TrackEvent(
        Tensor<T> audioWaveform,
        IEnumerable<Tensor<T>> frames,
        AudioVisualEvent initialEvent,
        double frameRate)
    {
        var frameList = frames.ToList();
        var trajectory = new List<AudioVisualEvent> { initialEvent };

        double currentTime = initialEvent.EndTime;
        double videoDuration = frameList.Count / frameRate;
        double windowSize = initialEvent.EndTime - initialEvent.StartTime;

        // Get reference features from initial event
        var (initAudio, initFrames) = ExtractSegment(
            audioWaveform, frameList, initialEvent.StartTime, initialEvent.EndTime, frameRate);
        var referenceFeatures = FuseFeatures(EncodeAudio(initAudio), EncodeVisual(initFrames));

        // Track forward in time
        while (currentTime < videoDuration)
        {
            double searchStart = currentTime;
            double searchEnd = Math.Min(currentTime + windowSize * 2, videoDuration);

            var (searchAudio, searchFrames) = ExtractSegment(
                audioWaveform, frameList, searchStart, searchEnd, frameRate);

            var searchFeatures = FuseFeatures(EncodeAudio(searchAudio), EncodeVisual(searchFrames));
            var similarity = ComputeCosineSimilarity(referenceFeatures, searchFeatures);

            if (_numOps.ToDouble(similarity) < 0.5)
            {
                break; // Event ended
            }

            // Refine boundaries
            var bbox = LocalizeSpatially(searchAudio, searchFrames);

            trajectory.Add(new AudioVisualEvent
            {
                StartTime = searchStart,
                EndTime = searchEnd,
                Label = initialEvent.Label,
                Confidence = _numOps.ToDouble(similarity),
                Modality = initialEvent.Modality,
                BoundingBox = bbox
            });

            currentTime = searchEnd;
        }

        return trajectory;
    }

    /// <inheritdoc/>
    public IEnumerable<(double StartTime, double EndTime, T SyncQuality, string Description)> DetectSyncEvents(
        Tensor<T> audioWaveform,
        IEnumerable<Tensor<T>> frames,
        double frameRate)
    {
        var frameList = frames.ToList();
        var syncEvents = new List<(double, double, T, string)>();

        if (frameList.Count == 0)
        {
            return syncEvents;
        }

        double videoDuration = frameList.Count / frameRate;
        double windowSize = 1.0; // 1 second windows

        for (double t = 0; t < videoDuration - windowSize; t += windowSize / 2)
        {
            var (audioSeg, frameSeg) = ExtractSegment(
                audioWaveform, frameList, t, t + windowSize, frameRate);

            var audioFeatures = EncodeAudio(audioSeg);
            var visualFeatures = EncodeVisual(frameSeg);

            // Measure sync quality via cross-correlation of feature magnitudes
            var syncQuality = ComputeSyncQuality(audioFeatures, visualFeatures);

            if (_numOps.ToDouble(syncQuality) > 0.6)
            {
                string description = DescribeSyncEvent(audioSeg, frameSeg);
                syncEvents.Add((t, t + windowSize, syncQuality, description));
            }
        }

        return MergeOverlappingSyncEvents(syncEvents);
    }

    /// <inheritdoc/>
    public IEnumerable<(double StartTime, double EndTime, string SceneDescription)> SegmentScenes(
        Tensor<T> audioWaveform,
        IEnumerable<Tensor<T>> frames,
        double frameRate)
    {
        var frameList = frames.ToList();
        var scenes = new List<(double, double, string)>();

        if (frameList.Count == 0)
        {
            return scenes;
        }

        double videoDuration = frameList.Count / frameRate;
        var segmentFeatures = new List<(double Time, Vector<T> Features)>();

        // Compute features at regular intervals
        double interval = 0.5; // 500ms intervals
        for (double t = 0; t < videoDuration; t += interval)
        {
            var (audioSeg, frameSeg) = ExtractSegment(
                audioWaveform, frameList, t, Math.Min(t + interval, videoDuration), frameRate);

            var features = FuseFeatures(EncodeAudio(audioSeg), EncodeVisual(frameSeg));
            segmentFeatures.Add((t, features));
        }

        // Find scene boundaries via feature discontinuities
        var boundaries = new List<int> { 0 };
        T threshold = _numOps.FromDouble(0.4);

        for (int i = 1; i < segmentFeatures.Count; i++)
        {
            var similarity = ComputeCosineSimilarity(
                segmentFeatures[i - 1].Features,
                segmentFeatures[i].Features);

            if (_numOps.Compare(_numOps.Subtract(_numOps.One, similarity), threshold) > 0)
            {
                boundaries.Add(i);
            }
        }
        boundaries.Add(segmentFeatures.Count);

        // Create scene segments
        for (int i = 0; i < boundaries.Count - 1; i++)
        {
            int startIdx = boundaries[i];
            int endIdx = boundaries[i + 1];
            double startTime = segmentFeatures[startIdx].Time;
            double endTime = segmentFeatures[Math.Min(endIdx, segmentFeatures.Count - 1)].Time + interval;

            var (audioSeg, frameSeg) = ExtractSegment(audioWaveform, frameList, startTime, endTime, frameRate);
            string description = GenerateSceneDescription(audioSeg, frameSeg);

            scenes.Add((startTime, endTime, description));
        }

        return scenes;
    }

    /// <inheritdoc/>
    public IEnumerable<(double Time, string Caption)> GenerateDenseCaptions(
        Tensor<T> audioWaveform,
        IEnumerable<Tensor<T>> frames,
        double frameRate)
    {
        var events = DetectEvents(audioWaveform, frames, frameRate).ToList();
        var captions = new List<(double, string)>();

        foreach (var evt in events)
        {
            double midTime = (evt.StartTime + evt.EndTime) / 2;
            string caption = $"{evt.Label} detected ({evt.Confidence:F2} confidence)";

            if (evt.BoundingBox.HasValue)
            {
                var box = evt.BoundingBox.Value;
                caption += $" at position ({box.X}, {box.Y})";
            }

            captions.Add((midTime, caption));
        }

        return captions.OrderBy(c => c.Item1);
    }

    /// <inheritdoc/>
    public (string Answer, IEnumerable<(double StartTime, double EndTime)> Evidence) AnswerEventQuestion(
        Tensor<T> audioWaveform,
        IEnumerable<Tensor<T>> frames,
        string question,
        double frameRate)
    {
        var events = DetectEvents(audioWaveform, frames, frameRate).ToList();
        var evidence = new List<(double, double)>();
        string answer;

        // Parse question type
        var questionLower = question.ToLowerInvariant();

        if (questionLower.Contains("when") || questionLower.Contains("what time"))
        {
            // Temporal question
            var relevantEvents = FindRelevantEvents(events, question);
            if (relevantEvents.Any())
            {
                var evt = relevantEvents.First();
                answer = $"The event occurs from {evt.StartTime:F1}s to {evt.EndTime:F1}s";
                evidence.AddRange(relevantEvents.Select(e => (e.StartTime, e.EndTime)));
            }
            else
            {
                answer = "No relevant event found in the video.";
            }
        }
        else if (questionLower.Contains("where"))
        {
            // Spatial question
            var relevantEvents = FindRelevantEvents(events, question).Where(e => e.BoundingBox.HasValue);
            if (relevantEvents.Any())
            {
                var evt = relevantEvents.First();
                var box = evt.BoundingBox!.Value;
                answer = $"The event is located at position ({box.X}, {box.Y}) with size ({box.Width}x{box.Height})";
                evidence.AddRange(relevantEvents.Select(e => (e.StartTime, e.EndTime)));
            }
            else
            {
                answer = "Could not determine spatial location.";
            }
        }
        else if (questionLower.Contains("how many") || questionLower.Contains("count"))
        {
            // Counting question
            var relevantEvents = FindRelevantEvents(events, question);
            answer = $"Found {relevantEvents.Count()} occurrences";
            evidence.AddRange(relevantEvents.Select(e => (e.StartTime, e.EndTime)));
        }
        else
        {
            // General question
            var relevantEvents = FindRelevantEvents(events, question);
            if (relevantEvents.Any())
            {
                var labels = relevantEvents.Select(e => e.Label).Distinct();
                answer = $"Detected: {string.Join(", ", labels)}";
                evidence.AddRange(relevantEvents.Select(e => (e.StartTime, e.EndTime)));
            }
            else
            {
                answer = "No relevant events found.";
            }
        }

        return (answer, evidence);
    }

    /// <inheritdoc/>
    public IEnumerable<(double StartTime, double EndTime, T AnomalyScore, string Description)> DetectAnomalies(
        Tensor<T> audioWaveform,
        IEnumerable<Tensor<T>> frames,
        double frameRate)
    {
        var frameList = frames.ToList();
        var anomalies = new List<(double, double, T, string)>();

        if (frameList.Count == 0)
        {
            return anomalies;
        }

        double videoDuration = frameList.Count / frameRate;
        var allFeatures = new List<Vector<T>>();

        // Compute features for all segments
        for (double t = 0; t < videoDuration; t += _temporalResolution)
        {
            var (audioSeg, frameSeg) = ExtractSegment(
                audioWaveform, frameList, t, t + _temporalResolution, frameRate);
            allFeatures.Add(FuseFeatures(EncodeAudio(audioSeg), EncodeVisual(frameSeg)));
        }

        // Compute mean and variance
        var meanFeatures = ComputeMeanVector(allFeatures);

        // Find anomalies based on distance from mean
        for (int i = 0; i < allFeatures.Count; i++)
        {
            var distance = ComputeEuclideanDistance(allFeatures[i], meanFeatures);

            // Apply anomaly detection head
            var featureTensor = Tensor<T>.FromVector(allFeatures[i]);
            var anomalyOutput = _anomalyDetectionHead.Forward(featureTensor);
            var anomalyData = anomalyOutput.ToVector();

            var anomalyScore = anomalyData.Length > 0
                ? MathHelper.Sigmoid(anomalyData[0])
                : distance;

            if (_numOps.ToDouble(anomalyScore) > 0.7)
            {
                double startTime = i * _temporalResolution;
                double endTime = (i + 1) * _temporalResolution;
                string description = DescribeAnomaly(allFeatures[i], meanFeatures);

                anomalies.Add((startTime, endTime, anomalyScore, description));
            }
        }

        return MergeContiguousAnomalies(anomalies);
    }

    /// <inheritdoc/>
    public (Tensor<T> AudioToVisualAttention, Tensor<T> VisualToAudioAttention) ComputeEventAttention(
        Tensor<T> audioSegment,
        IEnumerable<Tensor<T>> frameSegment)
    {
        var audioFeatures = EncodeAudio(audioSegment);
        var visualFeatures = EncodeVisual(frameSegment);

        // Compute attention matrices
        int audioLen = audioFeatures.Length;
        int visualLen = visualFeatures.Length;

        var audioToVisual = new T[audioLen * visualLen];
        var visualToAudio = new T[visualLen * audioLen];

        // Compute attention weights
        for (int i = 0; i < audioLen; i++)
        {
            for (int j = 0; j < visualLen; j++)
            {
                var product = _numOps.Multiply(audioFeatures[i], visualFeatures[j]);
                audioToVisual[i * visualLen + j] = product;
                visualToAudio[j * audioLen + i] = product;
            }
        }

        // Apply softmax normalization - convert arrays to vectors
        var a2vVector = new Vector<T>(audioToVisual.Length);
        for (int i = 0; i < audioToVisual.Length; i++) a2vVector[i] = audioToVisual[i];
        var a2vTensor = new Tensor<T>(new[] { audioLen, visualLen }, a2vVector);

        var v2aVector = new Vector<T>(visualToAudio.Length);
        for (int i = 0; i < visualToAudio.Length; i++) v2aVector[i] = visualToAudio[i];
        var v2aTensor = new Tensor<T>(new[] { visualLen, audioLen }, v2aVector);

        return (SoftmaxTensor(a2vTensor, axis: 1), SoftmaxTensor(v2aTensor, axis: 1));
    }

    #endregion

    #region Helper Methods

    private (Tensor<T> Audio, List<Tensor<T>> Frames) ExtractSegment(
        Tensor<T> audioWaveform,
        List<Tensor<T>> frames,
        double startTime,
        double endTime,
        double frameRate)
    {
        // Extract audio segment
        var audioData = audioWaveform.ToVector();
        int audioSampleRate = 16000; // Assume 16kHz
        int startSample = Math.Max(0, (int)(startTime * audioSampleRate));
        int endSample = Math.Min(audioData.Length, (int)(endTime * audioSampleRate));

        int audioLength = Math.Max(1, endSample - startSample);
        var segmentAudioVector = new Vector<T>(audioLength);
        for (int i = 0; i < audioLength && (startSample + i) < audioData.Length; i++)
        {
            segmentAudioVector[i] = audioData[startSample + i];
        }

        // Extract frame segment
        int startFrame = Math.Max(0, (int)(startTime * frameRate));
        int endFrame = Math.Min(frames.Count, (int)(endTime * frameRate));

        var segmentFrames = new List<Tensor<T>>();
        for (int i = startFrame; i < endFrame; i++)
        {
            segmentFrames.Add(frames[i]);
        }

        if (segmentFrames.Count == 0 && frames.Count > 0)
        {
            segmentFrames.Add(frames[Math.Min(startFrame, frames.Count - 1)]);
        }

        return (Tensor<T>.FromVector(segmentAudioVector), segmentFrames);
    }

    private Vector<T> FuseFeatures(Vector<T> audioFeatures, Vector<T> visualFeatures)
    {
        // Concatenate features
        int totalLength = audioFeatures.Length + visualFeatures.Length;
        var fused = new Vector<T>(totalLength);

        for (int i = 0; i < audioFeatures.Length; i++)
        {
            fused[i] = audioFeatures[i];
        }
        for (int i = 0; i < visualFeatures.Length; i++)
        {
            fused[audioFeatures.Length + i] = visualFeatures[i];
        }

        // Apply cross-modal attention
        var fusedTensor = Tensor<T>.FromVector(fused);
        var attended = fusedTensor;
        foreach (var layer in _crossModalAttentionLayers)
        {
            attended = layer.Forward(attended);
        }

        return attended.ToVector();
    }

    private (int X, int Y, int Width, int Height)? LocalizeSpatially(
        Tensor<T> audioSegment,
        IEnumerable<Tensor<T>> frameSegment)
    {
        var audioFeatures = EncodeAudio(audioSegment);
        var visualFeatures = EncodeVisual(frameSegment);
        var fusedFeatures = FuseFeatures(audioFeatures, visualFeatures);

        // Apply spatial localization head
        var fusedTensor = Tensor<T>.FromVector(fusedFeatures);
        var spatialOutput = _spatialLocalizationHead.Forward(fusedTensor);
        var bbox = spatialOutput.ToVector();

        if (bbox.Length < 4)
        {
            return null;
        }

        // Convert normalized coordinates to pixel coordinates (assuming 224x224)
        int imgSize = 224;
        int x = (int)(_numOps.ToDouble(MathHelper.Sigmoid(bbox[0])) * imgSize);
        int y = (int)(_numOps.ToDouble(MathHelper.Sigmoid(bbox[1])) * imgSize);
        int w = (int)(_numOps.ToDouble(MathHelper.Sigmoid(bbox[2])) * imgSize);
        int h = (int)(_numOps.ToDouble(MathHelper.Sigmoid(bbox[3])) * imgSize);

        return (x, y, Math.Max(1, w), Math.Max(1, h));
    }

    private string DetermineModality(Tensor<T> audioSegment, IEnumerable<Tensor<T>> frameSegment)
    {
        var audioEnergy = ComputeEnergy(audioSegment.ToVector());
        var visualActivity = ComputeVisualActivity(frameSegment.ToList());

        double audioVal = _numOps.ToDouble(audioEnergy);
        double visualVal = _numOps.ToDouble(visualActivity);

        if (audioVal > 0.7 && visualVal > 0.7) return "both";
        if (audioVal > visualVal) return "audio";
        return "visual";
    }

    private T ComputeEnergy(Vector<T> signal)
    {
        T sum = _numOps.Zero;
        for (int i = 0; i < signal.Length; i++)
        {
            sum = _numOps.Add(sum, _numOps.Multiply(signal[i], signal[i]));
        }
        return _numOps.Sqrt(_numOps.Divide(sum, _numOps.FromDouble(Math.Max(1, signal.Length))));
    }

    private T ComputeVisualActivity(List<Tensor<T>> frames)
    {
        if (frames.Count < 2)
        {
            return _numOps.Zero;
        }

        T totalDiff = _numOps.Zero;
        for (int i = 1; i < frames.Count; i++)
        {
            var curr = frames[i].ToVector();
            var prev = frames[i - 1].ToVector();

            int minLen = Math.Min(curr.Length, prev.Length);
            for (int j = 0; j < minLen; j++)
            {
                var diff = _numOps.Abs(_numOps.Subtract(curr[j], prev[j]));
                totalDiff = _numOps.Add(totalDiff, diff);
            }
        }

        return _numOps.Divide(totalDiff, _numOps.FromDouble(frames.Count * frames[0].ToVector().Length));
    }

    private Vector<T> EncodeTextDescription(string description)
    {
        // Simple bag-of-words encoding
        var result = new Vector<T>(_embeddingDimension);
        var words = description.ToLowerInvariant().Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);

        foreach (var word in words)
        {
            // Use deterministic FNV-1a hash instead of string.GetHashCode()
            // which is not deterministic across .NET versions or processes
            int hash = Math.Abs(GetDeterministicHashCode(word)) % _embeddingDimension;
            result[hash] = _numOps.Add(result[hash], _numOps.One);
        }

        // L2 normalize
        return NormalizeVector(result);
    }

    /// <summary>
    /// Computes a deterministic hash code using the FNV-1a algorithm.
    /// Unlike string.GetHashCode(), this produces consistent results across runs.
    /// </summary>
    private static int GetDeterministicHashCode(string str)
    {
        unchecked
        {
            const int FNV_OFFSET_BASIS = unchecked((int)2166136261);
            const int FNV_PRIME = 16777619;

            int hash = FNV_OFFSET_BASIS;
            foreach (char c in str)
            {
                hash ^= c;
                hash *= FNV_PRIME;
            }
            return hash;
        }
    }

    private T ComputeCosineSimilarity(Vector<T> a, Vector<T> b)
    {
        int minLen = Math.Min(a.Length, b.Length);
        T dot = _numOps.Zero;
        T normA = _numOps.Zero;
        T normB = _numOps.Zero;

        for (int i = 0; i < minLen; i++)
        {
            dot = _numOps.Add(dot, _numOps.Multiply(a[i], b[i]));
            normA = _numOps.Add(normA, _numOps.Multiply(a[i], a[i]));
            normB = _numOps.Add(normB, _numOps.Multiply(b[i], b[i]));
        }

        var denom = _numOps.Multiply(_numOps.Sqrt(normA), _numOps.Sqrt(normB));
        if (_numOps.ToDouble(denom) < 1e-8)
        {
            return _numOps.Zero;
        }

        return _numOps.Divide(dot, denom);
    }

    private Vector<T> NormalizeVector(Vector<T> vec)
    {
        T norm = _numOps.Zero;
        for (int i = 0; i < vec.Length; i++)
        {
            norm = _numOps.Add(norm, _numOps.Multiply(vec[i], vec[i]));
        }
        norm = _numOps.Sqrt(norm);

        if (_numOps.ToDouble(norm) < 1e-8)
        {
            return vec;
        }

        var result = new Vector<T>(vec.Length);
        for (int i = 0; i < vec.Length; i++)
        {
            result[i] = _numOps.Divide(vec[i], norm);
        }
        return result;
    }

    private T ComputeAverageScore(List<T> scores, int start, int end)
    {
        T sum = _numOps.Zero;
        for (int i = start; i < end; i++)
        {
            sum = _numOps.Add(sum, scores[i]);
        }
        return _numOps.Divide(sum, _numOps.FromDouble(end - start));
    }

    private Vector<T> Softmax(Vector<T> logits)
    {
        var result = new Vector<T>(logits.Length);
        T maxVal = logits[0];
        for (int i = 1; i < logits.Length; i++)
        {
            if (_numOps.Compare(logits[i], maxVal) > 0)
            {
                maxVal = logits[i];
            }
        }

        T sum = _numOps.Zero;
        for (int i = 0; i < logits.Length; i++)
        {
            result[i] = _numOps.Exp(_numOps.Subtract(logits[i], maxVal));
            sum = _numOps.Add(sum, result[i]);
        }

        for (int i = 0; i < logits.Length; i++)
        {
            result[i] = _numOps.Divide(result[i], sum);
        }

        return result;
    }

    private Tensor<T> SoftmaxTensor(Tensor<T> tensor, int axis)
    {
        // Use the Engine's vectorized Softmax implementation which handles
        // arbitrary tensor ranks and axis parameters efficiently
        return Engine.Softmax(tensor, axis);
    }

    private T ComputeSyncQuality(Vector<T> audioFeatures, Vector<T> visualFeatures)
    {
        return ComputeCosineSimilarity(audioFeatures, visualFeatures);
    }

    private string DescribeSyncEvent(Tensor<T> audioSeg, IEnumerable<Tensor<T>> frameSeg)
    {
        return "Audio-visual synchronization detected";
    }

    private IEnumerable<(double, double, T, string)> MergeOverlappingSyncEvents(
        List<(double StartTime, double EndTime, T SyncQuality, string Description)> events)
    {
        if (events.Count == 0)
        {
            return events;
        }

        var sorted = events.OrderBy(e => e.StartTime).ToList();
        var merged = new List<(double, double, T, string)>();
        var current = sorted[0];

        for (int i = 1; i < sorted.Count; i++)
        {
            if (sorted[i].StartTime <= current.EndTime)
            {
                // Merge
                var maxQuality = _numOps.Compare(sorted[i].SyncQuality, current.SyncQuality) > 0
                    ? sorted[i].SyncQuality
                    : current.SyncQuality;
                current = (current.StartTime, Math.Max(current.EndTime, sorted[i].EndTime), maxQuality, current.Description);
            }
            else
            {
                merged.Add(current);
                current = sorted[i];
            }
        }
        merged.Add(current);

        return merged;
    }

    private string GenerateSceneDescription(Tensor<T> audioSeg, IEnumerable<Tensor<T>> frameSeg)
    {
        var classification = ClassifyEvent(audioSeg, frameSeg, _supportedCategories);
        var topLabel = classification.OrderByDescending(kvp => _numOps.ToDouble(kvp.Value)).First();
        return $"Scene with {topLabel.Key}";
    }

    private IEnumerable<AudioVisualEvent> FindRelevantEvents(List<AudioVisualEvent> events, string query)
    {
        var queryWords = query.ToLowerInvariant()
            .Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries)
            .ToHashSet();

        return events.Where(e =>
        {
            var labelWords = e.Label.ToLowerInvariant().Replace('_', ' ').Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
            return labelWords.Any(w => queryWords.Contains(w));
        });
    }

    private Vector<T> ComputeMeanVector(List<Vector<T>> vectors)
    {
        if (vectors.Count == 0)
        {
            return new Vector<T>(_embeddingDimension * 2);
        }

        var result = new Vector<T>(vectors[0].Length);
        foreach (var vec in vectors)
        {
            for (int i = 0; i < result.Length && i < vec.Length; i++)
            {
                result[i] = _numOps.Add(result[i], vec[i]);
            }
        }

        var divisor = _numOps.FromDouble(vectors.Count);
        for (int i = 0; i < result.Length; i++)
        {
            result[i] = _numOps.Divide(result[i], divisor);
        }

        return result;
    }

    private T ComputeEuclideanDistance(Vector<T> a, Vector<T> b)
    {
        T sum = _numOps.Zero;
        int minLen = Math.Min(a.Length, b.Length);

        for (int i = 0; i < minLen; i++)
        {
            var diff = _numOps.Subtract(a[i], b[i]);
            sum = _numOps.Add(sum, _numOps.Multiply(diff, diff));
        }

        return _numOps.Sqrt(sum);
    }

    private string DescribeAnomaly(Vector<T> features, Vector<T> mean)
    {
        return "Anomalous audio-visual pattern detected";
    }

    private IEnumerable<(double, double, T, string)> MergeContiguousAnomalies(
        List<(double StartTime, double EndTime, T AnomalyScore, string Description)> anomalies)
    {
        if (anomalies.Count == 0)
        {
            return anomalies;
        }

        var sorted = anomalies.OrderBy(a => a.StartTime).ToList();
        var merged = new List<(double, double, T, string)>();
        var current = sorted[0];

        for (int i = 1; i < sorted.Count; i++)
        {
            if (sorted[i].StartTime <= current.EndTime + _temporalResolution)
            {
                var maxScore = _numOps.Compare(sorted[i].AnomalyScore, current.AnomalyScore) > 0
                    ? sorted[i].AnomalyScore
                    : current.AnomalyScore;
                current = (current.StartTime, sorted[i].EndTime, maxScore, current.Description);
            }
            else
            {
                merged.Add(current);
                current = sorted[i];
            }
        }
        merged.Add(current);

        return merged;
    }

    #endregion

    #region NeuralNetworkBase Implementation

    /// <inheritdoc/>
    public override Tensor<T> Predict(Tensor<T> input)
    {
        // GPU-resident optimization: use TryForwardGpuOptimized for speedup
        if (TryForwardGpuOptimized(input, out var gpuResult))
            return gpuResult;

        var audioFeatures = EncodeAudio(input);
        return Tensor<T>.FromVector(audioFeatures);
    }

    /// <inheritdoc/>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        SetTrainingMode(true);
        var prediction = Predict(input);
        var loss = _lossFunction.CalculateLoss(prediction.ToVector(), expectedOutput.ToVector());
        LastLoss = loss;
        SetTrainingMode(false);
    }

    /// <inheritdoc/>
    public override void UpdateParameters(Vector<T> gradients)
    {
        if (gradients.Length != ParameterCount)
        {
            throw new ArgumentException(
                $"Gradient vector length ({gradients.Length}) must match parameter count ({ParameterCount}).",
                nameof(gradients));
        }

        // Get current parameters
        var currentParams = GetParameters();

        // Apply gradient descent update: params = params - learning_rate * gradients
        T learningRate = NumOps.FromDouble(0.001); // Default learning rate
        for (int i = 0; i < currentParams.Length; i++)
        {
            currentParams[i] = NumOps.Subtract(currentParams[i], NumOps.Multiply(learningRate, gradients[i]));
        }

        // Set the updated parameters
        SetParameters(currentParams);
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            Name = "AudioVisualEventLocalizationNetwork",
            ModelType = ModelType.NeuralNetwork,
            FeatureCount = _embeddingDimension,
            Complexity = _numEncoderLayers * 2 + 4,
            Description = "Audio-visual event localization network for temporal and spatial event detection"
        };
    }

    /// <inheritdoc/>
    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_embeddingDimension);
        writer.Write(_temporalResolution);
        writer.Write(_numEncoderLayers);
        writer.Write(_supportedCategories.Count);
        foreach (var category in _supportedCategories)
        {
            writer.Write(category);
        }
    }

    /// <inheritdoc/>
    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        // Read serialized values
        int embDim = reader.ReadInt32();
        double tempRes = reader.ReadDouble();
        int numLayers = reader.ReadInt32();
        int categoryCount = reader.ReadInt32();
        var categories = new List<string>();
        for (int i = 0; i < categoryCount; i++)
        {
            categories.Add(reader.ReadString());
        }

        // Validate that loaded values match current instance configuration
        if (embDim != _embeddingDimension)
        {
            throw new InvalidOperationException(
                $"Loaded embedding dimension ({embDim}) doesn't match current ({_embeddingDimension}).");
        }

        if (Math.Abs(tempRes - _temporalResolution) > 0.0001)
        {
            throw new InvalidOperationException(
                $"Loaded temporal resolution ({tempRes}) doesn't match current ({_temporalResolution}).");
        }

        if (numLayers != _numEncoderLayers)
        {
            throw new InvalidOperationException(
                $"Loaded encoder layers ({numLayers}) doesn't match current ({_numEncoderLayers}).");
        }

        if (categoryCount != _supportedCategories.Count)
        {
            throw new InvalidOperationException(
                $"Loaded category count ({categoryCount}) doesn't match current ({_supportedCategories.Count}).");
        }
    }

    /// <inheritdoc/>
    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new AudioVisualEventLocalizationNetwork<T>(
            Architecture,
            _embeddingDimension,
            _temporalResolution,
            _numEncoderLayers,
            _supportedCategories);
    }

    /// <inheritdoc/>
    public override Vector<T> GetParameters()
    {
        var allParams = new List<T>();

        void AddLayerParams(ILayer<T> layer)
        {
            var p = layer.GetParameters();
            for (int i = 0; i < p.Length; i++)
            {
                allParams.Add(p[i]);
            }
        }

        AddLayerParams(_audioInputProjection);
        foreach (var layer in _audioEncoderLayers) AddLayerParams(layer);
        AddLayerParams(_audioOutputProjection);

        AddLayerParams(_visualInputProjection);
        foreach (var layer in _visualEncoderLayers) AddLayerParams(layer);
        AddLayerParams(_visualOutputProjection);

        foreach (var layer in _temporalAttentionLayers) AddLayerParams(layer);
        AddLayerParams(_temporalProposalHead);

        foreach (var layer in _crossModalAttentionLayers) AddLayerParams(layer);
        AddLayerParams(_eventClassificationHead);
        AddLayerParams(_temporalBoundaryHead);
        AddLayerParams(_spatialLocalizationHead);
        AddLayerParams(_anomalyDetectionHead);

        var result = new Vector<T>(allParams.Count);
        for (int i = 0; i < allParams.Count; i++)
        {
            result[i] = allParams[i];
        }

        return result;
    }

    /// <inheritdoc/>
    public override void SetParameters(Vector<T> parameters)
    {
        var offset = 0;

        void SetLayerParams(ILayer<T> layer)
        {
            var count = layer.ParameterCount;
            var p = new Vector<T>(count);
            for (int i = 0; i < count; i++)
            {
                if (offset + i < parameters.Length)
                {
                    p[i] = parameters[offset + i];
                }
            }
            layer.SetParameters(p);
            offset += count;
        }

        SetLayerParams(_audioInputProjection);
        foreach (var layer in _audioEncoderLayers) SetLayerParams(layer);
        SetLayerParams(_audioOutputProjection);

        SetLayerParams(_visualInputProjection);
        foreach (var layer in _visualEncoderLayers) SetLayerParams(layer);
        SetLayerParams(_visualOutputProjection);

        foreach (var layer in _temporalAttentionLayers) SetLayerParams(layer);
        SetLayerParams(_temporalProposalHead);

        foreach (var layer in _crossModalAttentionLayers) SetLayerParams(layer);
        SetLayerParams(_eventClassificationHead);
        SetLayerParams(_temporalBoundaryHead);
        SetLayerParams(_spatialLocalizationHead);
        SetLayerParams(_anomalyDetectionHead);
    }

    /// <inheritdoc/>
    public override int ParameterCount
    {
        get
        {
            var count = 0;

            count += _audioInputProjection.ParameterCount;
            foreach (var layer in _audioEncoderLayers) count += layer.ParameterCount;
            count += _audioOutputProjection.ParameterCount;

            count += _visualInputProjection.ParameterCount;
            foreach (var layer in _visualEncoderLayers) count += layer.ParameterCount;
            count += _visualOutputProjection.ParameterCount;

            foreach (var layer in _temporalAttentionLayers) count += layer.ParameterCount;
            count += _temporalProposalHead.ParameterCount;

            foreach (var layer in _crossModalAttentionLayers) count += layer.ParameterCount;
            count += _eventClassificationHead.ParameterCount;
            count += _temporalBoundaryHead.ParameterCount;
            count += _spatialLocalizationHead.ParameterCount;
            count += _anomalyDetectionHead.ParameterCount;

            return count;
        }
    }

    /// <inheritdoc/>
    public override IFullModel<T, Tensor<T>, Tensor<T>> DeepCopy()
    {
        var copy = new AudioVisualEventLocalizationNetwork<T>(
            Architecture,
            _embeddingDimension,
            _temporalResolution,
            _numEncoderLayers,
            _supportedCategories,
            _optimizer,
            _lossFunction);

        copy.SetParameters(GetParameters());
        return copy;
    }

    #endregion
}
