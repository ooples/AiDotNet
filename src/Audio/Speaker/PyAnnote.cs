using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Onnx;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Audio.Speaker;

/// <summary>
/// pyannote 3.x end-to-end speaker diarization model.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// pyannote.audio 3.x (Plaquet &amp; Bredin, ASRU 2023) is a state-of-the-art speaker
/// diarization pipeline using end-to-end neural segmentation with PyanNet architecture.
/// It segments audio into speaker turns, supports overlapping speech detection, and
/// achieves 11.2% DER on AMI Mix-Headset benchmark.
/// </para>
/// <para>
/// <b>For Beginners:</b> pyannote figures out "who spoke when" in a recording with
/// multiple speakers. It's like automatically labeling a meeting transcript with
/// "Speaker A: 0:00-0:15, Speaker B: 0:15-0:45..." It can even detect when two people
/// talk at the same time (overlapping speech).
///
/// <b>Usage:</b>
/// <code>
/// var arch = new NeuralNetworkArchitecture&lt;float&gt;(inputFeatures: 80, outputSize: 7);
/// var model = new PyAnnote&lt;float&gt;(arch, "pyannote_segmentation.onnx");
/// var result = model.Diarize(meetingAudio);
/// foreach (var seg in result.Segments) Console.WriteLine($"{seg.Speaker}: {seg.StartTime:F1}s - {seg.EndTime:F1}s");
/// </code>
/// </para>
/// </remarks>
public class PyAnnote<T> : SpeakerRecognitionBase<T>, ISpeakerDiarizer<T>
{
    #region Fields

    private readonly PyAnnoteOptions _options;
    public override ModelOptions GetOptions() => _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private bool _useNativeMode;
    private bool _disposed;

    #endregion

    #region ISpeakerDiarizer Properties

    public new int SampleRate => _options.SampleRate;
    public double MinSegmentDuration => _options.MinSegmentDuration;
    public bool SupportsOverlapDetection => _options.EnableOverlapDetection;
    public new bool IsOnnxMode => !_useNativeMode && OnnxEncoder is not null;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a pyannote diarization model in ONNX inference mode.
    /// </summary>
    public PyAnnote(NeuralNetworkArchitecture<T> architecture, string modelPath, PyAnnoteOptions? options = null)
        : base(architecture)
    {
        _options = options ?? new PyAnnoteOptions();
        _useNativeMode = false;
        base.SampleRate = _options.SampleRate;
        EmbeddingDimension = _options.EmbeddingDim;
        OnnxEncoder = new OnnxModel<T>(modelPath, _options.OnnxOptions);
        _optimizer = new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        InitializeLayers();
    }

    /// <summary>
    /// Creates a pyannote diarization model in native training mode.
    /// </summary>
    public PyAnnote(NeuralNetworkArchitecture<T> architecture, PyAnnoteOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture)
    {
        _options = options ?? new PyAnnoteOptions();
        _useNativeMode = true;
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this);
        base.SampleRate = _options.SampleRate;
        EmbeddingDimension = _options.EmbeddingDim;
        InitializeLayers();
    }

    internal static async Task<PyAnnote<T>> CreateAsync(PyAnnoteOptions? options = null, IProgress<double>? progress = null, CancellationToken cancellationToken = default)
    {
        options ??= new PyAnnoteOptions();
        string mp = options.ModelPath ?? string.Empty;
        if (string.IsNullOrEmpty(mp))
        {
            var dl = new OnnxModelDownloader();
            mp = await dl.DownloadAsync("pyannote", "pyannote_segmentation.onnx", progress: progress, cancellationToken);
            options.ModelPath = mp;
        }
        int numPowersetClasses = 1 + options.MaxSpeakersPerChunk + (options.MaxSpeakersPerChunk * (options.MaxSpeakersPerChunk - 1)) / 2;
        var arch = new NeuralNetworkArchitecture<T>(inputFeatures: options.NumMels, outputSize: numPowersetClasses);
        return new PyAnnote<T>(arch, mp, options);
    }

    #endregion

    #region ISpeakerDiarizer

    public DiarizationResult<T> Diarize(Tensor<T> audio, int? numSpeakers = null, int minSpeakers = 1, int maxSpeakers = 10)
    {
        ThrowIfDisposed();
        int effectiveMax = numSpeakers ?? _options.MaxSpeakers ?? maxSpeakers;

        // Segment audio into overlapping chunks
        var chunks = SegmentIntoChunks(audio);

        // Run segmentation model on each chunk to get per-frame speaker activity
        var chunkActivations = new List<Tensor<T>>();
        foreach (var chunk in chunks)
        {
            var features = PreprocessAudio(chunk.Audio);
            Tensor<T> activations = IsOnnxMode && OnnxEncoder is not null ? OnnxEncoder.Run(features) : Predict(features);
            chunkActivations.Add(activations);
        }

        // Aggregate chunk-level activations into global speaker segments
        var segments = AggregateChunkActivations(chunks, chunkActivations, audio.Length, effectiveMax);

        // Compute statistics
        double totalDuration = (double)audio.Length / _options.SampleRate;
        var speakerLabels = segments.Select(s => s.Speaker).Distinct().ToList();
        var stats = ComputeSpeakerStatistics(segments, totalDuration);
        var overlaps = _options.EnableOverlapDetection ? DetectOverlaps(segments) : Array.Empty<OverlapRegion<T>>();

        return new DiarizationResult<T>
        {
            Segments = segments,
            NumSpeakers = speakerLabels.Count,
            SpeakerLabels = speakerLabels,
            TotalDuration = totalDuration,
            OverlapRegions = overlaps,
            SpeakerStats = stats
        };
    }

    public Task<DiarizationResult<T>> DiarizeAsync(Tensor<T> audio, int? numSpeakers = null, int minSpeakers = 1, int maxSpeakers = 10, CancellationToken cancellationToken = default)
        => Task.Run(() => { cancellationToken.ThrowIfCancellationRequested(); return Diarize(audio, numSpeakers, minSpeakers, maxSpeakers); }, cancellationToken);

    public DiarizationResult<T> DiarizeWithKnownSpeakers(Tensor<T> audio, IReadOnlyList<SpeakerProfile<T>> knownSpeakers, bool allowUnknownSpeakers = true)
    {
        ThrowIfDisposed();
        // First diarize without known speakers
        var result = Diarize(audio, maxSpeakers: Math.Max(knownSpeakers.Count, 10));

        // Extract embeddings for each detected speaker
        var speakerEmbeddings = ExtractSpeakerEmbeddings(audio, result);

        // Match detected speakers to known profiles
        var labelMap = new Dictionary<string, string>();
        int unknownCount = 0;
        foreach (var kvp in speakerEmbeddings)
        {
            string bestMatch = allowUnknownSpeakers ? $"Unknown_{unknownCount++}" : "Unknown";
            double bestSim = _options.ClusteringThreshold;

            foreach (var profile in knownSpeakers)
            {
                double sim = NumOps.ToDouble(ComputeCosineSimilarity(kvp.Value, profile.Embedding));
                if (sim > bestSim) { bestSim = sim; bestMatch = profile.SpeakerId; }
            }
            labelMap[kvp.Key] = bestMatch;
        }

        // Remap segment labels
        var remappedSegments = result.Segments.Select(s => new SpeakerSegment<T>
        {
            Speaker = labelMap.TryGetValue(s.Speaker, out var mapped) ? mapped : s.Speaker,
            StartTime = s.StartTime, EndTime = s.EndTime, Confidence = s.Confidence
        }).ToList();

        var newLabels = remappedSegments.Select(s => s.Speaker).Distinct().ToList();
        double totalDuration = (double)audio.Length / _options.SampleRate;
        return new DiarizationResult<T>
        {
            Segments = remappedSegments, NumSpeakers = newLabels.Count, SpeakerLabels = newLabels,
            TotalDuration = totalDuration, OverlapRegions = result.OverlapRegions,
            SpeakerStats = ComputeSpeakerStatistics(remappedSegments, totalDuration)
        };
    }

    public IReadOnlyDictionary<string, Tensor<T>> ExtractSpeakerEmbeddings(Tensor<T> audio, DiarizationResult<T> diarizationResult)
    {
        ThrowIfDisposed();
        var result = new Dictionary<string, Tensor<T>>();
        foreach (var label in diarizationResult.SpeakerLabels)
        {
            var speakerSegs = diarizationResult.Segments.Where(s => s.Speaker == label).ToList();
            var embeddings = new List<Tensor<T>>();
            foreach (var seg in speakerSegs)
            {
                int start = (int)(seg.StartTime * _options.SampleRate);
                int end = Math.Min((int)(seg.EndTime * _options.SampleRate), audio.Length);
                int len = end - start;
                if (len < _options.SampleRate * 0.5) continue;
                var segAudio = new Tensor<T>([len]);
                for (int i = 0; i < len; i++) segAudio[i] = audio[start + i];
                var features = PreprocessAudio(segAudio);
                Tensor<T> emb = IsOnnxMode && OnnxEncoder is not null ? OnnxEncoder.Run(features) : Predict(features);
                embeddings.Add(emb);
            }
            if (embeddings.Count > 0) result[label] = AggregateEmbeddings(embeddings);
        }
        return result;
    }

    public DiarizationResult<T> RefineDiarization(Tensor<T> audio, DiarizationResult<T> previousResult, T mergeThreshold)
    {
        ThrowIfDisposed();
        double threshold = NumOps.ToDouble(mergeThreshold);

        // Re-diarize with adjusted clustering
        var origThreshold = _options.ClusteringThreshold;
        _options.ClusteringThreshold = threshold;
        var result = Diarize(audio);
        _options.ClusteringThreshold = origThreshold;
        return result;
    }

    #endregion

    #region NeuralNetworkBase

    protected override void InitializeLayers()
    {
        if (!_useNativeMode) return;
        if (Architecture.Layers is not null && Architecture.Layers.Count > 0) Layers.AddRange(Architecture.Layers);
        else Layers.AddRange(LayerHelper<T>.CreateDefaultPyAnnoteLayers(
            numMels: _options.NumMels, lstmHiddenSize: _options.LSTMHiddenSize,
            numLSTMLayers: _options.NumLSTMLayers, linearDim: _options.LinearDim,
            maxSpeakersPerChunk: _options.MaxSpeakersPerChunk, dropoutRate: _options.DropoutRate));
    }

    public override Tensor<T> Predict(Tensor<T> input)
    {
        ThrowIfDisposed();
        if (IsOnnxMode && OnnxEncoder is not null) return OnnxEncoder.Run(input);
        var c = input; foreach (var l in Layers) c = l.Forward(c); return c;
    }

    public override void Train(Tensor<T> input, Tensor<T> expected)
    {
        if (IsOnnxMode) throw new NotSupportedException("Training not supported in ONNX mode.");
        SetTrainingMode(true);
        var output = Predict(input);
        var grad = LossFunction.CalculateDerivative(output.ToVector(), expected.ToVector());
        var gt = Tensor<T>.FromVector(grad);
        for (int i = Layers.Count - 1; i >= 0; i--) gt = Layers[i].Backward(gt);
        _optimizer.UpdateParameters(Layers);
        SetTrainingMode(false);
    }

    public override void UpdateParameters(Vector<T> parameters)
    {
        if (!_useNativeMode) throw new NotSupportedException("ONNX mode.");
        int idx = 0; foreach (var l in Layers) { int c = l.ParameterCount; l.UpdateParameters(parameters.Slice(idx, c)); idx += c; }
    }

    protected override Tensor<T> PreprocessAudio(Tensor<T> rawAudio)
    {
        if (MfccExtractor is not null) return MfccExtractor.Extract(rawAudio);
        return rawAudio;
    }

    protected override Tensor<T> PostprocessOutput(Tensor<T> o) => o;

    public override ModelMetadata<T> GetModelMetadata()
    {
        var m = new ModelMetadata<T>
        {
            Name = _useNativeMode ? "PyAnnote-Native" : "PyAnnote-ONNX",
            Description = "pyannote.audio 3.x Speaker Diarization (Plaquet & Bredin, ASRU 2023)",
            ModelType = ModelType.NeuralNetwork, FeatureCount = _options.NumMels,
            Complexity = _options.NumLSTMLayers
        };
        m.AdditionalInfo["MaxSpeakersPerChunk"] = _options.MaxSpeakersPerChunk.ToString();
        m.AdditionalInfo["OverlapDetection"] = _options.EnableOverlapDetection.ToString();
        m.AdditionalInfo["ChunkDuration"] = _options.ChunkDurationSeconds.ToString();
        return m;
    }

    protected override void SerializeNetworkSpecificData(BinaryWriter w)
    {
        w.Write(_useNativeMode); w.Write(_options.ModelPath ?? string.Empty);
        w.Write(_options.SampleRate); w.Write(_options.NumMels); w.Write(_options.EmbeddingDim);
        w.Write(_options.SincNetFilters); w.Write(_options.LSTMHiddenSize); w.Write(_options.NumLSTMLayers);
        w.Write(_options.LinearDim); w.Write(_options.MaxSpeakersPerChunk);
        w.Write(_options.ChunkDurationSeconds); w.Write(_options.ChunkStepSeconds);
        w.Write(_options.ClusteringThreshold); w.Write(_options.MinSegmentDuration);
        w.Write(_options.EnableOverlapDetection); w.Write(_options.DropoutRate);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader r)
    {
        _useNativeMode = r.ReadBoolean(); string mp = r.ReadString(); if (!string.IsNullOrEmpty(mp)) _options.ModelPath = mp;
        _options.SampleRate = r.ReadInt32(); _options.NumMels = r.ReadInt32(); _options.EmbeddingDim = r.ReadInt32();
        _options.SincNetFilters = r.ReadInt32(); _options.LSTMHiddenSize = r.ReadInt32(); _options.NumLSTMLayers = r.ReadInt32();
        _options.LinearDim = r.ReadInt32(); _options.MaxSpeakersPerChunk = r.ReadInt32();
        _options.ChunkDurationSeconds = r.ReadDouble(); _options.ChunkStepSeconds = r.ReadDouble();
        _options.ClusteringThreshold = r.ReadDouble(); _options.MinSegmentDuration = r.ReadDouble();
        _options.EnableOverlapDetection = r.ReadBoolean(); _options.DropoutRate = r.ReadDouble();
        if (!_useNativeMode && _options.ModelPath is { } p && !string.IsNullOrEmpty(p)) OnnxEncoder = new OnnxModel<T>(p, _options.OnnxOptions);
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance() => new PyAnnote<T>(Architecture, _options);

    #endregion

    #region Private Helpers

    private List<AudioChunk<T>> SegmentIntoChunks(Tensor<T> audio)
    {
        var chunks = new List<AudioChunk<T>>();
        int chunkSamples = (int)(_options.ChunkDurationSeconds * _options.SampleRate);
        int stepSamples = (int)(_options.ChunkStepSeconds * _options.SampleRate);

        for (int start = 0; start + chunkSamples <= audio.Length; start += stepSamples)
        {
            var chunk = new Tensor<T>([chunkSamples]);
            for (int i = 0; i < chunkSamples; i++) chunk[i] = audio[start + i];
            chunks.Add(new AudioChunk<T> { Audio = chunk, StartSample = start, StartTime = start / (double)_options.SampleRate, EndTime = (start + chunkSamples) / (double)_options.SampleRate });
        }

        // Handle tail if audio doesn't divide evenly
        if (chunks.Count == 0 || chunks[^1].StartSample + chunkSamples < audio.Length)
        {
            int start = Math.Max(0, audio.Length - chunkSamples);
            int len = Math.Min(chunkSamples, audio.Length - start);
            var chunk = new Tensor<T>([len]);
            for (int i = 0; i < len; i++) chunk[i] = audio[start + i];
            chunks.Add(new AudioChunk<T> { Audio = chunk, StartSample = start, StartTime = start / (double)_options.SampleRate, EndTime = audio.Length / (double)_options.SampleRate });
        }

        return chunks;
    }

    private List<SpeakerSegment<T>> AggregateChunkActivations(List<AudioChunk<T>> chunks, List<Tensor<T>> activations, int totalSamples, int maxSpeakers)
    {
        double totalDuration = (double)totalSamples / _options.SampleRate;
        double frameStep = _options.ChunkStepSeconds;
        int numFrames = (int)(totalDuration / frameStep) + 1;
        int maxSpk = Math.Min(maxSpeakers, _options.MaxSpeakersPerChunk);

        // Create per-frame speaker scores
        var frameScores = new double[numFrames, maxSpk];
        var frameCounts = new int[numFrames];

        for (int ci = 0; ci < chunks.Count && ci < activations.Count; ci++)
        {
            int frameIdx = (int)(chunks[ci].StartTime / frameStep);
            if (frameIdx >= numFrames) frameIdx = numFrames - 1;
            // Simplified: assign the dominant activation values to frame
            for (int s = 0; s < maxSpk && s < activations[ci].Length; s++)
            {
                double val = NumOps.ToDouble(activations[ci][s]);
                frameScores[frameIdx, s] += val;
            }
            frameCounts[frameIdx]++;
        }

        // Normalize and find dominant speaker per frame
        var segments = new List<SpeakerSegment<T>>();
        int currentSpeaker = -1;
        double segStart = 0;

        for (int f = 0; f < numFrames; f++)
        {
            int bestSpk = 0;
            double bestScore = double.NegativeInfinity;
            double count = Math.Max(1, frameCounts[f]);

            for (int s = 0; s < maxSpk; s++)
            {
                double score = frameScores[f, s] / count;
                if (score > bestScore) { bestScore = score; bestSpk = s; }
            }

            if (bestSpk != currentSpeaker)
            {
                if (currentSpeaker >= 0)
                {
                    double segEnd = f * frameStep;
                    if (segEnd - segStart >= _options.MinSegmentDuration)
                    {
                        segments.Add(new SpeakerSegment<T>
                        {
                            Speaker = $"Speaker_{currentSpeaker}",
                            StartTime = segStart, EndTime = segEnd,
                            Confidence = NumOps.FromDouble(1.0)
                        });
                    }
                }
                currentSpeaker = bestSpk;
                segStart = f * frameStep;
            }
        }

        // Add final segment
        if (currentSpeaker >= 0 && totalDuration - segStart >= _options.MinSegmentDuration)
        {
            segments.Add(new SpeakerSegment<T>
            {
                Speaker = $"Speaker_{currentSpeaker}",
                StartTime = segStart, EndTime = totalDuration,
                Confidence = NumOps.FromDouble(1.0)
            });
        }

        return segments;
    }

    private IReadOnlyDictionary<string, SpeakerStatistics<T>> ComputeSpeakerStatistics(
        IReadOnlyList<SpeakerSegment<T>> segments, double totalDuration)
    {
        var stats = new Dictionary<string, SpeakerStatistics<T>>();
        var grouped = segments.GroupBy(s => s.Speaker);
        foreach (var group in grouped)
        {
            var segs = group.ToList();
            double speakingTime = segs.Sum(s => s.Duration);
            stats[group.Key] = new SpeakerStatistics<T>
            {
                TotalSpeakingTime = speakingTime,
                NumTurns = segs.Count,
                AverageTurnDuration = segs.Count > 0 ? speakingTime / segs.Count : 0,
                SpeakingPercentage = totalDuration > 0 ? (speakingTime / totalDuration) * 100 : 0
            };
        }
        return stats;
    }

    private static IReadOnlyList<OverlapRegion<T>> DetectOverlaps(IReadOnlyList<SpeakerSegment<T>> segments)
    {
        var overlaps = new List<OverlapRegion<T>>();
        for (int i = 0; i < segments.Count; i++)
        {
            for (int j = i + 1; j < segments.Count; j++)
            {
                if (segments[i].Speaker == segments[j].Speaker) continue;
                double overlapStart = Math.Max(segments[i].StartTime, segments[j].StartTime);
                double overlapEnd = Math.Min(segments[i].EndTime, segments[j].EndTime);
                if (overlapEnd > overlapStart)
                {
                    overlaps.Add(new OverlapRegion<T>
                    {
                        StartTime = overlapStart, EndTime = overlapEnd,
                        Speakers = new[] { segments[i].Speaker, segments[j].Speaker }
                    });
                }
            }
        }
        return overlaps;
    }

    #endregion

    #region Disposal

    private void ThrowIfDisposed() { if (_disposed) throw new ObjectDisposedException(GetType().FullName ?? nameof(PyAnnote<T>)); }

    protected override void Dispose(bool disposing)
    {
        if (_disposed) return;
        if (disposing) OnnxEncoder?.Dispose();
        _disposed = true;
        base.Dispose(disposing);
    }

    #endregion
}

/// <summary>
/// Represents an audio chunk for pyannote processing.
/// </summary>
internal class AudioChunk<T>
{
    public required Tensor<T> Audio { get; set; }
    public int StartSample { get; set; }
    public double StartTime { get; set; }
    public double EndTime { get; set; }
}
