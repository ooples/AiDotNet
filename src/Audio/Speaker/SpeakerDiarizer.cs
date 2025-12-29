using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

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
/// var diarizer = new SpeakerDiarizer&lt;float&gt;();
/// var turns = diarizer.Diarize(audioTensor);
/// foreach (var turn in turns)
///     Console.WriteLine($"{turn.SpeakerId}: {turn.StartTime:F2}s - {turn.EndTime:F2}s");
/// </code>
/// </para>
/// </remarks>
public class SpeakerDiarizer<T> : IDisposable
{
    /// <summary>
    /// Gets numeric operations for type T.
    /// </summary>
    protected readonly INumericOperations<T> NumOps;
    private readonly SpeakerEmbeddingExtractor<T> _embeddingExtractor;
    private readonly SpeakerDiarizerOptions _options;
    private bool _disposed;

    /// <summary>
    /// Gets the sample rate.
    /// </summary>
    public int SampleRate => _options.SampleRate;

    /// <summary>
    /// Gets the minimum turn duration in seconds.
    /// </summary>
    public double MinTurnDuration => _options.MinTurnDuration;

    /// <summary>
    /// Gets the clustering threshold.
    /// </summary>
    public double ClusteringThreshold => _options.ClusteringThreshold;

    /// <summary>
    /// Creates a new speaker diarizer.
    /// </summary>
    /// <param name="options">Diarization options.</param>
    public SpeakerDiarizer(SpeakerDiarizerOptions? options = null)
    {
        NumOps = MathHelper.GetNumericOperations<T>();
        _options = options ?? new SpeakerDiarizerOptions();

        _embeddingExtractor = new SpeakerEmbeddingExtractor<T>(new SpeakerEmbeddingOptions
        {
            SampleRate = _options.SampleRate,
            EmbeddingDimension = _options.EmbeddingDimension,
            ModelPath = _options.EmbeddingModelPath
        });
    }

    /// <summary>
    /// Performs diarization on audio.
    /// </summary>
    /// <param name="audio">Audio samples as a tensor.</param>
    /// <returns>List of speaker turns.</returns>
    public DiarizationResult Diarize(Tensor<T> audio)
    {
        ThrowIfDisposed();

        // Segment audio into windows
        var segments = SegmentAudio(audio);

        // Extract embeddings for each segment
        var embeddings = ExtractSegmentEmbeddings(segments);

        // Cluster embeddings
        var labels = ClusterEmbeddings(embeddings);

        // Merge consecutive same-speaker segments
        var turns = CreateSpeakerTurns(segments, labels);

        return new DiarizationResult
        {
            Turns = turns,
            NumSpeakers = labels.Distinct().Count(),
            Duration = (double)audio.Length / _options.SampleRate
        };
    }

    /// <summary>
    /// Performs diarization on audio.
    /// </summary>
    /// <param name="audio">Audio samples as a vector.</param>
    /// <returns>List of speaker turns.</returns>
    public DiarizationResult Diarize(Vector<T> audio)
    {
        var tensor = new Tensor<T>([audio.Length]);
        for (int i = 0; i < audio.Length; i++)
        {
            tensor[i] = audio[i];
        }
        return Diarize(tensor);
    }

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

    private int[] ClusterEmbeddings(List<SpeakerEmbedding<T>> embeddings)
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

        // Apply constraints (min/max speakers)
        int numClusters = labels.Distinct().Count();
        if (_options.MaxSpeakers.HasValue && numClusters > _options.MaxSpeakers.Value)
        {
            labels = MergeClusters(labels, similarityMatrix, embeddings, _options.MaxSpeakers.Value);
        }

        return labels;
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

    private int[] MergeClusters(int[] labels, double[,] similarity,
        List<SpeakerEmbedding<T>> embeddings, int maxClusters)
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

    private List<SpeakerTurn> CreateSpeakerTurns(
        List<AudioSegment<T>> segments, int[] labels)
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

    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName);
    }

    /// <summary>
    /// Disposes resources.
    /// </summary>
    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Disposes managed resources.
    /// </summary>
    protected virtual void Dispose(bool disposing)
    {
        if (_disposed) return;

        if (disposing)
        {
            _embeddingExtractor.Dispose();
        }

        _disposed = true;
    }
}

/// <summary>
/// Represents an audio segment for processing.
/// </summary>
internal class AudioSegment<T>
{
    public required Tensor<T> Audio { get; set; }
    public int StartSample { get; set; }
    public double StartTime { get; set; }
    public double EndTime { get; set; }
}
