using AiDotNet.Audio.Features;
using AiDotNet.Onnx;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Audio.Speaker;

/// <summary>
/// Extracts speaker embeddings (d-vectors) from audio for speaker recognition.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Speaker embeddings are compact vector representations that capture the
/// unique characteristics of a speaker's voice. These can be used for
/// speaker verification (is this the same person?) and speaker identification
/// (who is speaking?).
/// </para>
/// <para><b>For Beginners:</b> Each person's voice has unique characteristics
/// like pitch, rhythm, and timbre (tone color). This class converts audio into
/// a numerical "fingerprint" of the speaker's voice.
///
/// These embeddings are vectors (lists of numbers) that are:
/// - Close together for the same speaker
/// - Far apart for different speakers
///
/// Usage:
/// <code>
/// var extractor = new SpeakerEmbeddingExtractor&lt;float&gt;();
/// var embedding1 = extractor.Extract(audio1);
/// var embedding2 = extractor.Extract(audio2);
/// double similarity = embedding1.CosineSimilarity(embedding2);
/// if (similarity &gt; 0.7)
///     Console.WriteLine("Same speaker!");
/// </code>
/// </para>
/// </remarks>
public class SpeakerEmbeddingExtractor<T> : IDisposable
{
    /// <summary>
    /// Gets numeric operations for type T.
    /// </summary>
    protected readonly INumericOperations<T> NumOps;
    private readonly MfccExtractor<T> _mfccExtractor;
    private readonly OnnxModel<T>? _onnxModel;
    private readonly SpeakerEmbeddingOptions _options;
    private bool _disposed;

    /// <summary>
    /// Gets the embedding dimension.
    /// </summary>
    public int EmbeddingDimension => _options.EmbeddingDimension;

    /// <summary>
    /// Gets whether a neural model is loaded.
    /// </summary>
    public bool HasNeuralModel => _onnxModel?.IsLoaded == true;

    /// <summary>
    /// Gets the expected sample rate for input audio.
    /// </summary>
    public int SampleRate => _options.SampleRate;

    /// <summary>
    /// Gets whether this model is running in ONNX inference mode.
    /// </summary>
    public bool IsOnnxMode => _onnxModel is not null;

    /// <summary>
    /// Creates a new speaker embedding extractor.
    /// </summary>
    /// <param name="options">Extraction options.</param>
    public SpeakerEmbeddingExtractor(SpeakerEmbeddingOptions? options = null)
    {
        NumOps = MathHelper.GetNumericOperations<T>();
        _options = options ?? new SpeakerEmbeddingOptions();

        _mfccExtractor = new MfccExtractor<T>(new MfccOptions
        {
            SampleRate = _options.SampleRate,
            FftSize = _options.FftSize,
            HopLength = _options.HopLength,
            NumCoefficients = _options.NumMfcc,
            AppendDelta = true,
            AppendDeltaDelta = true
        });

        // Load neural model if path provided
        if (_options.ModelPath is not null && _options.ModelPath.Length > 0)
        {
            _onnxModel = new OnnxModel<T>(_options.ModelPath, _options.OnnxOptions);
        }
    }

    /// <summary>
    /// Extracts a speaker embedding from audio.
    /// </summary>
    /// <param name="audio">Audio samples as a tensor.</param>
    /// <returns>Speaker embedding vector.</returns>
    public SpeakerEmbedding<T> Extract(Tensor<T> audio)
    {
        ThrowIfDisposed();

        if (_onnxModel is not null)
        {
            return ExtractWithNeuralModel(audio);
        }
        else
        {
            return ExtractWithMfccStatistics(audio);
        }
    }

    /// <summary>
    /// Extracts a speaker embedding from audio.
    /// </summary>
    /// <param name="audio">Audio samples as a vector.</param>
    /// <returns>Speaker embedding vector.</returns>
    public SpeakerEmbedding<T> Extract(Vector<T> audio)
    {
        var tensor = new Tensor<T>([audio.Length]);
        for (int i = 0; i < audio.Length; i++)
        {
            tensor[i] = audio[i];
        }
        return Extract(tensor);
    }

    /// <summary>
    /// Extracts embeddings from multiple audio segments.
    /// </summary>
    /// <param name="segments">List of audio segments.</param>
    /// <returns>List of speaker embeddings.</returns>
    public List<SpeakerEmbedding<T>> ExtractBatch(IEnumerable<Tensor<T>> segments)
    {
        return segments.Select(Extract).ToList();
    }

    private SpeakerEmbedding<T> ExtractWithNeuralModel(Tensor<T> audio)
    {
        if (_onnxModel is null)
            throw new InvalidOperationException("Neural model not loaded.");

        // Extract MFCCs first
        var mfccs = _mfccExtractor.Extract(audio);

        // Run through neural model
        var output = _onnxModel.Run(mfccs);

        // Extract embedding from output
        var embedding = new T[_options.EmbeddingDimension];
        for (int i = 0; i < Math.Min(embedding.Length, output.Length); i++)
        {
            embedding[i] = output[i];
        }

        return new SpeakerEmbedding<T>
        {
            Vector = embedding,
            Duration = (double)audio.Length / _options.SampleRate,
            NumFrames = mfccs.Shape[0]
        };
    }

    private SpeakerEmbedding<T> ExtractWithMfccStatistics(Tensor<T> audio)
    {
        // Extract MFCCs
        var mfccs = _mfccExtractor.Extract(audio);
        int numFrames = mfccs.Shape[0];
        int numFeatures = mfccs.Shape[1]; // MFCC + deltas + delta-deltas

        // Compute statistics as embedding
        // Mean, std, min, max for each MFCC coefficient
        int statsPerCoef = 4; // mean, std, min, max
        int embeddingDim = Math.Min(_options.EmbeddingDimension, numFeatures * statsPerCoef);

        var embedding = new T[_options.EmbeddingDimension];
        int idx = 0;

        for (int c = 0; c < numFeatures && idx < embeddingDim; c++)
        {
            // Collect values for this coefficient
            var values = new double[numFrames];
            for (int f = 0; f < numFrames; f++)
            {
                values[f] = NumOps.ToDouble(mfccs[f, c]);
            }

            // Compute statistics
            double mean = values.Average();
            double std = Math.Sqrt(values.Select(v => (v - mean) * (v - mean)).Average());
            double min = values.Min();
            double max = values.Max();

            if (idx < embeddingDim) embedding[idx++] = NumOps.FromDouble(mean);
            if (idx < embeddingDim) embedding[idx++] = NumOps.FromDouble(std);
            if (idx < embeddingDim) embedding[idx++] = NumOps.FromDouble(min);
            if (idx < embeddingDim) embedding[idx++] = NumOps.FromDouble(max);
        }

        // Normalize embedding
        NormalizeEmbeddingArray(embedding);

        return new SpeakerEmbedding<T>
        {
            Vector = embedding,
            Duration = (double)audio.Length / _options.SampleRate,
            NumFrames = numFrames
        };
    }

    private void NormalizeEmbeddingArray(T[] embedding)
    {
        // L2 normalization
        double sumSquared = 0;
        for (int i = 0; i < embedding.Length; i++)
        {
            double v = NumOps.ToDouble(embedding[i]);
            sumSquared += v * v;
        }

        double norm = Math.Sqrt(sumSquared);
        if (norm > 1e-10)
        {
            for (int i = 0; i < embedding.Length; i++)
            {
                double v = NumOps.ToDouble(embedding[i]) / norm;
                embedding[i] = NumOps.FromDouble(v);
            }
        }
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
            _onnxModel?.Dispose();
        }

        _disposed = true;
    }

    /// <summary>
    /// Extracts speaker embedding from audio as a Tensor.
    /// </summary>
    /// <param name="audio">Audio samples as a tensor.</param>
    /// <returns>Speaker embedding tensor.</returns>
    public Tensor<T> ExtractTensor(Tensor<T> audio)
    {
        var result = Extract(audio);
        var tensor = new Tensor<T>([result.Vector.Length]);
        for (int i = 0; i < result.Vector.Length; i++)
        {
            tensor[i] = result.Vector[i];
        }
        return tensor;
    }

    /// <summary>
    /// Computes cosine similarity between two speaker embeddings.
    /// </summary>
    /// <param name="embedding1">First embedding.</param>
    /// <param name="embedding2">Second embedding.</param>
    /// <returns>Cosine similarity (0-1 for normalized embeddings).</returns>
    public T ComputeSimilarity(SpeakerEmbedding<T> embedding1, SpeakerEmbedding<T> embedding2)
    {
        return NumOps.FromDouble(embedding1.CosineSimilarity(embedding2));
    }
}

/// <summary>
/// Represents a speaker embedding vector.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
public class SpeakerEmbedding<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Gets or sets the embedding vector.
    /// </summary>
    public required T[] Vector { get; set; }

    /// <summary>
    /// Gets or sets the duration of the source audio in seconds.
    /// </summary>
    public double Duration { get; set; }

    /// <summary>
    /// Gets or sets the number of frames used.
    /// </summary>
    public int NumFrames { get; set; }

    /// <summary>
    /// Computes cosine similarity with another embedding.
    /// </summary>
    public double CosineSimilarity(SpeakerEmbedding<T> other)
    {
        double dot = 0;
        double norm1 = 0;
        double norm2 = 0;

        int len = Math.Min(Vector.Length, other.Vector.Length);
        for (int i = 0; i < len; i++)
        {
            double v1 = NumOps.ToDouble(Vector[i]);
            double v2 = NumOps.ToDouble(other.Vector[i]);
            dot += v1 * v2;
            norm1 += v1 * v1;
            norm2 += v2 * v2;
        }

        double denominator = Math.Sqrt(norm1 * norm2);
        if (denominator < 1e-10) return 0;

        return dot / denominator;
    }

    /// <summary>
    /// Computes Euclidean distance from another embedding.
    /// </summary>
    public double EuclideanDistance(SpeakerEmbedding<T> other)
    {
        double sumSquared = 0;
        int len = Math.Min(Vector.Length, other.Vector.Length);

        for (int i = 0; i < len; i++)
        {
            double diff = NumOps.ToDouble(Vector[i]) - NumOps.ToDouble(other.Vector[i]);
            sumSquared += diff * diff;
        }

        return Math.Sqrt(sumSquared);
    }
}

/// <summary>
/// Configuration options for speaker embedding extraction.
/// </summary>
public class SpeakerEmbeddingOptions
{
    /// <summary>
    /// Gets or sets the sample rate.
    /// </summary>
    public int SampleRate { get; set; } = 16000;

    /// <summary>
    /// Gets or sets the FFT size.
    /// </summary>
    public int FftSize { get; set; } = 512;

    /// <summary>
    /// Gets or sets the hop length.
    /// </summary>
    public int HopLength { get; set; } = 160;

    /// <summary>
    /// Gets or sets the number of MFCC coefficients.
    /// </summary>
    public int NumMfcc { get; set; } = 40;

    /// <summary>
    /// Gets or sets the embedding dimension.
    /// </summary>
    public int EmbeddingDimension { get; set; } = 256;

    /// <summary>
    /// Gets or sets the path to the neural embedding model.
    /// </summary>
    public string? ModelPath { get; set; }

    /// <summary>
    /// Gets or sets the ONNX options.
    /// </summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();
}
