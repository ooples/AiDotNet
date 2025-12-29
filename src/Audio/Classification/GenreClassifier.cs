using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Audio.Features;
using AiDotNet.Onnx;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Audio.Classification;

/// <summary>
/// Music genre classification model.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Classifies audio into music genres using spectral features and optional neural network models.
/// Supports common genres: rock, pop, hip-hop, jazz, classical, electronic, country, R&amp;B, metal, folk.
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
/// var classifier = new GenreClassifier&lt;float&gt;();
///
/// var audio = LoadAudio("song.wav");
/// var result = classifier.Classify(audio);
///
/// Console.WriteLine($"Genre: {result.PredictedGenre} ({result.Confidence:P0})");
/// foreach (var (genre, prob) in result.TopPredictions)
/// {
///     Console.WriteLine($"  {genre}: {prob:P0}");
/// }
/// </code>
/// </para>
/// </remarks>
public class GenreClassifier<T> : IDisposable
{
    private readonly INumericOperations<T> _numOps;
    private readonly GenreClassifierOptions _options;
    private readonly MfccExtractor<T> _mfccExtractor;
    private readonly SpectralFeatureExtractor<T> _spectralExtractor;
    private readonly OnnxModel<T>? _model;
    private bool _disposed;

    /// <summary>Standard music genres.</summary>
    public static readonly string[] StandardGenres = new[]
    {
        "blues", "classical", "country", "disco", "hiphop",
        "jazz", "metal", "pop", "reggae", "rock"
    };

    /// <summary>
    /// Gets the supported genre labels.
    /// </summary>
    public string[] Genres => _options.CustomGenres ?? StandardGenres;

    /// <summary>
    /// Creates a new GenreClassifier instance.
    /// </summary>
    /// <param name="options">Classification options.</param>
    public GenreClassifier(GenreClassifierOptions? options = null)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _options = options ?? new GenreClassifierOptions();

        _mfccExtractor = new MfccExtractor<T>(new MfccOptions
        {
            SampleRate = _options.SampleRate,
            NumCoefficients = _options.NumMfccs,
            FftSize = _options.FftSize,
            HopLength = _options.HopLength
        });

        _spectralExtractor = new SpectralFeatureExtractor<T>(new SpectralFeatureOptions
        {
            SampleRate = _options.SampleRate,
            FftSize = _options.FftSize,
            HopLength = _options.HopLength,
            FeatureTypes = SpectralFeatureType.Centroid | SpectralFeatureType.Rolloff
        });

        if (_options.ModelPath is not null && _options.ModelPath.Length > 0)
        {
            _model = new OnnxModel<T>(_options.ModelPath, _options.OnnxOptions);
        }
    }

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

        return new GenreClassifier<T>(options);
    }

    /// <summary>
    /// Classifies the genre of an audio clip.
    /// </summary>
    /// <param name="audio">Audio waveform.</param>
    /// <returns>Classification result with predicted genre and probabilities.</returns>
    public GenreClassificationResult Classify(Tensor<T> audio)
    {
        ThrowIfDisposed();

        // Extract features
        var features = ExtractFeatures(audio);

        // Classify
        double[] probabilities;
        if (_model is not null)
        {
            probabilities = ClassifyWithModel(features);
        }
        else
        {
            probabilities = ClassifyWithRules(features);
        }

        // Build result
        var genres = Genres;
        int bestIdx = 0;
        double bestProb = 0;

        for (int i = 0; i < probabilities.Length; i++)
        {
            if (probabilities[i] > bestProb)
            {
                bestProb = probabilities[i];
                bestIdx = i;
            }
        }

        var topPredictions = new List<(string Genre, double Probability)>();
        var sorted = probabilities
            .Select((p, i) => (Genre: genres[i], Probability: p))
            .OrderByDescending(x => x.Probability)
            .Take(_options.TopK);

        foreach (var pred in sorted)
        {
            topPredictions.Add(pred);
        }

        return new GenreClassificationResult
        {
            PredictedGenre = genres[bestIdx],
            Confidence = bestProb,
            AllProbabilities = genres.Zip(probabilities, (g, p) => (g, p))
                .ToDictionary(x => x.g, x => x.p),
            TopPredictions = topPredictions,
            Features = features
        };
    }

    /// <summary>
    /// Classifies audio asynchronously.
    /// </summary>
    public Task<GenreClassificationResult> ClassifyAsync(
        Tensor<T> audio,
        CancellationToken cancellationToken = default)
    {
        return Task.Run(() => Classify(audio), cancellationToken);
    }

    /// <summary>
    /// Classifies multiple audio segments in batch.
    /// </summary>
    public List<GenreClassificationResult> ClassifyBatch(IEnumerable<Tensor<T>> audioSegments)
    {
        return audioSegments.Select(Classify).ToList();
    }

    private GenreFeatures ExtractFeatures(Tensor<T> audio)
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
                sum += _numOps.ToDouble(mfccs[t, c]);
            }
            mfccMean[c] = sum / numFrames;

            double sumSq = 0;
            for (int t = 0; t < numFrames; t++)
            {
                double diff = _numOps.ToDouble(mfccs[t, c]) - mfccMean[c];
                sumSq += diff * diff;
            }
            mfccStd[c] = Math.Sqrt(sumSq / numFrames);
        }

        // Extract spectral features - result is [numFrames, numFeatures]
        // With Centroid | Rolloff, features are in order: [centroid, rolloff]
        var spectralResult = _spectralExtractor.Extract(audio);
        int spectralFrames = spectralResult.Shape[0];

        // Extract centroid and rolloff from columns
        var spectralCentroid = new Tensor<T>([spectralFrames]);
        var spectralRolloff = new Tensor<T>([spectralFrames]);
        for (int t = 0; t < spectralFrames; t++)
        {
            spectralCentroid[t] = spectralResult[t, 0]; // Centroid is first
            spectralRolloff[t] = spectralResult.Shape[1] > 1 ? spectralResult[t, 1] : _numOps.Zero; // Rolloff is second
        }

        // Compute temporal features
        double zeroCrossingRate = ComputeZeroCrossingRate(audio);
        double rmsEnergy = ComputeRmsEnergy(audio);
        double tempo = EstimateTempo(audio);

        return new GenreFeatures
        {
            MfccMean = mfccMean,
            MfccStd = mfccStd,
            SpectralCentroidMean = ComputeMean(spectralCentroid),
            SpectralCentroidStd = ComputeStd(spectralCentroid),
            SpectralRolloffMean = ComputeMean(spectralRolloff),
            ZeroCrossingRate = zeroCrossingRate,
            RmsEnergy = rmsEnergy,
            Tempo = tempo
        };
    }

    private double ComputeMean(Tensor<T> tensor)
    {
        double sum = 0;
        for (int i = 0; i < tensor.Length; i++)
        {
            sum += _numOps.ToDouble(tensor[i]);
        }
        return sum / tensor.Length;
    }

    private double ComputeStd(Tensor<T> tensor)
    {
        double mean = ComputeMean(tensor);
        double sumSq = 0;
        for (int i = 0; i < tensor.Length; i++)
        {
            double diff = _numOps.ToDouble(tensor[i]) - mean;
            sumSq += diff * diff;
        }
        return Math.Sqrt(sumSq / tensor.Length);
    }

    private double ComputeZeroCrossingRate(Tensor<T> audio)
    {
        int crossings = 0;
        for (int i = 1; i < audio.Length; i++)
        {
            double prev = _numOps.ToDouble(audio[i - 1]);
            double curr = _numOps.ToDouble(audio[i]);
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
            double val = _numOps.ToDouble(audio[i]);
            sumSq += val * val;
        }
        return Math.Sqrt(sumSq / audio.Length);
    }

    private double EstimateTempo(Tensor<T> audio)
    {
        // Simple onset-based tempo estimation
        int frameSize = _options.HopLength;
        int numFrames = audio.Length / frameSize;
        if (numFrames < 2) return 120.0; // Default

        var energies = new double[numFrames];
        for (int f = 0; f < numFrames; f++)
        {
            double sum = 0;
            for (int i = 0; i < frameSize && f * frameSize + i < audio.Length; i++)
            {
                double val = _numOps.ToDouble(audio[f * frameSize + i]);
                sum += val * val;
            }
            energies[f] = sum;
        }

        // Compute onset strength (difference in energy)
        var onsets = new double[numFrames - 1];
        for (int f = 1; f < numFrames; f++)
        {
            onsets[f - 1] = Math.Max(0, energies[f] - energies[f - 1]);
        }

        // Find dominant periodicity using autocorrelation
        int minLag = (int)(0.5 * _options.SampleRate / frameSize); // 120 BPM
        int maxLag = (int)(2.0 * _options.SampleRate / frameSize); // 30 BPM
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

    private double[] ClassifyWithModel(GenreFeatures features)
    {
        if (_model is null)
            throw new InvalidOperationException("Model not loaded.");

        // Flatten features to input tensor
        int numFeatures = features.MfccMean.Length * 2 + 5; // MFCCs + spectral + temporal
        var input = new Tensor<T>([1, numFeatures]);

        int idx = 0;
        foreach (var val in features.MfccMean)
            input[0, idx++] = _numOps.FromDouble(val);
        foreach (var val in features.MfccStd)
            input[0, idx++] = _numOps.FromDouble(val);
        input[0, idx++] = _numOps.FromDouble(features.SpectralCentroidMean);
        input[0, idx++] = _numOps.FromDouble(features.SpectralCentroidStd);
        input[0, idx++] = _numOps.FromDouble(features.ZeroCrossingRate);
        input[0, idx++] = _numOps.FromDouble(features.RmsEnergy);
        input[0, idx++] = _numOps.FromDouble(features.Tempo / 200.0); // Normalize

        // Run model
        var output = _model.Run(input);

        // Get probabilities (apply softmax if needed)
        var probs = new double[Genres.Length];
        for (int i = 0; i < Math.Min(output.Length, Genres.Length); i++)
        {
            probs[i] = _numOps.ToDouble(output[i]);
        }

        // Apply softmax
        double maxLogit = probs.Max();
        double sumExp = 0;
        for (int i = 0; i < probs.Length; i++)
        {
            probs[i] = Math.Exp(probs[i] - maxLogit);
            sumExp += probs[i];
        }
        for (int i = 0; i < probs.Length; i++)
        {
            probs[i] /= sumExp;
        }

        return probs;
    }

    private double[] ClassifyWithRules(GenreFeatures features)
    {
        // Rule-based classification using heuristics
        var probs = new double[Genres.Length];
        var genres = Genres;

        for (int i = 0; i < genres.Length; i++)
        {
            probs[i] = ComputeGenreScore(genres[i], features);
        }

        // Normalize to probabilities
        double sum = probs.Sum();
        if (sum > 0)
        {
            for (int i = 0; i < probs.Length; i++)
            {
                probs[i] /= sum;
            }
        }
        else
        {
            // Uniform distribution as fallback
            for (int i = 0; i < probs.Length; i++)
            {
                probs[i] = 1.0 / genres.Length;
            }
        }

        return probs;
    }

    private double ComputeGenreScore(string genre, GenreFeatures features)
    {
        // Heuristic scoring based on feature characteristics
        double score = 1.0;

        double centroid = features.SpectralCentroidMean;
        double zcr = features.ZeroCrossingRate;
        double tempo = features.Tempo;
        double energy = features.RmsEnergy;

        switch (genre.ToLowerInvariant())
        {
            case "classical":
                // Lower tempo, wider dynamic range, lower ZCR
                score *= GaussianScore(tempo, 90, 30);
                score *= GaussianScore(zcr, 0.03, 0.02);
                score *= GaussianScore(centroid, 1500, 500);
                break;

            case "jazz":
                // Medium tempo, complex harmonics
                score *= GaussianScore(tempo, 110, 30);
                score *= GaussianScore(centroid, 2000, 600);
                break;

            case "rock":
                // Higher energy, medium-high tempo
                score *= GaussianScore(tempo, 120, 25);
                score *= GaussianScore(energy, 0.15, 0.08);
                score *= GaussianScore(centroid, 2500, 800);
                break;

            case "metal":
                // High energy, fast tempo, high ZCR
                score *= GaussianScore(tempo, 140, 30);
                score *= GaussianScore(energy, 0.2, 0.08);
                score *= GaussianScore(zcr, 0.12, 0.05);
                break;

            case "pop":
                // Medium tempo, moderate energy
                score *= GaussianScore(tempo, 115, 20);
                score *= GaussianScore(energy, 0.1, 0.05);
                break;

            case "hiphop":
                // Lower tempo, prominent bass
                score *= GaussianScore(tempo, 90, 20);
                score *= GaussianScore(centroid, 1800, 600);
                break;

            case "disco":
            case "electronic":
                // Steady tempo around 120, electronic sounds
                score *= GaussianScore(tempo, 125, 15);
                score *= GaussianScore(centroid, 3000, 800);
                break;

            case "blues":
                // Slow to medium tempo, expressive
                score *= GaussianScore(tempo, 85, 25);
                score *= GaussianScore(centroid, 1600, 500);
                break;

            case "country":
                // Medium tempo, acoustic characteristics
                score *= GaussianScore(tempo, 100, 25);
                score *= GaussianScore(zcr, 0.05, 0.02);
                break;

            case "reggae":
                // Specific tempo range, offbeat emphasis
                score *= GaussianScore(tempo, 80, 20);
                score *= GaussianScore(energy, 0.08, 0.04);
                break;

            default:
                score = 0.5;
                break;
        }

        return Math.Max(score, 0.01);
    }

    private static double GaussianScore(double value, double mean, double std)
    {
        double z = (value - mean) / std;
        return Math.Exp(-0.5 * z * z);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(GetType().FullName);
    }

    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    protected virtual void Dispose(bool disposing)
    {
        if (_disposed) return;

        if (disposing)
        {
            _model?.Dispose();
        }

        _disposed = true;
    }
}

/// <summary>
/// Options for genre classification.
/// </summary>
public class GenreClassifierOptions
{
    /// <summary>Audio sample rate. Default: 22050.</summary>
    public int SampleRate { get; set; } = 22050;

    /// <summary>FFT size. Default: 2048.</summary>
    public int FftSize { get; set; } = 2048;

    /// <summary>Hop length. Default: 512.</summary>
    public int HopLength { get; set; } = 512;

    /// <summary>Number of MFCCs to extract. Default: 20.</summary>
    public int NumMfccs { get; set; } = 20;

    /// <summary>Number of top predictions to return. Default: 3.</summary>
    public int TopK { get; set; } = 3;

    /// <summary>Custom genre labels (optional).</summary>
    public string[]? CustomGenres { get; set; }

    /// <summary>Path to ONNX model file (optional).</summary>
    public string? ModelPath { get; set; }

    /// <summary>ONNX model options.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();
}

/// <summary>
/// Features extracted for genre classification.
/// </summary>
public class GenreFeatures
{
    /// <summary>Mean of MFCC coefficients across time.</summary>
    public required double[] MfccMean { get; init; }

    /// <summary>Standard deviation of MFCC coefficients across time.</summary>
    public required double[] MfccStd { get; init; }

    /// <summary>Mean spectral centroid.</summary>
    public double SpectralCentroidMean { get; init; }

    /// <summary>Standard deviation of spectral centroid.</summary>
    public double SpectralCentroidStd { get; init; }

    /// <summary>Mean spectral rolloff.</summary>
    public double SpectralRolloffMean { get; init; }

    /// <summary>Zero crossing rate.</summary>
    public double ZeroCrossingRate { get; init; }

    /// <summary>RMS energy.</summary>
    public double RmsEnergy { get; init; }

    /// <summary>Estimated tempo in BPM.</summary>
    public double Tempo { get; init; }
}

/// <summary>
/// Result of genre classification.
/// </summary>
public class GenreClassificationResult
{
    /// <summary>Most likely genre.</summary>
    public required string PredictedGenre { get; init; }

    /// <summary>Confidence score for predicted genre (0-1).</summary>
    public double Confidence { get; init; }

    /// <summary>Probabilities for all genres.</summary>
    public required Dictionary<string, double> AllProbabilities { get; init; }

    /// <summary>Top K predictions with probabilities.</summary>
    public required List<(string Genre, double Probability)> TopPredictions { get; init; }

    /// <summary>Extracted features used for classification.</summary>
    public required GenreFeatures Features { get; init; }
}
