using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Audio.Features;
using AiDotNet.Diffusion.Audio;
using AiDotNet.Onnx;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;
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
/// Usage:
/// <code>
/// var classifier = new SceneClassifier&lt;float&gt;();
///
/// var audio = LoadAudio("recording.wav");
/// var result = classifier.Classify(audio);
///
/// Console.WriteLine($"Scene: {result.PredictedScene} ({result.Confidence:P0})");
/// Console.WriteLine($"Category: {result.Category}");
/// </code>
/// </para>
/// </remarks>
public class SceneClassifier<T> : IDisposable
{
    /// <summary>
    /// Gets numeric operations for type T.
    /// </summary>
    protected readonly INumericOperations<T> NumOps;
    private readonly SceneClassifierOptions _options;
    private readonly MelSpectrogram<T> _melSpectrogram;
    private readonly MfccExtractor<T> _mfccExtractor;
    private readonly OnnxModel<T>? _model;
    private bool _disposed;

    /// <summary>Standard acoustic scene labels (DCASE-style).</summary>
    public static readonly string[] StandardScenes = new[]
    {
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
    };

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

    /// <summary>
    /// Gets the supported scene labels.
    /// </summary>
    public string[] Scenes => _options.CustomScenes ?? StandardScenes;

    /// <summary>
    /// Creates a new SceneClassifier instance.
    /// </summary>
    /// <param name="options">Classification options.</param>
    public SceneClassifier(SceneClassifierOptions? options = null)
    {
        NumOps = MathHelper.GetNumericOperations<T>();
        _options = options ?? new SceneClassifierOptions();

        _melSpectrogram = new MelSpectrogram<T>(
            sampleRate: _options.SampleRate,
            nMels: _options.NumMels,
            nFft: _options.FftSize,
            hopLength: _options.HopLength,
            fMin: 0,
            fMax: _options.SampleRate / 2,
            logMel: true);

        _mfccExtractor = new MfccExtractor<T>(new MfccOptions
        {
            SampleRate = _options.SampleRate,
            NumCoefficients = _options.NumMfccs,
            FftSize = _options.FftSize,
            HopLength = _options.HopLength
        });

        if (_options.ModelPath is not null && _options.ModelPath.Length > 0)
        {
            _model = new OnnxModel<T>(_options.ModelPath, _options.OnnxOptions);
        }
    }

    /// <summary>
    /// Creates a SceneClassifier asynchronously with model download.
    /// </summary>
    public static async Task<SceneClassifier<T>> CreateAsync(
        SceneClassifierOptions? options = null,
        IProgress<double>? progress = null,
        CancellationToken cancellationToken = default)
    {
        options ??= new SceneClassifierOptions();

        if (options.ModelPath is null || options.ModelPath.Length == 0)
        {
            var downloader = new OnnxModelDownloader();
            options.ModelPath = await downloader.DownloadAsync(
                "acoustic-scene-classifier",
                "model.onnx",
                progress: progress,
                cancellationToken);
        }

        return new SceneClassifier<T>(options);
    }

    /// <summary>
    /// Classifies the acoustic scene of an audio recording.
    /// </summary>
    /// <param name="audio">Audio waveform.</param>
    /// <returns>Classification result with predicted scene and probabilities.</returns>
    public SceneClassificationResult Classify(Tensor<T> audio)
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
        var scenes = Scenes;
        int bestIdx = 0;
        double bestProb = 0;

        for (int i = 0; i < probabilities.Length && i < scenes.Length; i++)
        {
            if (probabilities[i] > bestProb)
            {
                bestProb = probabilities[i];
                bestIdx = i;
            }
        }

        string predictedScene = bestIdx < scenes.Length ? scenes[bestIdx] : "unknown";
        string category = GetCategory(predictedScene);

        var topPredictions = new List<(string Scene, double Probability)>();
        var sorted = probabilities
            .Select((p, i) => (Scene: i < scenes.Length ? scenes[i] : "unknown", Probability: p))
            .OrderByDescending(x => x.Probability)
            .Take(_options.TopK);

        foreach (var pred in sorted)
        {
            topPredictions.Add(pred);
        }

        return new SceneClassificationResult
        {
            PredictedScene = predictedScene,
            Category = category,
            Confidence = bestProb,
            AllProbabilities = scenes.Zip(probabilities, (s, p) => (s, p))
                .ToDictionary(x => x.s, x => x.p),
            TopPredictions = topPredictions,
            Features = features
        };
    }

    /// <summary>
    /// Classifies audio asynchronously.
    /// </summary>
    public Task<SceneClassificationResult> ClassifyAsync(
        Tensor<T> audio,
        CancellationToken cancellationToken = default)
    {
        return Task.Run(() => Classify(audio), cancellationToken);
    }

    /// <summary>
    /// Classifies scene category (indoor, outdoor, transportation).
    /// </summary>
    public (string Category, double Confidence) ClassifyCategory(Tensor<T> audio)
    {
        var result = Classify(audio);
        return (result.Category, result.Confidence);
    }

    private SceneFeatures ExtractFeatures(Tensor<T> audio)
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
            // Avoid division by zero for single-frame audio
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

        // Compute band energies (for scene-specific patterns)
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
        // Compute contrast between peaks and valleys in spectrum
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

    private double[] ClassifyWithModel(SceneFeatures features)
    {
        if (_model is null)
            throw new InvalidOperationException("Model not loaded.");

        // Flatten features
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

        var output = _model.Run(input);

        var probs = new double[Scenes.Length];
        for (int i = 0; i < Math.Min(output.Length, Scenes.Length); i++)
        {
            probs[i] = NumOps.ToDouble(output[i]);
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

    private double[] ClassifyWithRules(SceneFeatures features)
    {
        var scenes = Scenes;
        var probs = new double[scenes.Length];

        for (int i = 0; i < scenes.Length; i++)
        {
            probs[i] = ComputeSceneScore(scenes[i], features);
        }

        // Normalize
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
            for (int i = 0; i < probs.Length; i++)
            {
                probs[i] = 1.0 / scenes.Length;
            }
        }

        return probs;
    }

    private double ComputeSceneScore(string scene, SceneFeatures features)
    {
        double score = 1.0;
        string lower = scene.ToLowerInvariant();

        // Indoor scenes: lower energy variance, lower ZCR
        if (lower.Contains("office") || lower.Contains("library") ||
            lower.Contains("classroom") || lower.Contains("home"))
        {
            score *= GaussianScore(features.EnergyVariance, 0.1, 0.1);
            score *= GaussianScore(features.ZeroCrossingRate, 0.03, 0.02);
            score *= GaussianScore(features.RmsEnergy, 0.05, 0.03);
        }

        // Outdoor urban: higher energy, traffic patterns
        if (lower.Contains("street") || lower.Contains("traffic") ||
            lower.Contains("city") || lower.Contains("construction"))
        {
            score *= GaussianScore(features.RmsEnergy, 0.1, 0.05);
            score *= GaussianScore(features.BandEnergies[0], 0.2, 0.1); // More low freq
            score *= GaussianScore(features.EnergyVariance, 0.3, 0.2);
        }

        // Nature scenes: moderate variance, natural spectral patterns
        if (lower.Contains("park") || lower.Contains("forest") ||
            lower.Contains("beach") || lower.Contains("countryside"))
        {
            score *= GaussianScore(features.SpectralFlatness, 0.3, 0.2);
            score *= GaussianScore(features.EnergyVariance, 0.2, 0.15);
        }

        // Transportation: specific spectral patterns
        if (lower.Contains("metro") || lower.Contains("train") ||
            lower.Contains("bus") || lower.Contains("car"))
        {
            score *= GaussianScore(features.BandEnergies[0], 0.25, 0.1); // Low freq (engine)
            score *= GaussianScore(features.SpectralFlatness, 0.4, 0.2);
        }

        if (lower.Contains("airport") || lower.Contains("airplane"))
        {
            score *= GaussianScore(features.SpectralFlatness, 0.5, 0.2);
            score *= GaussianScore(features.RmsEnergy, 0.15, 0.08);
        }

        // Mall/restaurant: moderate background noise
        if (lower.Contains("mall") || lower.Contains("restaurant") ||
            lower.Contains("cafe"))
        {
            score *= GaussianScore(features.SpectralCentroid, 50, 20);
            score *= GaussianScore(features.EnergyVariance, 0.15, 0.1);
        }

        return Math.Max(score, 0.01);
    }

    private static double GaussianScore(double value, double mean, double std)
    {
        double z = (value - mean) / std;
        return Math.Exp(-0.5 * z * z);
    }

    private static string GetCategory(string scene)
    {
        if (SceneCategories.TryGetValue(scene.ToLowerInvariant(), out var category))
        {
            return category;
        }
        return "unknown";
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
