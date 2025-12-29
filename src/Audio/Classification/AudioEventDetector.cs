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
/// Audio event detection model for identifying sounds in audio (AudioSet-style).
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Detects various audio events like speech, music, environmental sounds, and more.
/// Based on AudioSet ontology with 527+ event classes organized hierarchically.
/// </para>
/// <para><b>For Beginners:</b> Audio event detection answers "What sounds are in this audio?":
/// <list type="bullet">
/// <item>Human sounds: speech, laughter, coughing, footsteps</item>
/// <item>Animal sounds: dog barking, bird singing, cat meowing</item>
/// <item>Music: instruments, genres, singing</item>
/// <item>Environmental: traffic, rain, wind, construction</item>
/// </list>
///
/// Usage:
/// <code>
/// var detector = new AudioEventDetector&lt;float&gt;();
///
/// var audio = LoadAudio("recording.wav");
/// var events = detector.Detect(audio);
///
/// foreach (var evt in events)
/// {
///     Console.WriteLine($"{evt.Label}: {evt.Confidence:P0} at {evt.StartTime:F2}s - {evt.EndTime:F2}s");
/// }
/// </code>
/// </para>
/// </remarks>
public class AudioEventDetector<T> : IDisposable
{
    private readonly INumericOperations<T> _numOps;
    private readonly AudioEventDetectorOptions _options;
    private readonly MelSpectrogram<T> _melSpectrogram;
    private readonly OnnxModel<T>? _model;
    private bool _disposed;

    /// <summary>Common audio event categories from AudioSet.</summary>
    public static readonly string[] CommonEventLabels = new[]
    {
        // Human sounds
        "Speech", "Male speech", "Female speech", "Child speech",
        "Conversation", "Narration", "Laughter", "Crying", "Cough",
        "Sneeze", "Breathing", "Sigh", "Yawn", "Snoring",

        // Animal sounds
        "Dog", "Dog barking", "Dog howl", "Cat", "Cat meowing",
        "Bird", "Bird song", "Chirp", "Crow", "Rooster",

        // Music
        "Music", "Singing", "Guitar", "Piano", "Drums",
        "Bass guitar", "Electric guitar", "Violin", "Flute",

        // Environmental
        "Rain", "Thunder", "Wind", "Water", "Fire",
        "Traffic", "Car", "Car horn", "Siren", "Train",
        "Airplane", "Helicopter", "Engine",

        // Household
        "Door", "Knock", "Bell", "Telephone", "Alarm",
        "Keyboard typing", "Mouse click", "Printer",

        // Other
        "Silence", "Noise", "Static", "White noise", "Pink noise"
    };

    /// <summary>
    /// Gets the supported event labels.
    /// </summary>
    public string[] EventLabels => _options.CustomLabels ?? CommonEventLabels;

    /// <summary>
    /// Creates a new AudioEventDetector instance.
    /// </summary>
    /// <param name="options">Detection options.</param>
    public AudioEventDetector(AudioEventDetectorOptions? options = null)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _options = options ?? new AudioEventDetectorOptions();

        _melSpectrogram = new MelSpectrogram<T>(
            sampleRate: _options.SampleRate,
            nMels: _options.NumMels,
            nFft: _options.FftSize,
            hopLength: _options.HopLength,
            fMin: _options.FMin,
            fMax: _options.FMax,
            logMel: true);

        if (_options.ModelPath is not null && _options.ModelPath.Length > 0)
        {
            _model = new OnnxModel<T>(_options.ModelPath, _options.OnnxOptions);
        }
    }

    /// <summary>
    /// Creates an AudioEventDetector asynchronously with model download.
    /// </summary>
    public static async Task<AudioEventDetector<T>> CreateAsync(
        AudioEventDetectorOptions? options = null,
        IProgress<double>? progress = null,
        CancellationToken cancellationToken = default)
    {
        options ??= new AudioEventDetectorOptions();

        if (options.ModelPath is null || options.ModelPath.Length == 0)
        {
            var downloader = new OnnxModelDownloader();
            options.ModelPath = await downloader.DownloadAsync(
                "audio-event-detector",
                "model.onnx",
                progress: progress,
                cancellationToken);
        }

        return new AudioEventDetector<T>(options);
    }

    /// <summary>
    /// Detects audio events in the given audio.
    /// </summary>
    /// <param name="audio">Audio waveform.</param>
    /// <returns>List of detected audio events.</returns>
    public List<AudioEvent> Detect(Tensor<T> audio)
    {
        ThrowIfDisposed();

        // Split audio into windows
        var windows = SplitIntoWindows(audio);
        var allEvents = new List<AudioEvent>();

        for (int windowIdx = 0; windowIdx < windows.Count; windowIdx++)
        {
            var window = windows[windowIdx];
            double startTime = windowIdx * _options.WindowSize * (1 - _options.WindowOverlap);

            // Extract features
            var melSpec = _melSpectrogram.Forward(window);

            // Classify
            double[] scores;
            if (_model is not null)
            {
                scores = ClassifyWithModel(melSpec);
            }
            else
            {
                scores = ClassifyWithRules(melSpec, window);
            }

            // Get events above threshold
            var labels = EventLabels;
            for (int i = 0; i < scores.Length && i < labels.Length; i++)
            {
                if (scores[i] >= _options.Threshold)
                {
                    allEvents.Add(new AudioEvent
                    {
                        Label = labels[i],
                        Confidence = scores[i],
                        StartTime = startTime,
                        EndTime = startTime + _options.WindowSize
                    });
                }
            }
        }

        // Merge overlapping events of same class
        return MergeEvents(allEvents);
    }

    /// <summary>
    /// Detects audio events asynchronously.
    /// </summary>
    public Task<List<AudioEvent>> DetectAsync(
        Tensor<T> audio,
        CancellationToken cancellationToken = default)
    {
        return Task.Run(() => Detect(audio), cancellationToken);
    }

    /// <summary>
    /// Detects a single frame (no windowing).
    /// </summary>
    /// <param name="audio">Audio waveform for a single frame.</param>
    /// <returns>Dictionary of event labels to confidence scores.</returns>
    public Dictionary<string, double> DetectFrame(Tensor<T> audio)
    {
        ThrowIfDisposed();

        var melSpec = _melSpectrogram.Forward(audio);

        double[] scores;
        if (_model is not null)
        {
            scores = ClassifyWithModel(melSpec);
        }
        else
        {
            scores = ClassifyWithRules(melSpec, audio);
        }

        var result = new Dictionary<string, double>();
        var labels = EventLabels;

        for (int i = 0; i < scores.Length && i < labels.Length; i++)
        {
            result[labels[i]] = scores[i];
        }

        return result;
    }

    /// <summary>
    /// Gets the top K events for a single frame.
    /// </summary>
    public List<(string Label, double Confidence)> DetectTopK(Tensor<T> audio, int topK = 5)
    {
        var allScores = DetectFrame(audio);
        return allScores
            .OrderByDescending(x => x.Value)
            .Take(topK)
            .Select(x => (x.Key, x.Value))
            .ToList();
    }

    private List<Tensor<T>> SplitIntoWindows(Tensor<T> audio)
    {
        var windows = new List<Tensor<T>>();
        int windowSamples = (int)(_options.WindowSize * _options.SampleRate);
        int hopSamples = (int)(windowSamples * (1 - _options.WindowOverlap));

        for (int start = 0; start + windowSamples <= audio.Length; start += hopSamples)
        {
            var window = new Tensor<T>([windowSamples]);
            for (int i = 0; i < windowSamples; i++)
            {
                window[i] = audio[start + i];
            }
            windows.Add(window);
        }

        // Handle last partial window
        if (windows.Count == 0 && audio.Length > 0)
        {
            var window = new Tensor<T>([audio.Length]);
            for (int i = 0; i < audio.Length; i++)
            {
                window[i] = audio[i];
            }
            windows.Add(window);
        }

        return windows;
    }

    private double[] ClassifyWithModel(Tensor<T> melSpec)
    {
        if (_model is null)
            throw new InvalidOperationException("Model not loaded.");

        // Prepare input (add batch and channel dimensions)
        var input = new Tensor<T>([1, 1, melSpec.Shape[0], melSpec.Shape[1]]);
        for (int t = 0; t < melSpec.Shape[0]; t++)
        {
            for (int f = 0; f < melSpec.Shape[1]; f++)
            {
                input[0, 0, t, f] = melSpec[t, f];
            }
        }

        var output = _model.Run(input);

        // Apply sigmoid to get probabilities
        var scores = new double[EventLabels.Length];
        for (int i = 0; i < Math.Min(output.Length, scores.Length); i++)
        {
            double logit = _numOps.ToDouble(output[i]);
            scores[i] = 1.0 / (1.0 + Math.Exp(-logit)); // Sigmoid
        }

        return scores;
    }

    private double[] ClassifyWithRules(Tensor<T> melSpec, Tensor<T> audio)
    {
        var scores = new double[EventLabels.Length];
        var labels = EventLabels;

        // Compute basic features
        double energy = ComputeEnergy(audio);
        double zcr = ComputeZeroCrossingRate(audio);
        double spectralCentroid = ComputeSpectralCentroid(melSpec);
        double spectralFlatness = ComputeSpectralFlatness(melSpec);
        double lowFreqEnergy = ComputeBandEnergy(melSpec, 0, 10);
        double highFreqEnergy = ComputeBandEnergy(melSpec, melSpec.Shape[1] - 20, melSpec.Shape[1]);

        for (int i = 0; i < labels.Length; i++)
        {
            scores[i] = ComputeEventScore(labels[i], energy, zcr, spectralCentroid,
                spectralFlatness, lowFreqEnergy, highFreqEnergy);
        }

        return scores;
    }

    private double ComputeEnergy(Tensor<T> audio)
    {
        double sum = 0;
        for (int i = 0; i < audio.Length; i++)
        {
            double val = _numOps.ToDouble(audio[i]);
            sum += val * val;
        }
        return sum / audio.Length;
    }

    private double ComputeZeroCrossingRate(Tensor<T> audio)
    {
        int crossings = 0;
        for (int i = 1; i < audio.Length; i++)
        {
            double prev = _numOps.ToDouble(audio[i - 1]);
            double curr = _numOps.ToDouble(audio[i]);
            if ((prev >= 0 && curr < 0) || (prev < 0 && curr >= 0))
                crossings++;
        }
        return (double)crossings / audio.Length;
    }

    private double ComputeSpectralCentroid(Tensor<T> melSpec)
    {
        double weightedSum = 0;
        double totalSum = 0;

        for (int t = 0; t < melSpec.Shape[0]; t++)
        {
            for (int f = 0; f < melSpec.Shape[1]; f++)
            {
                double mag = _numOps.ToDouble(melSpec[t, f]);
                weightedSum += f * mag;
                totalSum += mag;
            }
        }

        return totalSum > 0 ? weightedSum / totalSum : 0;
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
                double mag = Math.Max(_numOps.ToDouble(melSpec[t, f]), 1e-10);
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

    private double ComputeBandEnergy(Tensor<T> melSpec, int startBin, int endBin)
    {
        double sum = 0;
        int count = 0;

        for (int t = 0; t < melSpec.Shape[0]; t++)
        {
            for (int f = startBin; f < endBin && f < melSpec.Shape[1]; f++)
            {
                sum += _numOps.ToDouble(melSpec[t, f]);
                count++;
            }
        }

        return count > 0 ? sum / count : 0;
    }

    private static double ComputeEventScore(
        string label,
        double energy,
        double zcr,
        double centroid,
        double flatness,
        double lowEnergy,
        double highEnergy)
    {
        double score = 0;
        string lowerLabel = label.ToLowerInvariant();

        // Silence detection
        if (lowerLabel == "silence")
        {
            return energy < 0.0001 ? 0.9 : 0.1;
        }

        // Noise detection
        if (lowerLabel.Contains("noise"))
        {
            return flatness > 0.8 ? 0.7 : 0.2;
        }

        // Speech detection (moderate ZCR, moderate centroid)
        if (lowerLabel.Contains("speech") || lowerLabel.Contains("conversation") ||
            lowerLabel.Contains("narration"))
        {
            if (energy > 0.001 && zcr > 0.02 && zcr < 0.15 && centroid > 30 && centroid < 80)
            {
                score = 0.6;
            }
        }

        // Music detection (wider spectrum, energy patterns)
        if (lowerLabel.Contains("music") || lowerLabel.Contains("singing") ||
            lowerLabel.Contains("guitar") || lowerLabel.Contains("piano"))
        {
            if (energy > 0.005 && flatness < 0.5)
            {
                score = 0.5;
            }
        }

        // Animal sounds
        if (lowerLabel.Contains("dog") || lowerLabel.Contains("bark"))
        {
            if (energy > 0.01 && zcr > 0.05 && zcr < 0.2)
            {
                score = 0.4;
            }
        }

        if (lowerLabel.Contains("bird") || lowerLabel.Contains("chirp"))
        {
            if (highEnergy > lowEnergy * 2 && zcr > 0.1)
            {
                score = 0.4;
            }
        }

        // Environmental
        if (lowerLabel.Contains("rain") || lowerLabel.Contains("water"))
        {
            if (flatness > 0.6 && energy > 0.001)
            {
                score = 0.4;
            }
        }

        if (lowerLabel.Contains("traffic") || lowerLabel.Contains("car") ||
            lowerLabel.Contains("engine"))
        {
            if (lowEnergy > highEnergy && energy > 0.005)
            {
                score = 0.4;
            }
        }

        return Math.Max(score, 0.05); // Minimum baseline
    }

    private List<AudioEvent> MergeEvents(List<AudioEvent> events)
    {
        if (events.Count == 0) return events;

        // Group by label
        var grouped = events.GroupBy(e => e.Label);
        var merged = new List<AudioEvent>();

        foreach (var group in grouped)
        {
            var sortedEvents = group.OrderBy(e => e.StartTime).ToList();
            var currentEvent = sortedEvents[0];

            for (int i = 1; i < sortedEvents.Count; i++)
            {
                var next = sortedEvents[i];

                // Check if events overlap or are adjacent
                if (next.StartTime <= currentEvent.EndTime + 0.1)
                {
                    // Merge
                    currentEvent = new AudioEvent
                    {
                        Label = currentEvent.Label,
                        StartTime = currentEvent.StartTime,
                        EndTime = Math.Max(currentEvent.EndTime, next.EndTime),
                        Confidence = Math.Max(currentEvent.Confidence, next.Confidence)
                    };
                }
                else
                {
                    merged.Add(currentEvent);
                    currentEvent = next;
                }
            }

            merged.Add(currentEvent);
        }

        return merged.OrderBy(e => e.StartTime).ToList();
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
/// Options for audio event detection.
/// </summary>
public class AudioEventDetectorOptions
{
    /// <summary>Audio sample rate. Default: 16000.</summary>
    public int SampleRate { get; set; } = 16000;

    /// <summary>FFT size. Default: 512.</summary>
    public int FftSize { get; set; } = 512;

    /// <summary>Hop length. Default: 160.</summary>
    public int HopLength { get; set; } = 160;

    /// <summary>Number of mel bands. Default: 64.</summary>
    public int NumMels { get; set; } = 64;

    /// <summary>Minimum frequency for mel filterbank. Default: 50.</summary>
    public int FMin { get; set; } = 50;

    /// <summary>Maximum frequency for mel filterbank. Default: 8000.</summary>
    public int FMax { get; set; } = 8000;

    /// <summary>Window size in seconds. Default: 1.0.</summary>
    public double WindowSize { get; set; } = 1.0;

    /// <summary>Window overlap ratio (0-1). Default: 0.5.</summary>
    public double WindowOverlap { get; set; } = 0.5;

    /// <summary>Confidence threshold for event detection. Default: 0.3.</summary>
    public double Threshold { get; set; } = 0.3;

    /// <summary>Custom event labels (optional).</summary>
    public string[]? CustomLabels { get; set; }

    /// <summary>Path to ONNX model file (optional).</summary>
    public string? ModelPath { get; set; }

    /// <summary>ONNX model options.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();
}

/// <summary>
/// Represents a detected audio event.
/// </summary>
public class AudioEvent
{
    /// <summary>Event label/category.</summary>
    public required string Label { get; init; }

    /// <summary>Confidence score (0-1).</summary>
    public double Confidence { get; init; }

    /// <summary>Event start time in seconds.</summary>
    public double StartTime { get; init; }

    /// <summary>Event end time in seconds.</summary>
    public double EndTime { get; init; }

    /// <summary>Event duration in seconds.</summary>
    public double Duration => EndTime - StartTime;

    public override string ToString() =>
        $"{Label} ({Confidence:P0}): {StartTime:F2}s - {EndTime:F2}s";
}
