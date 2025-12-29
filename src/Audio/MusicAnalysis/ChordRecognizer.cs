using AiDotNet.Audio.Features;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Audio.MusicAnalysis;

/// <summary>
/// Recognizes chords from audio using chromagram analysis.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Chord recognition works by comparing chromagram patterns against known chord templates.
/// Supports major, minor, diminished, augmented, dominant 7th, major 7th, and minor 7th chords.
/// </para>
/// <para><b>For Beginners:</b> A chord is a group of notes played together.
/// Different chords have different "shapes" in their chroma patterns:
/// - C major (C-E-G) has energy in bins 0, 4, and 7
/// - C minor (C-Eb-G) has energy in bins 0, 3, and 7
///
/// This algorithm looks at the audio and matches it against known chord patterns
/// to tell you what chord is playing at each moment.
///
/// Usage:
/// <code>
/// var recognizer = new ChordRecognizer&lt;float&gt;();
/// var chords = recognizer.Recognize(audioTensor);
/// foreach (var chord in chords)
///     Console.WriteLine($"{chord.StartTime:F2}s: {chord.Name}");
/// </code>
/// </para>
/// </remarks>
public class ChordRecognizer<T> : MusicAnalysisBase<T>
{
    private readonly ChromaExtractor<T> _chromaExtractor;
    private readonly ChordRecognizerOptions _options;
    private readonly Dictionary<string, double[]> _chordTemplates;

    // Chord quality names
    private static readonly string[] NoteNames = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];

    /// <summary>
    /// Creates a new chord recognizer.
    /// </summary>
    /// <param name="options">Chord recognition options.</param>
    public ChordRecognizer(ChordRecognizerOptions? options = null)
    {
        _options = options ?? new ChordRecognizerOptions();

        // Set base class properties
        SampleRate = _options.SampleRate;
        HopLength = _options.HopLength;
        FftSize = _options.FftSize;

        _chromaExtractor = new ChromaExtractor<T>(new ChromaOptions
        {
            SampleRate = _options.SampleRate,
            FftSize = _options.FftSize,
            HopLength = _options.HopLength
        });

        _chordTemplates = BuildChordTemplates();
    }

    /// <summary>
    /// Recognizes chords in the audio.
    /// </summary>
    /// <param name="audio">Audio samples as a tensor.</param>
    /// <returns>List of recognized chord segments.</returns>
    public List<ChordSegment> Recognize(Tensor<T> audio)
    {
        // Extract chroma features
        var chroma = _chromaExtractor.Extract(audio);

        // Classify each frame
        var frameChords = ClassifyFrames(chroma);

        // Merge consecutive same chords into segments
        var segments = MergeIntoSegments(frameChords);

        return segments;
    }

    /// <summary>
    /// Recognizes chords in the audio.
    /// </summary>
    /// <param name="audio">Audio samples as a vector.</param>
    /// <returns>List of recognized chord segments.</returns>
    public List<ChordSegment> Recognize(Vector<T> audio)
    {
        var tensor = new Tensor<T>([audio.Length]);
        for (int i = 0; i < audio.Length; i++)
        {
            tensor[i] = audio[i];
        }
        return Recognize(tensor);
    }

    private Dictionary<string, double[]> BuildChordTemplates()
    {
        var templates = new Dictionary<string, double[]>();

        // Build templates for each root note and chord quality
        for (int root = 0; root < 12; root++)
        {
            string rootName = NoteNames[root];

            // Major chord (root, major 3rd, perfect 5th)
            templates[$"{rootName}"] = CreateTemplate(root, [0, 4, 7]);

            // Minor chord (root, minor 3rd, perfect 5th)
            templates[$"{rootName}m"] = CreateTemplate(root, [0, 3, 7]);

            // Diminished chord (root, minor 3rd, diminished 5th)
            templates[$"{rootName}dim"] = CreateTemplate(root, [0, 3, 6]);

            // Augmented chord (root, major 3rd, augmented 5th)
            templates[$"{rootName}aug"] = CreateTemplate(root, [0, 4, 8]);

            // Dominant 7th (root, major 3rd, perfect 5th, minor 7th)
            templates[$"{rootName}7"] = CreateTemplate(root, [0, 4, 7, 10]);

            // Major 7th (root, major 3rd, perfect 5th, major 7th)
            templates[$"{rootName}maj7"] = CreateTemplate(root, [0, 4, 7, 11]);

            // Minor 7th (root, minor 3rd, perfect 5th, minor 7th)
            templates[$"{rootName}m7"] = CreateTemplate(root, [0, 3, 7, 10]);

            // Suspended 4th
            templates[$"{rootName}sus4"] = CreateTemplate(root, [0, 5, 7]);

            // Suspended 2nd
            templates[$"{rootName}sus2"] = CreateTemplate(root, [0, 2, 7]);
        }

        // Add "no chord" template
        templates["N"] = new double[12]; // All zeros

        return templates;
    }

    private static double[] CreateTemplate(int root, int[] intervals)
    {
        var template = new double[12];
        double weight = 1.0 / intervals.Length;

        foreach (int interval in intervals)
        {
            int bin = (root + interval) % 12;
            template[bin] = weight;
        }

        // Normalize
        double sum = template.Sum();
        if (sum > 0)
        {
            for (int i = 0; i < 12; i++)
            {
                template[i] /= sum;
            }
        }

        return template;
    }

    private List<(string chord, double confidence)> ClassifyFrames(Tensor<T> chroma)
    {
        int numFrames = chroma.Shape[0];
        var frameChords = new List<(string chord, double confidence)>();

        for (int f = 0; f < numFrames; f++)
        {
            // Extract chroma vector for this frame
            var chromaVector = new double[12];
            double maxChroma = 0;

            for (int c = 0; c < 12; c++)
            {
                chromaVector[c] = NumOps.ToDouble(chroma[f, c]);
                maxChroma = Math.Max(maxChroma, chromaVector[c]);
            }

            // Check if frame has enough energy
            if (maxChroma < _options.MinChromaEnergy)
            {
                frameChords.Add(("N", 1.0)); // No chord
                continue;
            }

            // Normalize
            double sum = chromaVector.Sum();
            if (sum > 0)
            {
                for (int c = 0; c < 12; c++)
                {
                    chromaVector[c] /= sum;
                }
            }

            // Find best matching template
            string bestChord = "N";
            double bestScore = _options.MinConfidence;

            foreach (var (chordName, template) in _chordTemplates)
            {
                double score = CosineSimilarity(chromaVector, template);

                if (score > bestScore)
                {
                    bestScore = score;
                    bestChord = chordName;
                }
            }

            frameChords.Add((bestChord, bestScore));
        }

        return frameChords;
    }

    private static double CosineSimilarity(double[] a, double[] b)
    {
        double dotProduct = 0;
        double normA = 0;
        double normB = 0;

        for (int i = 0; i < a.Length; i++)
        {
            dotProduct += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }

        double denominator = Math.Sqrt(normA) * Math.Sqrt(normB);
        if (denominator < 1e-10) return 0;

        return dotProduct / denominator;
    }

    private List<ChordSegment> MergeIntoSegments(List<(string chord, double confidence)> frameChords)
    {
        var segments = new List<ChordSegment>();
        if (frameChords.Count == 0) return segments;

        double frameRate = _options.SampleRate / (double)_options.HopLength;

        string currentChord = frameChords[0].chord;
        int startFrame = 0;
        double totalConfidence = frameChords[0].confidence;
        int count = 1;

        for (int i = 1; i < frameChords.Count; i++)
        {
            var (chord, confidence) = frameChords[i];

            if (chord == currentChord)
            {
                totalConfidence += confidence;
                count++;
            }
            else
            {
                // Save current segment
                double duration = (i - startFrame) / frameRate;
                if (duration >= _options.MinSegmentDuration && currentChord != "N")
                {
                    segments.Add(new ChordSegment
                    {
                        Name = currentChord,
                        StartTime = startFrame / frameRate,
                        EndTime = i / frameRate,
                        Confidence = totalConfidence / count
                    });
                }

                // Start new segment
                currentChord = chord;
                startFrame = i;
                totalConfidence = confidence;
                count = 1;
            }
        }

        // Save last segment
        if (currentChord != "N")
        {
            double duration = (frameChords.Count - startFrame) / frameRate;
            if (duration >= _options.MinSegmentDuration)
            {
                segments.Add(new ChordSegment
                {
                    Name = currentChord,
                    StartTime = startFrame / frameRate,
                    EndTime = frameChords.Count / frameRate,
                    Confidence = totalConfidence / count
                });
            }
        }

        return segments;
    }
}

/// <summary>
/// Represents a chord segment in audio.
/// </summary>
public class ChordSegment
{
    /// <summary>
    /// Gets or sets the chord name (e.g., "C", "Am", "G7").
    /// </summary>
    public string Name { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the start time in seconds.
    /// </summary>
    public double StartTime { get; set; }

    /// <summary>
    /// Gets or sets the end time in seconds.
    /// </summary>
    public double EndTime { get; set; }

    /// <summary>
    /// Gets the duration in seconds.
    /// </summary>
    public double Duration => EndTime - StartTime;

    /// <summary>
    /// Gets or sets the confidence score (0-1).
    /// </summary>
    public double Confidence { get; set; }
}

/// <summary>
/// Configuration options for chord recognition.
/// </summary>
public class ChordRecognizerOptions
{
    /// <summary>
    /// Gets or sets the sample rate.
    /// </summary>
    public int SampleRate { get; set; } = 22050;

    /// <summary>
    /// Gets or sets the FFT size.
    /// </summary>
    public int FftSize { get; set; } = 4096;

    /// <summary>
    /// Gets or sets the hop length.
    /// </summary>
    public int HopLength { get; set; } = 2048;

    /// <summary>
    /// Gets or sets the minimum chroma energy to consider.
    /// </summary>
    public double MinChromaEnergy { get; set; } = 0.01;

    /// <summary>
    /// Gets or sets the minimum confidence for chord detection.
    /// </summary>
    public double MinConfidence { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets the minimum segment duration in seconds.
    /// </summary>
    public double MinSegmentDuration { get; set; } = 0.2;
}
