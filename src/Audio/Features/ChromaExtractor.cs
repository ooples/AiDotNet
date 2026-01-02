using AiDotNet.Diffusion.Audio;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Audio.Features;

/// <summary>
/// Extracts chromagram (pitch class profile) features from audio signals.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// A chromagram represents the energy of the 12 pitch classes (C, C#, D, D#, E, F, F#, G, G#, A, A#, B)
/// regardless of octave. It's particularly useful for music analysis tasks.
/// </para>
/// <para><b>For Beginners:</b> In Western music, there are 12 notes that repeat in each octave.
/// A C note at 262 Hz and a C note at 524 Hz are both "C" - they're the same pitch class.
///
/// A chromagram collapses all octaves together, showing how much energy is in each of the 12 notes:
/// - Index 0: C (do)
/// - Index 1: C#/Db
/// - Index 2: D (re)
/// - ...and so on through B
///
/// This is useful for:
/// - Chord recognition (chords have characteristic chroma patterns)
/// - Key detection (which notes are emphasized in the music)
/// - Music similarity (songs in the same key have similar chromagrams)
/// - Cover song detection
///
/// Usage:
/// <code>
/// var chroma = new ChromaExtractor&lt;float&gt;();
/// var features = chroma.Extract(audioTensor);
/// // features.Shape = [numFrames, 12]
/// </code>
/// </para>
/// </remarks>
public class ChromaExtractor<T> : AudioFeatureExtractorBase<T>
{
    private readonly ShortTimeFourierTransform<T> _stft;
    private readonly double[,] _chromaFilterbank;
    private readonly bool _normalize;
    private readonly double _tuningFrequency;

    /// <inheritdoc/>
    public override string Name => "Chroma";

    /// <inheritdoc/>
    public override int FeatureDimension => 12;

    /// <summary>
    /// Initializes a new chroma feature extractor.
    /// </summary>
    /// <param name="options">Chroma extraction options.</param>
    public ChromaExtractor(ChromaOptions? options = null)
        : base(options)
    {
        var chromaOptions = options ?? new ChromaOptions();

        _normalize = chromaOptions.Normalize;
        _tuningFrequency = chromaOptions.TuningFrequency;

        // Create STFT processor
        _stft = new ShortTimeFourierTransform<T>(
            nFft: chromaOptions.FftSize,
            hopLength: chromaOptions.HopLength);

        // Create chroma filterbank
        _chromaFilterbank = CreateChromaFilterbank(
            chromaOptions.FftSize,
            chromaOptions.SampleRate,
            chromaOptions.TuningFrequency,
            chromaOptions.NumOctaves);
    }

    /// <inheritdoc/>
    public override Tensor<T> Extract(Tensor<T> audio)
    {
        // Compute STFT magnitude
        var stftResult = _stft.Forward(audio);

        // STFT result is Tensor<Complex<T>> with shape [numFrames, numFreqs]
        int numFrames = stftResult.Shape[0];
        int numFreqs = stftResult.Shape[1];

        // Compute power spectrum
        var powerSpectrum = new double[numFrames, numFreqs];
        for (int frame = 0; frame < numFrames; frame++)
        {
            for (int freq = 0; freq < numFreqs; freq++)
            {
                var complex = stftResult[frame, freq];
                double real = NumOps.ToDouble(complex.Real);
                double imag = NumOps.ToDouble(complex.Imaginary);
                powerSpectrum[frame, freq] = real * real + imag * imag;
            }
        }

        // Apply chroma filterbank
        var chroma = new double[numFrames, 12];

        for (int frame = 0; frame < numFrames; frame++)
        {
            for (int pitchClass = 0; pitchClass < 12; pitchClass++)
            {
                double sum = 0;
                for (int freq = 0; freq < numFreqs && freq < _chromaFilterbank.GetLength(1); freq++)
                {
                    sum += _chromaFilterbank[pitchClass, freq] * powerSpectrum[frame, freq];
                }
                chroma[frame, pitchClass] = sum;
            }

            // Normalize if requested
            if (_normalize)
            {
                double norm = 0;
                for (int p = 0; p < 12; p++)
                {
                    norm += chroma[frame, p] * chroma[frame, p];
                }
                norm = Math.Sqrt(norm);

                if (norm > 1e-10)
                {
                    for (int p = 0; p < 12; p++)
                    {
                        chroma[frame, p] /= norm;
                    }
                }
            }
        }

        // Convert to tensor
        var result = new Tensor<T>([numFrames, 12]);
        for (int frame = 0; frame < numFrames; frame++)
        {
            for (int p = 0; p < 12; p++)
            {
                result[frame, p] = NumOps.FromDouble(chroma[frame, p]);
            }
        }

        return result;
    }

    private static double[,] CreateChromaFilterbank(
        int fftSize,
        int sampleRate,
        double tuningFrequency,
        int numOctaves)
    {
        int numFreqs = fftSize / 2 + 1;
        var filterbank = new double[12, numFreqs];

        // Reference: A4 = tuningFrequency (typically 440 Hz)
        // Each pitch class spans 1/12 of an octave

        for (int freq = 1; freq < numFreqs; freq++)
        {
            double hz = (double)freq * sampleRate / fftSize;

            // Skip frequencies below the lowest C we care about
            double minFreq = tuningFrequency * Math.Pow(2, -4.75 - numOctaves / 2.0); // C below A4
            if (hz < minFreq) continue;

            // Skip frequencies above the highest C we care about
            double maxFreq = tuningFrequency * Math.Pow(2, -4.75 + numOctaves / 2.0);
            if (hz > maxFreq) continue;

            // Convert Hz to pitch class
            // A4 = 440 Hz = MIDI note 69 = pitch class 9 (A)
            // semitone = 12 * log2(f / A4) + 69
            // pitch class = semitone mod 12

            double semitone = 12 * MathHelper.Log2(hz / tuningFrequency) + 69;
            int pitchClass = ((int)Math.Round(semitone) % 12 + 12) % 12;

            // Calculate weight based on how close frequency is to the pitch class center
            double closestSemitone = Math.Round(semitone);
            double distance = Math.Abs(semitone - closestSemitone);

            // Use a cosine window for smooth interpolation
            double weight = Math.Cos(Math.PI * distance);
            weight = weight * weight; // Square for sharper peaks

            filterbank[pitchClass, freq] = weight;
        }

        // Normalize each pitch class filter
        for (int p = 0; p < 12; p++)
        {
            double sum = 0;
            for (int freq = 0; freq < numFreqs; freq++)
            {
                sum += filterbank[p, freq];
            }

            if (sum > 0)
            {
                for (int freq = 0; freq < numFreqs; freq++)
                {
                    filterbank[p, freq] /= sum;
                }
            }
        }

        return filterbank;
    }

    /// <summary>
    /// Gets the pitch class name for an index (0-11).
    /// </summary>
    /// <param name="index">The pitch class index.</param>
    /// <returns>The pitch class name (C, C#, D, etc.).</returns>
    public static string GetPitchClassName(int index)
    {
        return (index % 12) switch
        {
            0 => "C",
            1 => "C#",
            2 => "D",
            3 => "D#",
            4 => "E",
            5 => "F",
            6 => "F#",
            7 => "G",
            8 => "G#",
            9 => "A",
            10 => "A#",
            11 => "B",
            _ => "?"
        };
    }

    /// <summary>
    /// Gets the dominant pitch class for a chroma vector.
    /// </summary>
    /// <param name="chromaFrame">A chroma vector of length 12.</param>
    /// <returns>The index of the dominant pitch class (0-11).</returns>
    public int GetDominantPitchClass(T[] chromaFrame)
    {
        if (chromaFrame.Length != 12)
            throw new ArgumentException("Chroma frame must have exactly 12 elements.", nameof(chromaFrame));

        int maxIndex = 0;
        double maxValue = NumOps.ToDouble(chromaFrame[0]);

        for (int i = 1; i < 12; i++)
        {
            double value = NumOps.ToDouble(chromaFrame[i]);
            if (value > maxValue)
            {
                maxValue = value;
                maxIndex = i;
            }
        }

        return maxIndex;
    }
}
