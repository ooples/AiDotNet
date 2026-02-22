using AiDotNet.Diffusion;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Audio.Features;

/// <summary>
/// Extracts Mel-Frequency Cepstral Coefficients (MFCCs) from audio signals.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// MFCCs are a compact representation of the spectral envelope of an audio signal.
/// They are widely used in speech recognition, speaker identification, and music analysis.
/// </para>
/// <para><b>For Beginners:</b> MFCCs capture the "shape" of the audio's frequency content,
/// similar to how humans perceive sound. The process:
/// <list type="number">
/// <item>Compute the Mel spectrogram (power spectrum on perceptual scale)</item>
/// <item>Take the log (matches human loudness perception)</item>
/// <item>Apply DCT (decorrelates and compresses the information)</item>
/// <item>Keep only the first N coefficients (typically 13-40)</item>
/// </list>
///
/// Why MFCCs work well for speech:
/// - They capture formant frequencies (vocal tract resonances)
/// - They're robust to background noise
/// - They compress audio information efficiently
///
/// Usage:
/// <code>
/// var mfcc = new MfccExtractor&lt;float&gt;(new MfccOptions { NumCoefficients = 13 });
/// var features = mfcc.Extract(audioTensor);
/// // features.Shape = [numFrames, 13]
/// </code>
/// </para>
/// </remarks>
public class MfccExtractor<T> : AudioFeatureExtractorBase<T>
{
    private readonly MelSpectrogram<T> _melSpectrogram;
    private readonly int _numCoefficients;
    private readonly bool _includeEnergy;
    private readonly bool _appendDelta;
    private readonly bool _appendDeltaDelta;
    private readonly double[,] _dctMatrix;

    /// <inheritdoc/>
    public override string Name => "MFCC";

    /// <inheritdoc/>
    public override int FeatureDimension
    {
        get
        {
            int dim = _numCoefficients;
            if (_appendDelta) dim *= 2;
            if (_appendDeltaDelta) dim = dim / 2 * 3;
            return dim;
        }
    }

    /// <summary>
    /// Initializes a new MFCC extractor.
    /// </summary>
    /// <param name="options">MFCC extraction options.</param>
    public MfccExtractor(MfccOptions? options = null)
        : base(options)
    {
        var mfccOptions = options ?? new MfccOptions();

        _numCoefficients = mfccOptions.NumCoefficients;
        _includeEnergy = mfccOptions.IncludeEnergy;
        _appendDelta = mfccOptions.AppendDelta;
        _appendDeltaDelta = mfccOptions.AppendDeltaDelta;

        // Create mel spectrogram with log compression
        _melSpectrogram = new MelSpectrogram<T>(
            sampleRate: mfccOptions.SampleRate,
            nMels: mfccOptions.NumMels,
            nFft: mfccOptions.FftSize,
            hopLength: mfccOptions.HopLength,
            fMin: mfccOptions.FMin,
            fMax: mfccOptions.FMax,
            logMel: true);

        // Precompute DCT-II matrix
        _dctMatrix = CreateDctMatrix(mfccOptions.NumMels, _numCoefficients);
    }

    /// <inheritdoc/>
    public override Tensor<T> Extract(Tensor<T> audio)
    {
        // Compute log mel spectrogram
        var melSpec = _melSpectrogram.Forward(audio);

        int numFrames = melSpec.Shape[0];
        int numMels = melSpec.Shape[1];

        // Apply DCT to get MFCCs
        var mfccs = new double[numFrames, _numCoefficients];

        for (int frame = 0; frame < numFrames; frame++)
        {
            for (int c = 0; c < _numCoefficients; c++)
            {
                double sum = 0;
                for (int m = 0; m < numMels; m++)
                {
                    sum += _dctMatrix[c, m] * NumOps.ToDouble(melSpec[frame, m]);
                }
                mfccs[frame, c] = sum;
            }
        }

        // Optionally replace first coefficient with energy
        if (_includeEnergy)
        {
            for (int frame = 0; frame < numFrames; frame++)
            {
                double energy = 0;
                for (int m = 0; m < numMels; m++)
                {
                    double val = NumOps.ToDouble(melSpec[frame, m]);
                    energy += Math.Exp(val); // Undo log
                }
                mfccs[frame, 0] = Math.Log(Math.Max(energy, 1e-10));
            }
        }

        // Compute delta and delta-delta if requested
        if (!_appendDelta && !_appendDeltaDelta)
        {
            return CreateTensorFromArray(mfccs, numFrames, _numCoefficients);
        }

        double[,] delta = new double[numFrames, _numCoefficients];
        double[,] deltaDelta = new double[numFrames, _numCoefficients];

        if (_appendDelta || _appendDeltaDelta)
        {
            ComputeDeltas(mfccs, delta, numFrames, _numCoefficients);
        }

        if (_appendDeltaDelta)
        {
            ComputeDeltas(delta, deltaDelta, numFrames, _numCoefficients);
        }

        // Combine features
        int totalFeatures = _numCoefficients;
        if (_appendDelta) totalFeatures += _numCoefficients;
        if (_appendDeltaDelta) totalFeatures += _numCoefficients;

        var combined = new double[numFrames, totalFeatures];

        for (int frame = 0; frame < numFrames; frame++)
        {
            int offset = 0;

            // Copy MFCCs
            for (int c = 0; c < _numCoefficients; c++)
            {
                combined[frame, offset++] = mfccs[frame, c];
            }

            // Copy deltas
            if (_appendDelta)
            {
                for (int c = 0; c < _numCoefficients; c++)
                {
                    combined[frame, offset++] = delta[frame, c];
                }
            }

            // Copy delta-deltas
            if (_appendDeltaDelta)
            {
                for (int c = 0; c < _numCoefficients; c++)
                {
                    combined[frame, offset++] = deltaDelta[frame, c];
                }
            }
        }

        return CreateTensorFromArray(combined, numFrames, totalFeatures);
    }

    private static double[,] CreateDctMatrix(int numMels, int numCoefficients)
    {
        var dct = new double[numCoefficients, numMels];

        for (int c = 0; c < numCoefficients; c++)
        {
            for (int m = 0; m < numMels; m++)
            {
                dct[c, m] = Math.Cos(Math.PI * c * (2 * m + 1) / (2 * numMels));
            }
        }

        // Apply orthonormal scaling
        double scale0 = Math.Sqrt(1.0 / numMels);
        double scaleN = Math.Sqrt(2.0 / numMels);

        for (int m = 0; m < numMels; m++)
        {
            dct[0, m] *= scale0;
        }

        for (int c = 1; c < numCoefficients; c++)
        {
            for (int m = 0; m < numMels; m++)
            {
                dct[c, m] *= scaleN;
            }
        }

        return dct;
    }

    private static void ComputeDeltas(double[,] features, double[,] deltas, int numFrames, int numFeatures)
    {
        const int windowSize = 2;
        double denominator = 2 * (1 + 4); // sum of i^2 for i = 1,2

        for (int frame = 0; frame < numFrames; frame++)
        {
            for (int f = 0; f < numFeatures; f++)
            {
                double numerator = 0;

                for (int i = 1; i <= windowSize; i++)
                {
                    int prevFrame = Math.Max(0, frame - i);
                    int nextFrame = Math.Min(numFrames - 1, frame + i);

                    numerator += i * (features[nextFrame, f] - features[prevFrame, f]);
                }

                deltas[frame, f] = numerator / denominator;
            }
        }
    }

    private Tensor<T> CreateTensorFromArray(double[,] array, int rows, int cols)
    {
        var tensor = new Tensor<T>([rows, cols]);

        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                tensor[r, c] = NumOps.FromDouble(array[r, c]);
            }
        }

        return tensor;
    }
}
