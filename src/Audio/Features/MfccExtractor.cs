using AiDotNet.Attributes;
using AiDotNet.Diffusion.Audio;
using AiDotNet.Enums;
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
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.SignalProcessing)]
[ModelTask(ModelTask.FeatureExtraction)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Comparison of Parametric Representations for Monosyllabic Word Recognition in Continuously Spoken Sentences", "https://doi.org/10.1109/TASSP.1980.1163420", Year = 1980, Authors = "Steven B. Davis, Paul Mermelstein")]
public class MfccExtractor<T> : AudioFeatureExtractorBase<T>
{
    private readonly MelSpectrogram<T> _melSpectrogram;
    private readonly int _numCoefficients;
    private readonly bool _includeEnergy;
    private readonly bool _appendDelta;
    private readonly bool _appendDeltaDelta;
    private readonly double[,] _dctMatrix;
    private Tensor<T>? _dctTensor;

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
        if (mfccOptions.NumCoefficients < 0)
            throw new ArgumentOutOfRangeException(nameof(options), "MfccOptions.NumCoefficients must be non-negative.");
        if (mfccOptions.IncludeEnergy && mfccOptions.NumCoefficients == 0)
            throw new ArgumentException(
                "MfccOptions.IncludeEnergy replaces the first coefficient and requires NumCoefficients >= 1.", nameof(options));

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
        // Features are computed from fresh values as before (not part of the autodiff graph).
        using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();

        // Compute log mel spectrogram
        var melSpec = _melSpectrogram.Forward(audio);

        int numFrames = melSpec.Shape[0];
        int numMels = melSpec.Shape[1];

        // Apply DCT to get MFCCs: mfcc[frame, c] = sum_m dct[c, m] * mel[frame, m]
        var mfccs = numFrames == 0 || numMels == 0 || _numCoefficients == 0
            ? new Tensor<T>([numFrames, _numCoefficients])
            : Engine.TensorMatMul(
                melSpec,
                Engine.TensorTranspose(Engine.TensorSlice(DctTensor(), [0, 0], [_numCoefficients, numMels])));

        // Optionally replace first coefficient with energy: log(max(sum_m exp(mel), 1e-10))
        if (_includeEnergy && numFrames > 0)
        {
            var energy = numMels > 0
                ? Engine.ReduceSum(Engine.TensorExp(melSpec), [1], keepDims: true) // Undo log
                : new Tensor<T>([numFrames, 1]);
            T floor = NumOps.FromDouble(1e-10);
            var logEnergy = Engine.TensorLog(Engine.TensorWhere(
                Engine.TensorLessThan(energy, floor), Tensor<T>.CreateDefault([numFrames, 1], floor), energy));
            mfccs = _numCoefficients == 1
                ? logEnergy
                : Engine.Concat(new[] { logEnergy, Engine.TensorSlice(mfccs, [0, 1], [numFrames, _numCoefficients - 1]) }, 1);
        }

        // Compute delta and delta-delta if requested
        if (!_appendDelta && !_appendDeltaDelta)
        {
            return mfccs;
        }

        var delta = ComputeDeltas(mfccs);

        // Combine features: [mfcc | delta | delta-delta]
        var parts = new List<Tensor<T>> { mfccs };
        if (_appendDelta) parts.Add(delta);
        if (_appendDeltaDelta) parts.Add(ComputeDeltas(delta));

        return parts.Count == 1 ? mfccs : Engine.Concat(parts, 1);
    }

    /// <summary>
    /// The DCT matrix (a fixed table built once in the constructor) converted to
    /// <typeparamref name="T"/> once, in bulk.
    /// </summary>
    private Tensor<T> DctTensor()
    {
        if (_dctTensor is null)
        {
            int rows = _dctMatrix.GetLength(0), cols = _dctMatrix.GetLength(1);
            var values = new double[rows * cols];
            Buffer.BlockCopy(_dctMatrix, 0, values, 0, values.Length * sizeof(double));
            _dctTensor = new Tensor<double>(values, [rows, cols]).Cast<T>();
        }
        return _dctTensor;
    }

    private static double[,] CreateDctMatrix(int numMels, int numCoefficients)
    {
        var dct = new double[numCoefficients, numMels];
        if (numCoefficients == 0 || numMels == 0)
            return dct; // nothing to fill; the coefficient-0 scaling below needs a row

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

    private Tensor<T> ComputeDeltas(Tensor<T> features)
    {
        // Regression deltas over +-2 frames with edge frames repeated:
        // delta[t] = (1 * (x[t+1] - x[t-1]) + 2 * (x[t+2] - x[t-2])) / 10
        const int windowSize = 2;
        double denominator = 2 * (1 + 4); // sum of i^2 for i = 1,2

        int numFrames = features.Shape[0];
        int numFeatures = features.Shape[1];
        if (numFrames == 0 || numFeatures == 0)
            return new Tensor<T>([numFrames, numFeatures]);

        var edge = new[] { windowSize, numFeatures };
        var padded = Engine.Concat(new[]
        {
            Engine.TensorBroadcastTo(Engine.TensorSlice(features, [0, 0], [1, numFeatures]), edge),
            features,
            Engine.TensorBroadcastTo(Engine.TensorSlice(features, [numFrames - 1, 0], [1, numFeatures]), edge)
        }, 0);

        var tapWeights = new[] { NumOps.One, NumOps.FromDouble(2) };
        Tensor<T>? numerator = null;
        for (int i = 1; i <= windowSize; i++)
        {
            var next = Engine.TensorSlice(padded, [windowSize + i, 0], [numFrames, numFeatures]);
            var prev = Engine.TensorSlice(padded, [windowSize - i, 0], [numFrames, numFeatures]);
            var difference = Engine.TensorSubtract(next, prev);
            var term = i == 1 ? difference : Engine.TensorMultiplyScalar(difference, tapWeights[i - 1]);
            numerator = numerator is null ? term : Engine.TensorAdd(numerator, term);
        }

        return Engine.TensorDivideScalar(numerator ?? new Tensor<T>([numFrames, numFeatures]), NumOps.FromDouble(denominator));
    }
}
