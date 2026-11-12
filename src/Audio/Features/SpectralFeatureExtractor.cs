using AiDotNet.Attributes;
using AiDotNet.Diffusion.Audio;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Audio.Features;

/// <summary>
/// Extracts spectral features from audio signals including centroid, bandwidth, rolloff, and flux.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Spectral features describe the shape and characteristics of an audio signal's
/// frequency content. They are widely used for audio classification, music analysis,
/// and speech processing.
/// </para>
/// <para><b>For Beginners:</b> These features describe "what the sound looks like" in terms of frequency:
/// <list type="bullet">
/// <item><b>Spectral Centroid</b>: The "center of mass" of the spectrum - high for bright sounds (cymbals),
///   low for dull sounds (bass drum). Think of it as the "brightness" of the sound.</item>
/// <item><b>Spectral Bandwidth</b>: How spread out the frequencies are. Wide for rich sounds (orchestra),
///   narrow for pure tones (flute).</item>
/// <item><b>Spectral Rolloff</b>: The frequency below which most (e.g., 85%) of the energy is concentrated.
///   Useful for distinguishing voiced from unvoiced speech.</item>
/// <item><b>Spectral Flux</b>: How much the spectrum changes between frames. High during transients (drum hits),
///   low during sustained sounds.</item>
/// <item><b>Spectral Flatness</b>: How "noisy" vs "tonal" the sound is. 1.0 = pure noise, 0.0 = pure tone.</item>
/// </list>
///
/// Usage:
/// <code>
/// var extractor = new SpectralFeatureExtractor&lt;float&gt;();
/// var features = extractor.Extract(audioTensor);
/// // features.Shape = [numFrames, numFeatures]
/// </code>
/// </para>
/// </remarks>
[ModelDomain(ModelDomain.Audio)]
[ModelCategory(ModelCategory.SignalProcessing)]
[ModelTask(ModelTask.FeatureExtraction)]
[ModelComplexity(ModelComplexity.Low)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("A Tutorial on Onset Detection in Music Signals", "https://doi.org/10.1109/TSP.2005.851331", Year = 2005, Authors = "Juan Pablo Bello, Laurent Daudet, Samer Abdallah, Chris Duxbury, Mike Davies, Mark B. Sandler")]
public class SpectralFeatureExtractor<T> : AudioFeatureExtractorBase<T>
{
    private readonly ShortTimeFourierTransform<T> _stft;
    private readonly SpectralFeatureType _featureTypes;
    private readonly double _rolloffPercentage;
    private readonly int _featureDimension;

    /// <inheritdoc/>
    public override string Name => "SpectralFeatures";

    /// <inheritdoc/>
    public override int FeatureDimension => _featureDimension;

    /// <summary>
    /// Initializes a new spectral feature extractor.
    /// </summary>
    /// <param name="options">Spectral feature extraction options.</param>
    public SpectralFeatureExtractor(SpectralFeatureOptions? options = null)
        : base(options)
    {
        var spectralOptions = options ?? new SpectralFeatureOptions();

        _featureTypes = spectralOptions.FeatureTypes;
        _rolloffPercentage = spectralOptions.RolloffPercentage;

        // Create STFT processor
        _stft = new ShortTimeFourierTransform<T>(
            nFft: spectralOptions.FftSize,
            hopLength: spectralOptions.HopLength);

        // Count enabled features
        _featureDimension = CountEnabledFeatures(_featureTypes);
    }

    /// <inheritdoc/>
    public override Tensor<T> Extract(Tensor<T> audio)
    {
        // Features are computed from fresh values as before (not part of the autodiff graph).
        using var noGrad = new AiDotNet.Tensors.Engines.Autodiff.NoGradScope<T>();

        // Compute STFT - returns Tensor<Complex<T>> with shape [numFrames, numFreqs]
        var stftResult = _stft.Forward(audio);

        int numFrames = stftResult.Shape[0];
        int numFreqs = stftResult.Shape[1];

        // Compute magnitude spectrum
        var real = Engine.TensorReal(stftResult);
        var imag = Engine.TensorImag(stftResult);
        var magnitude = Engine.TensorSqrt(Engine.TensorAdd(Engine.TensorSquare(real), Engine.TensorSquare(imag)));

        // Calculate frequency bins in Hz
        // Guard against division by zero when numFreqs == 1 (only DC component)
        double freqScale = numFreqs > 1 ? (double)SampleRate / (2 * (numFreqs - 1)) : 0.0;
        var freqBins = Engine.TensorMultiplyScalar(Engine.TensorArange<T>(0, numFreqs, 1), NumOps.FromDouble(freqScale));

        // Extract features (each a [numFrames] column)
        var features = new List<Tensor<T>>();

        if (numFrames > 0)
        {
            if (_featureTypes.HasFlag(SpectralFeatureType.Centroid))
            {
                features.Add(ComputeSpectralCentroid(magnitude, freqBins, numFrames, numFreqs));
            }

            if (_featureTypes.HasFlag(SpectralFeatureType.Bandwidth))
            {
                var centroids = ComputeSpectralCentroid(magnitude, freqBins, numFrames, numFreqs);
                features.Add(ComputeSpectralBandwidth(magnitude, freqBins, centroids, numFrames, numFreqs));
            }

            if (_featureTypes.HasFlag(SpectralFeatureType.Rolloff))
            {
                features.Add(ComputeSpectralRolloff(magnitude, freqScale, numFrames, numFreqs));
            }

            if (_featureTypes.HasFlag(SpectralFeatureType.Flux))
            {
                features.Add(ComputeSpectralFlux(magnitude, numFrames, numFreqs));
            }

            if (_featureTypes.HasFlag(SpectralFeatureType.Flatness))
            {
                features.Add(ComputeSpectralFlatness(magnitude, numFrames, numFreqs));
            }

            if (_featureTypes.HasFlag(SpectralFeatureType.Contrast))
            {
                features.AddRange(ComputeSpectralContrast(magnitude, numFrames, numFreqs));
            }

            if (_featureTypes.HasFlag(SpectralFeatureType.ZeroCrossingRate))
            {
                features.Add(ComputeZeroCrossingRate(audio, numFrames));
            }
        }

        // Combine all features into result tensor [numFrames, featureDimension]
        if (features.Count == 0)
            return new Tensor<T>([numFrames, _featureDimension]);

        var columns = Engine.TensorStack(features.ToArray(), 1);
        if (features.Count >= _featureDimension)
            return features.Count == _featureDimension ? columns : Engine.TensorSlice(columns, [0, 0], [numFrames, _featureDimension]);
        return Engine.Concat(new[] { columns, new Tensor<T>([numFrames, _featureDimension - features.Count]) }, 1);
    }

    /// <summary>Per-frame value where <paramref name="condition"/> holds, else zero.</summary>
    private Tensor<T> WhereOrZero(Tensor<T> condition, Tensor<T> value)
        => Engine.TensorWhere(condition, value, new Tensor<T>(value.Shape.ToArray()));

    /// <summary>Frames as rows: <paramref name="column"/> <c>[numFrames]</c> broadcast to <c>[numFrames, numFreqs]</c>.</summary>
    private Tensor<T> AcrossFreqs(Tensor<T> column, int numFrames, int numFreqs)
        => Engine.TensorBroadcastTo(Engine.Reshape(column, [numFrames, 1]), [numFrames, numFreqs]);

    /// <summary>Bins as columns: <paramref name="row"/> <c>[numFreqs]</c> broadcast to <c>[numFrames, numFreqs]</c>.</summary>
    private Tensor<T> AcrossFrames(Tensor<T> row, int numFrames, int numFreqs)
        => Engine.TensorBroadcastTo(Engine.Reshape(row, [1, numFreqs]), [numFrames, numFreqs]);

    /// <summary>Sum over the bins of each frame, <c>[numFrames]</c>.</summary>
    private Tensor<T> SumOverFreqs(Tensor<T> x, int numFrames, int numFreqs)
        => numFreqs == 0 ? new Tensor<T>([numFrames]) : Engine.ReduceSum(x, [1], keepDims: false);

    private Tensor<T> ComputeSpectralCentroid(Tensor<T> magnitude, Tensor<T> freqBins, int numFrames, int numFreqs)
    {
        // sum_f freq * |X| / sum_f |X| where the denominator exceeds 1e-10, else 0
        var numerator = SumOverFreqs(Engine.TensorMultiply(magnitude, AcrossFrames(freqBins, numFrames, numFreqs)), numFrames, numFreqs);
        var denominator = SumOverFreqs(magnitude, numFrames, numFreqs);

        return WhereOrZero(
            Engine.TensorGreaterThan(denominator, NumOps.FromDouble(1e-10)),
            Engine.TensorDivide(numerator, denominator));
    }

    private Tensor<T> ComputeSpectralBandwidth(Tensor<T> magnitude, Tensor<T> freqBins, Tensor<T> centroids, int numFrames, int numFreqs)
    {
        // sqrt(sum_f |X| (freq - centroid)^2 / sum_f |X|) where the denominator exceeds 1e-10, else 0
        var diff = Engine.TensorSubtract(AcrossFrames(freqBins, numFrames, numFreqs), AcrossFreqs(centroids, numFrames, numFreqs));
        var numerator = SumOverFreqs(Engine.TensorMultiply(Engine.TensorMultiply(magnitude, diff), diff), numFrames, numFreqs);
        var denominator = SumOverFreqs(magnitude, numFrames, numFreqs);

        return WhereOrZero(
            Engine.TensorGreaterThan(denominator, NumOps.FromDouble(1e-10)),
            Engine.TensorSqrt(Engine.TensorDivide(numerator, denominator)));
    }

    private Tensor<T> ComputeSpectralRolloff(Tensor<T> magnitude, double freqScale, int numFrames, int numFreqs)
    {
        if (numFreqs == 0)
            return new Tensor<T>([numFrames]);

        // Frequency of the first bin whose cumulative magnitude reaches rolloff * total (0 if none)
        var totalEnergy = SumOverFreqs(magnitude, numFrames, numFreqs);
        var threshold = Engine.TensorMultiplyScalar(totalEnergy, NumOps.FromDouble(_rolloffPercentage));
        var cumEnergy = Engine.TensorCumSum(magnitude, 1);
        var reached = Engine.TensorGreaterEqual(cumEnergy, AcrossFreqs(threshold, numFrames, numFreqs));

        // First reaching bin: the smallest bin index among reaching bins (numFreqs when none)
        var binIndex = AcrossFrames(Engine.TensorArange<T>(0, numFreqs, 1), numFrames, numFreqs);
        var candidates = Engine.TensorWhere(
            reached, binIndex, Tensor<T>.CreateDefault([numFrames, numFreqs], NumOps.FromDouble(numFreqs)));
        var firstBin = Engine.TensorAmin(candidates, [1], keepDims: false);

        return WhereOrZero(
            Engine.TensorLessThan(firstBin, NumOps.FromDouble(numFreqs)),
            Engine.TensorMultiplyScalar(firstBin, NumOps.FromDouble(freqScale)));
    }

    private Tensor<T> ComputeSpectralFlux(Tensor<T> magnitude, int numFrames, int numFreqs)
    {
        // Frame 0 has no predecessor (flux 0); otherwise the L2 norm of the frame-to-frame change
        if (numFrames < 2)
            return new Tensor<T>([numFrames]);

        var diff = Engine.TensorSubtract(
            Engine.TensorSlice(magnitude, [1, 0], [numFrames - 1, numFreqs]),
            Engine.TensorSlice(magnitude, [0, 0], [numFrames - 1, numFreqs]));
        var flux = Engine.TensorSqrt(SumOverFreqs(Engine.TensorSquare(diff), numFrames - 1, numFreqs));

        return Engine.Concat(new[] { new Tensor<T>([1]), flux }, 0);
    }

    private Tensor<T> ComputeSpectralFlatness(Tensor<T> magnitude, int numFrames, int numFreqs)
    {
        // Geometric / arithmetic mean over the bins above 1e-10 (0 when none, or their sum is tiny)
        T epsilon = NumOps.FromDouble(1e-10);
        var zeros = new Tensor<T>([numFrames, numFreqs]);
        var above = Engine.TensorGreaterThan(magnitude, epsilon);
        var logSum = SumOverFreqs(Engine.TensorWhere(above, Engine.TensorLog(magnitude), zeros), numFrames, numFreqs);
        var sum = SumOverFreqs(Engine.TensorWhere(above, magnitude, zeros), numFrames, numFreqs);
        var count = SumOverFreqs(above, numFrames, numFreqs);

        var geometricMean = Engine.TensorExp(Engine.TensorDivide(logSum, count));
        var arithmeticMean = Engine.TensorDivide(sum, count);
        var valid = Engine.TensorMultiply(
            Engine.TensorGreaterThan(count, NumOps.Zero),
            Engine.TensorGreaterThan(sum, epsilon));

        return WhereOrZero(valid, Engine.TensorDivide(geometricMean, arithmeticMean));
    }

    private Tensor<T>[] ComputeSpectralContrast(Tensor<T> magnitude, int numFrames, int numFreqs)
    {
        const int numBands = 6;
        var contrast = new Tensor<T>[numBands];

        int freqsPerBand = numFreqs / numBands;

        // Per-band peak/valley size (a quarter of the band, at least one bin), converted to T in bulk
        var peakCounts = new int[numBands];
        for (int band = 0; band < numBands; band++)
        {
            int bandCount = Math.Min((band + 1) * freqsPerBand, numFreqs) - band * freqsPerBand;
            peakCounts[band] = Math.Max(1, bandCount / 4);
        }
        T[] divisors = new Tensor<int>(peakCounts, [numBands]).Cast<T>().ToArray();

        for (int band = 0; band < numBands; band++)
        {
            int startFreq = band * freqsPerBand;
            int endFreq = Math.Min((band + 1) * freqsPerBand, numFreqs);
            int count = endFreq - startFreq;

            if (count <= 0)
            {
                contrast[band] = new Tensor<T>([numFrames]);
                continue;
            }

            // Sorted band magnitudes; contrast = mean of the top 25% - mean of the bottom 25%
            var (sorted, _) = Engine.TensorSort(
                Engine.TensorSlice(magnitude, [0, startFreq], [numFrames, count]), axis: 1);
            int peakCount = peakCounts[band];
            T divisor = divisors[band];

            var peakMean = Engine.TensorDivideScalar(
                Engine.ReduceSum(Engine.TensorSlice(sorted, [0, count - peakCount], [numFrames, peakCount]), [1], keepDims: false), divisor);
            var valleyMean = Engine.TensorDivideScalar(
                Engine.ReduceSum(Engine.TensorSlice(sorted, [0, 0], [numFrames, peakCount]), [1], keepDims: false), divisor);

            contrast[band] = Engine.TensorSubtract(peakMean, valleyMean);
        }

        return contrast;
    }

    private Tensor<T> ComputeZeroCrossingRate(Tensor<T> audio, int numFrames)
    {
        var crossings = new int[numFrames];
        var intervals = new int[numFrames];
        var audioData = AudioEngineOps.Flatten(Engine, audio);
        int hopLength = HopLength;
        int windowLength = WindowLength;

        for (int frame = 0; frame < numFrames; frame++)
        {
            int startSample = frame * hopLength;
            int validSamples = Math.Min(windowLength, audioData.Length - startSample);
            if (validSamples <= 1) continue;

            // Strict sign changes between consecutive samples of the frame
            crossings[frame] = AudioEngineOps.StrictSignChangeCount(Engine, Engine.TensorSlice(audioData, [startSample], [validSamples]));
            intervals[frame] = validSamples - 1;
        }

        // crossings / (validSamples - 1), 0 for frames with fewer than two samples
        var intervalTensor = new Tensor<int>(intervals, [numFrames]).Cast<T>();
        return WhereOrZero(
            Engine.TensorGreaterThan(intervalTensor, NumOps.Zero),
            Engine.TensorDivide(new Tensor<int>(crossings, [numFrames]).Cast<T>(), intervalTensor));
    }

    private static int CountEnabledFeatures(SpectralFeatureType types)
    {
        int count = 0;
        if (types.HasFlag(SpectralFeatureType.Centroid)) count++;
        if (types.HasFlag(SpectralFeatureType.Bandwidth)) count++;
        if (types.HasFlag(SpectralFeatureType.Rolloff)) count++;
        if (types.HasFlag(SpectralFeatureType.Flux)) count++;
        if (types.HasFlag(SpectralFeatureType.Flatness)) count++;
        if (types.HasFlag(SpectralFeatureType.Contrast)) count += 6; // 6 frequency bands
        if (types.HasFlag(SpectralFeatureType.ZeroCrossingRate)) count++;
        return count;
    }

    /// <summary>
    /// Gets the column index for a specific feature type in the extracted feature tensor.
    /// </summary>
    /// <param name="featureType">The feature type to find the index for.</param>
    /// <returns>
    /// The column index of the feature in the output tensor, or -1 if the feature is not enabled.
    /// </returns>
    /// <remarks>
    /// Feature indices depend on which features are enabled. For example, if only Flux and Flatness
    /// are enabled, Flux will be at index 0 and Flatness at index 1. With Basic features
    /// (Centroid, Bandwidth, Rolloff, Flux, Flatness), Flux will be at index 3.
    /// </remarks>
    public int GetFeatureIndex(SpectralFeatureType featureType)
    {
        if (!_featureTypes.HasFlag(featureType))
        {
            return -1; // Feature not enabled
        }

        int index = 0;

        // Features are added in this specific order in Extract()
        if (featureType == SpectralFeatureType.Centroid)
            return index;
        if (_featureTypes.HasFlag(SpectralFeatureType.Centroid))
            index++;

        if (featureType == SpectralFeatureType.Bandwidth)
            return index;
        if (_featureTypes.HasFlag(SpectralFeatureType.Bandwidth))
            index++;

        if (featureType == SpectralFeatureType.Rolloff)
            return index;
        if (_featureTypes.HasFlag(SpectralFeatureType.Rolloff))
            index++;

        if (featureType == SpectralFeatureType.Flux)
            return index;
        if (_featureTypes.HasFlag(SpectralFeatureType.Flux))
            index++;

        if (featureType == SpectralFeatureType.Flatness)
            return index;
        if (_featureTypes.HasFlag(SpectralFeatureType.Flatness))
            index++;

        if (featureType == SpectralFeatureType.Contrast)
            return index; // Returns first band index; 6 bands total
        if (_featureTypes.HasFlag(SpectralFeatureType.Contrast))
            index += 6;

        if (featureType == SpectralFeatureType.ZeroCrossingRate)
            return index;

        // Should not reach here if feature was enabled
        return -1;
    }
}
