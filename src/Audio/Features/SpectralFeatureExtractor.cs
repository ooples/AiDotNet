using AiDotNet.Diffusion;
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
        // Compute STFT - returns Tensor<Complex<T>> with shape [numFrames, numFreqs]
        var stftResult = _stft.Forward(audio);

        int numFrames = stftResult.Shape[0];
        int numFreqs = stftResult.Shape[1];

        // Compute magnitude spectrum
        var magnitude = new double[numFrames, numFreqs];
        var power = new double[numFrames, numFreqs];

        for (int frame = 0; frame < numFrames; frame++)
        {
            for (int freq = 0; freq < numFreqs; freq++)
            {
                var complex = stftResult[frame, freq];
                double real = NumOps.ToDouble(complex.Real);
                double imag = NumOps.ToDouble(complex.Imaginary);
                magnitude[frame, freq] = Math.Sqrt(real * real + imag * imag);
                power[frame, freq] = real * real + imag * imag;
            }
        }

        // Calculate frequency bins in Hz
        var freqBins = new double[numFreqs];
        // Guard against division by zero when numFreqs == 1 (only DC component)
        double freqScale = numFreqs > 1 ? (double)SampleRate / (2 * (numFreqs - 1)) : 0.0;
        for (int freq = 0; freq < numFreqs; freq++)
        {
            freqBins[freq] = freq * freqScale;
        }

        // Extract features
        var features = new List<double[]>();

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
            features.Add(ComputeSpectralRolloff(magnitude, freqBins, numFrames, numFreqs));
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
            var contrast = ComputeSpectralContrast(magnitude, numFrames, numFreqs);
            foreach (var band in contrast)
            {
                features.Add(band);
            }
        }

        if (_featureTypes.HasFlag(SpectralFeatureType.ZeroCrossingRate))
        {
            features.Add(ComputeZeroCrossingRate(audio, numFrames));
        }

        // Combine all features into result tensor
        var result = new Tensor<T>([numFrames, _featureDimension]);
        int featureOffset = 0;

        foreach (var featureArray in features)
        {
            for (int frame = 0; frame < numFrames; frame++)
            {
                result[frame, featureOffset] = NumOps.FromDouble(featureArray[frame]);
            }
            featureOffset++;
        }

        return result;
    }

    private double[] ComputeSpectralCentroid(double[,] magnitude, double[] freqBins, int numFrames, int numFreqs)
    {
        var centroid = new double[numFrames];

        for (int frame = 0; frame < numFrames; frame++)
        {
            double numerator = 0;
            double denominator = 0;

            for (int freq = 0; freq < numFreqs; freq++)
            {
                numerator += freqBins[freq] * magnitude[frame, freq];
                denominator += magnitude[frame, freq];
            }

            centroid[frame] = denominator > 1e-10 ? numerator / denominator : 0;
        }

        return centroid;
    }

    private double[] ComputeSpectralBandwidth(double[,] magnitude, double[] freqBins, double[] centroids, int numFrames, int numFreqs)
    {
        var bandwidth = new double[numFrames];

        for (int frame = 0; frame < numFrames; frame++)
        {
            double numerator = 0;
            double denominator = 0;

            for (int freq = 0; freq < numFreqs; freq++)
            {
                double diff = freqBins[freq] - centroids[frame];
                numerator += magnitude[frame, freq] * diff * diff;
                denominator += magnitude[frame, freq];
            }

            bandwidth[frame] = denominator > 1e-10 ? Math.Sqrt(numerator / denominator) : 0;
        }

        return bandwidth;
    }

    private double[] ComputeSpectralRolloff(double[,] magnitude, double[] freqBins, int numFrames, int numFreqs)
    {
        var rolloff = new double[numFrames];

        for (int frame = 0; frame < numFrames; frame++)
        {
            double totalEnergy = 0;
            for (int freq = 0; freq < numFreqs; freq++)
            {
                totalEnergy += magnitude[frame, freq];
            }

            double threshold = totalEnergy * _rolloffPercentage;
            double cumEnergy = 0;

            for (int freq = 0; freq < numFreqs; freq++)
            {
                cumEnergy += magnitude[frame, freq];
                if (cumEnergy >= threshold)
                {
                    rolloff[frame] = freqBins[freq];
                    break;
                }
            }
        }

        return rolloff;
    }

    private static double[] ComputeSpectralFlux(double[,] magnitude, int numFrames, int numFreqs)
    {
        var flux = new double[numFrames];

        for (int frame = 1; frame < numFrames; frame++)
        {
            double sum = 0;
            for (int freq = 0; freq < numFreqs; freq++)
            {
                double diff = magnitude[frame, freq] - magnitude[frame - 1, freq];
                sum += diff * diff;
            }
            flux[frame] = Math.Sqrt(sum);
        }

        return flux;
    }

    private static double[] ComputeSpectralFlatness(double[,] magnitude, int numFrames, int numFreqs)
    {
        var flatness = new double[numFrames];

        for (int frame = 0; frame < numFrames; frame++)
        {
            double logSum = 0;
            double sum = 0;
            int count = 0;

            for (int freq = 0; freq < numFreqs; freq++)
            {
                double val = magnitude[frame, freq];
                if (val > 1e-10)
                {
                    logSum += Math.Log(val);
                    sum += val;
                    count++;
                }
            }

            if (count > 0 && sum > 1e-10)
            {
                double geometricMean = Math.Exp(logSum / count);
                double arithmeticMean = sum / count;
                flatness[frame] = geometricMean / arithmeticMean;
            }
        }

        return flatness;
    }

    private static double[][] ComputeSpectralContrast(double[,] magnitude, int numFrames, int numFreqs)
    {
        const int numBands = 6;
        var contrast = new double[numBands][];
        for (int b = 0; b < numBands; b++)
        {
            contrast[b] = new double[numFrames];
        }

        int freqsPerBand = numFreqs / numBands;

        for (int frame = 0; frame < numFrames; frame++)
        {
            for (int band = 0; band < numBands; band++)
            {
                int startFreq = band * freqsPerBand;
                int endFreq = Math.Min((band + 1) * freqsPerBand, numFreqs);

                // Get magnitudes for this band
                var bandMags = new List<double>();
                for (int freq = startFreq; freq < endFreq; freq++)
                {
                    bandMags.Add(magnitude[frame, freq]);
                }

                if (bandMags.Count > 0)
                {
                    bandMags.Sort();
                    int peakCount = Math.Max(1, bandMags.Count / 4);

                    // Peak mean (top 25%)
                    double peakMean = 0;
                    for (int i = bandMags.Count - peakCount; i < bandMags.Count; i++)
                    {
                        peakMean += bandMags[i];
                    }
                    peakMean /= peakCount;

                    // Valley mean (bottom 25%)
                    double valleyMean = 0;
                    for (int i = 0; i < peakCount; i++)
                    {
                        valleyMean += bandMags[i];
                    }
                    valleyMean /= peakCount;

                    contrast[band][frame] = peakMean - valleyMean;
                }
            }
        }

        return contrast;
    }

    private double[] ComputeZeroCrossingRate(Tensor<T> audio, int numFrames)
    {
        var zcr = new double[numFrames];
        var audioData = audio.ToArray();
        int hopLength = HopLength;
        int windowLength = WindowLength;

        for (int frame = 0; frame < numFrames; frame++)
        {
            int startSample = frame * hopLength;
            int crossings = 0;

            for (int i = startSample + 1; i < startSample + windowLength && i < audioData.Length; i++)
            {
                double prev = NumOps.ToDouble(audioData[i - 1]);
                double curr = NumOps.ToDouble(audioData[i]);

                if ((prev > 0 && curr < 0) || (prev < 0 && curr > 0))
                {
                    crossings++;
                }
            }

            int validSamples = Math.Min(windowLength, audioData.Length - startSample);
            zcr[frame] = validSamples > 1 ? (double)crossings / (validSamples - 1) : 0;
        }

        return zcr;
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
