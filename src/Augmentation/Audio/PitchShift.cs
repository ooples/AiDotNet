using AiDotNet.Tensors;

namespace AiDotNet.Augmentation.Audio;

/// <summary>
/// Shifts the pitch of audio without changing tempo using WSOLA (Waveform Similarity Overlap-Add).
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Pitch shifting makes audio sound higher or lower,
/// like the difference between a high and low voice, while keeping the same duration.</para>
/// <para><b>Algorithm:</b> Uses WSOLA (Waveform Similarity Overlap-Add) for time-stretching
/// followed by resampling. This preserves audio quality better than simple resampling.</para>
/// <para><b>When to use:</b>
/// <list type="bullet">
/// <item>Speech recognition to handle different voice pitches</item>
/// <item>Music analysis to handle different keys</item>
/// <item>Voice cloning and synthesis training</item>
/// </list>
/// </para>
/// <para><b>Semitone reference:</b> 12 semitones = 1 octave (doubling/halving frequency)</para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class PitchShift<T> : AudioAugmenterBase<T>
{
    // WSOLA parameters
    private readonly int _frameSize;
    private readonly int _hopSize;
    private readonly int _searchRange;

    /// <summary>
    /// Gets the minimum pitch shift in semitones.
    /// </summary>
    /// <remarks>
    /// <para>Default: -2.0 semitones (about -12% frequency)</para>
    /// <para>Negative values lower the pitch.</para>
    /// </remarks>
    public double MinSemitones { get; }

    /// <summary>
    /// Gets the maximum pitch shift in semitones.
    /// </summary>
    /// <remarks>
    /// <para>Default: 2.0 semitones (about +12% frequency)</para>
    /// <para>Positive values raise the pitch.</para>
    /// </remarks>
    public double MaxSemitones { get; }

    /// <summary>
    /// Creates a new pitch shift augmentation.
    /// </summary>
    /// <param name="minSemitones">Minimum pitch shift in semitones (default: -2.0).</param>
    /// <param name="maxSemitones">Maximum pitch shift in semitones (default: 2.0).</param>
    /// <param name="probability">Probability of applying this augmentation (default: 0.5).</param>
    /// <param name="sampleRate">Sample rate of the audio in Hz (default: 16000).</param>
    public PitchShift(
        double minSemitones = -2.0,
        double maxSemitones = 2.0,
        double probability = 0.5,
        int sampleRate = 16000) : base(probability, sampleRate)
    {
        if (minSemitones > maxSemitones)
        {
            throw new ArgumentException("Minimum semitones must be less than or equal to maximum semitones.");
        }

        MinSemitones = minSemitones;
        MaxSemitones = maxSemitones;

        // WSOLA frame parameters based on sample rate
        // Frame size ~25ms, hop size ~6.25ms (75% overlap)
        _frameSize = Math.Max(256, (int)(sampleRate * 0.025));
        _hopSize = _frameSize / 4;
        _searchRange = _hopSize / 2;
    }

    /// <inheritdoc />
    protected override Tensor<T> ApplyAugmentation(Tensor<T> data, AugmentationContext<T> context)
    {
        // Sample random pitch shift
        double semitones = context.GetRandomDouble(MinSemitones, MaxSemitones);

        // Convert semitones to frequency ratio
        // ratio = 2^(semitones/12)
        double pitchRatio = Math.Pow(2, semitones / 12.0);

        if (Math.Abs(pitchRatio - 1.0) < 0.001)
        {
            return data.Clone();
        }

        // Apply pitch shifting: time-stretch then resample
        return ApplyPitchShift(data, pitchRatio);
    }

    /// <summary>
    /// Applies pitch shift using WSOLA time-stretching followed by resampling.
    /// </summary>
    private Tensor<T> ApplyPitchShift(Tensor<T> waveform, double pitchRatio)
    {
        int samples = GetSampleCount(waveform);
        if (samples < _frameSize)
        {
            return waveform.Clone();
        }

        // Convert to double array for processing
        double[] input = new double[samples];
        for (int i = 0; i < samples; i++)
        {
            input[i] = NumOps.ToDouble(waveform[i]);
        }

        // Step 1: Time-stretch by pitchRatio using WSOLA
        // To shift pitch up by ratio, we stretch time by ratio, then resample back
        double[] stretched = TimeStretchWSOLA(input, pitchRatio);

        // Step 2: Resample back to original length
        double[] output = ResampleLinear(stretched, samples);

        // Convert back to tensor
        var result = new Tensor<T>(waveform.Shape);
        for (int i = 0; i < samples; i++)
        {
            result[i] = NumOps.FromDouble(output[i]);
        }

        return result;
    }

    /// <summary>
    /// Time-stretches audio using WSOLA algorithm.
    /// </summary>
    /// <param name="input">Input audio samples.</param>
    /// <param name="stretchFactor">Factor to stretch by (>1 = longer, &lt;1 = shorter).</param>
    /// <returns>Time-stretched audio.</returns>
    private double[] TimeStretchWSOLA(double[] input, double stretchFactor)
    {
        int inputLength = input.Length;
        int outputLength = (int)(inputLength * stretchFactor);
        if (outputLength < 1) outputLength = 1;

        double[] output = new double[outputLength];
        double[] windowWeights = new double[outputLength];

        // Precompute Hann window
        double[] hannWindow = new double[_frameSize];
        for (int i = 0; i < _frameSize; i++)
        {
            hannWindow[i] = 0.5 * (1 - Math.Cos(2 * Math.PI * i / (_frameSize - 1)));
        }

        // Synthesis hop size (input hop scaled by stretch factor)
        int synthesisHop = Math.Max(1, (int)(_hopSize * stretchFactor));

        // Analysis position tracks where we read from input
        double analysisPos = 0;
        int synthesisPos = 0;

        while (synthesisPos + _frameSize <= outputLength && (int)analysisPos + _frameSize <= inputLength)
        {
            int currentAnalysisPos = (int)analysisPos;

            // For subsequent frames, find best match using cross-correlation
            if (synthesisPos > 0)
            {
                int bestOffset = FindBestOffset(input, output, currentAnalysisPos, synthesisPos, inputLength, outputLength);
                currentAnalysisPos = Math.Max(0, Math.Min(inputLength - _frameSize, currentAnalysisPos + bestOffset));
            }

            // Overlap-add the frame with Hann window
            for (int i = 0; i < _frameSize && synthesisPos + i < outputLength; i++)
            {
                if (currentAnalysisPos + i < inputLength)
                {
                    output[synthesisPos + i] += input[currentAnalysisPos + i] * hannWindow[i];
                    windowWeights[synthesisPos + i] += hannWindow[i];
                }
            }

            analysisPos += _hopSize;
            synthesisPos += synthesisHop;
        }

        // Normalize by window weights to prevent amplitude changes
        for (int i = 0; i < outputLength; i++)
        {
            if (windowWeights[i] > 0.001)
            {
                output[i] /= windowWeights[i];
            }
        }

        return output;
    }

    /// <summary>
    /// Finds the best offset for frame alignment using cross-correlation.
    /// </summary>
    private int FindBestOffset(double[] input, double[] output, int analysisPos, int synthesisPos, int inputLength, int outputLength)
    {
        int bestOffset = 0;
        double bestCorrelation = double.MinValue;

        // Search for best matching position within search range
        int searchStart = -Math.Min(_searchRange, analysisPos);
        int searchEnd = Math.Min(_searchRange, inputLength - analysisPos - _frameSize);

        // Use overlap region for correlation (last part of previous frame)
        int overlapSize = Math.Min(_frameSize / 4, synthesisPos);
        if (overlapSize < 4)
        {
            return 0;
        }

        for (int offset = searchStart; offset <= searchEnd; offset++)
        {
            double correlation = 0;
            double norm1 = 0;
            double norm2 = 0;

            int testPos = analysisPos + offset;
            if (testPos < 0 || testPos + overlapSize > inputLength)
            {
                continue;
            }

            // Compute normalized cross-correlation
            for (int i = 0; i < overlapSize; i++)
            {
                int outIdx = synthesisPos - overlapSize + i;
                if (outIdx >= 0 && outIdx < outputLength && testPos + i < inputLength)
                {
                    double val1 = output[outIdx];
                    double val2 = input[testPos + i];
                    correlation += val1 * val2;
                    norm1 += val1 * val1;
                    norm2 += val2 * val2;
                }
            }

            double denominator = Math.Sqrt(norm1 * norm2);
            if (denominator > 0.0001)
            {
                correlation /= denominator;
            }

            if (correlation > bestCorrelation)
            {
                bestCorrelation = correlation;
                bestOffset = offset;
            }
        }

        return bestOffset;
    }

    /// <summary>
    /// Resamples audio to target length using linear interpolation.
    /// </summary>
    private static double[] ResampleLinear(double[] input, int targetLength)
    {
        int inputLength = input.Length;
        double[] output = new double[targetLength];

        double ratio = (double)(inputLength - 1) / (targetLength - 1);

        for (int i = 0; i < targetLength; i++)
        {
            double srcPos = i * ratio;
            int srcIndex = (int)srcPos;
            double frac = srcPos - srcIndex;

            if (srcIndex >= inputLength - 1)
            {
                output[i] = input[inputLength - 1];
            }
            else
            {
                output[i] = input[srcIndex] * (1 - frac) + input[srcIndex + 1] * frac;
            }
        }

        return output;
    }

    /// <inheritdoc />
    public override IDictionary<string, object> GetParameters()
    {
        var parameters = base.GetParameters();
        parameters["minSemitones"] = MinSemitones;
        parameters["maxSemitones"] = MaxSemitones;
        parameters["algorithm"] = "WSOLA";
        return parameters;
    }
}
