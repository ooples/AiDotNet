namespace AiDotNet.Initialization;

/// <summary>
/// Initializes a one-dimensional convolution as an octave-spaced FIR bandpass filterbank.
/// </summary>
/// <remarks>
/// <para>
/// The filters are ordered from low to high frequency. Their center frequencies are one octave
/// apart, with the uppermost band's high edge at Nyquist. Each symmetric, Hamming-windowed sinc
/// kernel has zero DC response and unit gain at its center frequency.
/// </para>
/// <para>
/// This is the initialization used by the FiNS room-impulse-response decoder. Steinmetz,
/// Ithapu, and Calamia report that octave-spaced bandpass initialization was critical for
/// convergence, while standard Kaiming initialization converged poorly.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public sealed class OctaveBandpassInitializationStrategy<T> : InitializationStrategyBase<T>
{
    /// <inheritdoc />
    public override bool IsLazy => false;

    /// <inheritdoc />
    public override bool LoadFromExternal => false;

    /// <inheritdoc />
    public override void InitializeWeights(Tensor<T> weights, int inputSize, int outputSize)
    {
        if (weights.Rank != 4 || weights.Shape[2] != 1)
        {
            throw new ArgumentException(
                "Octave bandpass initialization requires Conv1D weights shaped [outputChannels, inputChannels, 1, kernelSize].",
                nameof(weights));
        }

        int bandCount = weights.Shape[0];
        int inputChannels = weights.Shape[1];
        int kernelSize = weights.Shape[3];
        if (bandCount <= 0 || inputChannels <= 0 || kernelSize < 3)
        {
            throw new ArgumentException(
                "Octave bandpass initialization requires positive channel counts and a kernel size of at least three.",
                nameof(weights));
        }

        var destination = weights.AsWritableSpan();
        var kernel = new double[kernelSize];
        double centerSample = (kernelSize - 1) / 2.0;
        double octaveHalfWidth = Math.Sqrt(2.0);

        for (int band = 0; band < bandCount; band++)
        {
            int octavesBelowTop = bandCount - 1 - band;
            double centerFrequency = 0.5 / (octaveHalfWidth * Math.Pow(2.0, octavesBelowTop));
            double lowFrequency = centerFrequency / octaveHalfWidth;
            double highFrequency = Math.Min(0.5, centerFrequency * octaveHalfWidth);

            double mean = 0.0;
            for (int tap = 0; tap < kernelSize; tap++)
            {
                double offset = tap - centerSample;
                double idealBandpass = Math.Abs(offset) < 1e-12
                    ? 2.0 * (highFrequency - lowFrequency)
                    : (Math.Sin(2.0 * Math.PI * highFrequency * offset)
                        - Math.Sin(2.0 * Math.PI * lowFrequency * offset)) / (Math.PI * offset);
                double window = 0.54 - 0.46 * Math.Cos(2.0 * Math.PI * tap / (kernelSize - 1));
                kernel[tap] = idealBandpass * window;
                mean += kernel[tap];
            }

            // Windowing introduces a small DC residual. Removing it preserves symmetry and makes
            // every kernel a true bandpass filter rather than a bandpass-plus-DC filter.
            mean /= kernelSize;
            for (int tap = 0; tap < kernelSize; tap++)
            {
                kernel[tap] -= mean;
            }

            double responseReal = 0.0;
            double responseImaginary = 0.0;
            for (int tap = 0; tap < kernelSize; tap++)
            {
                double phase = -2.0 * Math.PI * centerFrequency * tap;
                responseReal += kernel[tap] * Math.Cos(phase);
                responseImaginary += kernel[tap] * Math.Sin(phase);
            }

            double centerGain = Math.Sqrt(responseReal * responseReal + responseImaginary * responseImaginary);
            if (centerGain <= 1e-12 || double.IsNaN(centerGain) || double.IsInfinity(centerGain))
            {
                throw new InvalidOperationException("Unable to normalize an octave bandpass kernel at its center frequency.");
            }

            double inputScale = 1.0 / Math.Sqrt(inputChannels);
            for (int inputChannel = 0; inputChannel < inputChannels; inputChannel++)
            {
                int kernelOffset = (band * inputChannels + inputChannel) * kernelSize;
                for (int tap = 0; tap < kernelSize; tap++)
                {
                    destination[kernelOffset + tap] = NumOps.FromDouble(kernel[tap] * inputScale / centerGain);
                }
            }
        }
    }

    /// <inheritdoc />
    public override void InitializeBiases(Tensor<T> biases)
    {
        ZeroInitializeBiases(biases);
    }
}
