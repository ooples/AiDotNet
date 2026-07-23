using AiDotNet.Interfaces;

namespace AiDotNet.Data.Transforms;

/// <summary>
/// Applies SpecAugment data augmentation (Park et al., 2019) to spectrogram tensors.
/// Performs time masking and frequency masking to improve model robustness.
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
/// <remarks>
/// <para>
/// SpecAugment applies two types of masking to spectrograms:
/// <list type="bullet">
/// <item><description><b>Frequency masking</b>: Masks consecutive frequency bins (vertical stripes).</description></item>
/// <item><description><b>Time masking</b>: Masks consecutive time frames (horizontal stripes).</description></item>
/// </list>
/// </para>
/// <para><b>For Beginners:</b> SpecAugment is a simple but effective augmentation for audio models.
/// It randomly "erases" parts of the spectrogram during training to prevent overfitting.
/// <code>
/// var augment = new SpecAugmentTransform&lt;float&gt;(freqMaskParam: 27, timeMaskParam: 100);
/// Tensor&lt;float&gt; augmented = augment.Apply(melSpectrogram);
/// </code>
/// </para>
/// <para>
/// Reference: Park, D.S. et al. "SpecAugment: A Simple Data Augmentation Method for ASR." Interspeech 2019.
/// </para>
/// </remarks>
public class SpecAugmentTransform<T> : ITransform<Tensor<T>, Tensor<T>>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly int _freqMaskParam;
    private readonly int _timeMaskParam;
    private readonly int _numFreqMasks;
    private readonly int _numTimeMasks;
    private readonly Random _random;

    /// <summary>
    /// Creates a new SpecAugment transform.
    /// </summary>
    /// <param name="freqMaskParam">Maximum width of frequency mask (F parameter). Default is 27.</param>
    /// <param name="timeMaskParam">Maximum width of time mask (T parameter). Default is 100.</param>
    /// <param name="numFreqMasks">Number of frequency masks to apply. Default is 1.</param>
    /// <param name="numTimeMasks">Number of time masks to apply. Default is 1.</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    public SpecAugmentTransform(
        int freqMaskParam = 27,
        int timeMaskParam = 100,
        int numFreqMasks = 1,
        int numTimeMasks = 1,
        int? seed = null)
    {
        _freqMaskParam = freqMaskParam;
        _timeMaskParam = timeMaskParam;
        _numFreqMasks = numFreqMasks;
        _numTimeMasks = numTimeMasks;
        _random = seed.HasValue ? Tensors.Helpers.RandomHelper.CreateSeededRandom(seed.Value) : Tensors.Helpers.RandomHelper.CreateSecureRandom();
    }

    /// <summary>
    /// Applies SpecAugment masking to a spectrogram tensor.
    /// </summary>
    /// <param name="input">Spectrogram tensor of shape [timeFrames, freqBins].</param>
    /// <returns>Masked spectrogram tensor (same shape, masked regions set to zero).</returns>
    public Tensor<T> Apply(Tensor<T> input)
    {
        if (input.Shape.Length < 2)
            return input;

        int timeFrames = input.Shape[0];
        int freqBins = input.Shape[1];

        // Clone input to avoid modifying original
        var result = input.Clone();
        var span = result.Data.Span;

        // Apply frequency masks
        T zero = NumOps.FromDouble(0.0);
        for (int m = 0; m < _numFreqMasks; m++)
        {
            int maskWidth = _random.Next(0, Math.Min(_freqMaskParam, freqBins) + 1);
            if (maskWidth == 0) continue;

            int maskStart = _random.Next(0, freqBins - maskWidth + 1);

            for (int t = 0; t < timeFrames; t++)
            {
                for (int f = maskStart; f < maskStart + maskWidth; f++)
                {
                    span[t * freqBins + f] = zero;
                }
            }
        }

        // Apply time masks
        for (int m = 0; m < _numTimeMasks; m++)
        {
            int maskWidth = _random.Next(0, Math.Min(_timeMaskParam, timeFrames) + 1);
            if (maskWidth == 0) continue;

            int maskStart = _random.Next(0, timeFrames - maskWidth + 1);

            for (int t = maskStart; t < maskStart + maskWidth; t++)
            {
                for (int f = 0; f < freqBins; f++)
                {
                    span[t * freqBins + f] = zero;
                }
            }
        }

        return result;
    }
}
