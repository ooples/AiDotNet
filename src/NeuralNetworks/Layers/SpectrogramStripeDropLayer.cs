using AiDotNet.Attributes;
using AiDotNet.Interfaces;

namespace AiDotNet.NeuralNetworks.Layers;

/// <summary>
/// SpecAugment-style stripe dropping on a spectrogram during training (Park et al. 2019), as PANNs and HTS-AT
/// apply it inside the network (torchlibrosa <c>SpecAugmentation</c>): zeroes random bands of time frames and then
/// random bands of frequency bins in every training forward, and passes the input through unchanged otherwise.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// Input: <c>[batch, channels, time, frequency]</c>. For each sample, <c>timeStripes</c> times: a width
/// <c>w</c> is drawn from <c>[0, timeDropWidth)</c>, a start from <c>[0, T − w)</c>, and those frames are zeroed in
/// every channel; then the same along frequency with <c>freqDropWidth</c> and <c>freqStripes</c>. HTS-AT uses
/// widths 64 (time) and 8 (frequency) with two stripes each. A width is capped below the axis length, so a short
/// spectrogram is never wiped out entirely.
/// </para>
/// <para>The drop is a multiplication by a 0/1 mask, so gradients flow through the kept positions.</para>
/// <para><b>For Beginners:</b> While training, a few random slices of time and of pitch are blanked out, so the
/// model learns not to depend on any single moment or frequency band.</para>
/// </remarks>
[LayerCategory(LayerCategory.Regularization)]
[LayerTask(LayerTask.Regularization)]
[LayerProperty(IsTrainable = false, HasTrainingMode = true, TestInputShape = "2, 1, 16, 8", TestConstructorArgs = "4, 2, 2, 2")]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width,
    Direction = TensorLayoutDirection.Input)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width,
    Direction = TensorLayoutDirection.Output)]
[ElementWiseShape(Note = "Multiplies by a 0/1 mask of the input's own shape.")]
[AutoParameters]
public partial class SpectrogramStripeDropLayer<T> : LayerBase<T>
{
    private readonly int _timeDropWidth;
    private readonly int _timeStripes;
    private readonly int _freqDropWidth;
    private readonly int _freqStripes;
    private long _forwardCounter;

    /// <inheritdoc/>
    public override bool SupportsTraining => true;

    /// <summary>Creates the stripe dropper. HTS-AT: <c>(64, 2, 8, 2)</c>.</summary>
    /// <param name="timeDropWidth">Exclusive upper bound of a time stripe's width.</param>
    /// <param name="timeStripes">Time stripes per sample.</param>
    /// <param name="freqDropWidth">Exclusive upper bound of a frequency stripe's width.</param>
    /// <param name="freqStripes">Frequency stripes per sample.</param>
    public SpectrogramStripeDropLayer([LayerState] int timeDropWidth, [LayerState] int timeStripes,
        [LayerState] int freqDropWidth, [LayerState] int freqStripes)
        : base(new[] { -1, -1, -1 }, new[] { -1, -1, -1 })
    {
        if (timeDropWidth <= 0) throw new ArgumentOutOfRangeException(nameof(timeDropWidth));
        if (timeStripes < 0) throw new ArgumentOutOfRangeException(nameof(timeStripes));
        if (freqDropWidth <= 0) throw new ArgumentOutOfRangeException(nameof(freqDropWidth));
        if (freqStripes < 0) throw new ArgumentOutOfRangeException(nameof(freqStripes));
        _timeDropWidth = timeDropWidth;
        _timeStripes = timeStripes;
        _freqDropWidth = freqDropWidth;
        _freqStripes = freqStripes;
    }

    /// <inheritdoc/>
    protected override Tensor<T> ForwardTraced(Tensor<T> input)
    {
        if (input.Rank != 4)
            throw new ArgumentException(
                $"SpectrogramStripeDropLayer expects [batch, channels, time, frequency]; got rank {input.Rank}.", nameof(input));
        if (!IsTrainingMode || (_timeStripes == 0 && _freqStripes == 0)) return input;

        long counter = System.Threading.Interlocked.Increment(ref _forwardCounter);
        var random = RandomSeed.HasValue
            ? RandomHelper.CreateSeededRandom(unchecked((int)((uint)RandomSeed.Value * 2654435761u ^ (uint)counter)))
            : RandomHelper.CreateSecureRandom();

        int batch = input.Shape[0], channels = input.Shape[1], time = input.Shape[2], freq = input.Shape[3];
        var keepTime = new bool[batch, time];
        var keepFreq = new bool[batch, freq];
        for (int b = 0; b < batch; b++)
        {
            for (int t = 0; t < time; t++) keepTime[b, t] = true;
            for (int f = 0; f < freq; f++) keepFreq[b, f] = true;
            Drop(random, _timeDropWidth, _timeStripes, time, b, keepTime);
            Drop(random, _freqDropWidth, _freqStripes, freq, b, keepFreq);
        }

        var mask = new Tensor<T>(input._shape);
        for (int b = 0; b < batch; b++)
            for (int c = 0; c < channels; c++)
                for (int t = 0; t < time; t++)
                    for (int f = 0; f < freq; f++)
                        if (keepTime[b, t] && keepFreq[b, f]) mask[b, c, t, f] = NumOps.One;
        return Engine.TensorMultiply(input, mask);
    }

    private static void Drop(Random random, int dropWidth, int stripes, int length, int b, bool[,] keep)
    {
        int widthBound = Math.Min(dropWidth, length);
        for (int s = 0; s < stripes; s++)
        {
            int width = random.Next(widthBound);
            int start = random.Next(length - width);
            for (int i = start; i < start + width; i++) keep[b, i] = false;
        }
    }

    /// <inheritdoc/>
    internal override Dictionary<string, string> GetMetadata()
    {
        var metadata = base.GetMetadata();
        var invariant = System.Globalization.CultureInfo.InvariantCulture;
        metadata["TimeDropWidth"] = _timeDropWidth.ToString(invariant);
        metadata["TimeStripes"] = _timeStripes.ToString(invariant);
        metadata["FreqDropWidth"] = _freqDropWidth.ToString(invariant);
        metadata["FreqStripes"] = _freqStripes.ToString(invariant);
        return metadata;
    }

    /// <inheritdoc/>
    public override void ResetState() { }
}
