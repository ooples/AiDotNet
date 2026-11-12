using AiDotNet.Augmentation;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Augmentation.Image;

/// <summary>
/// FMix augmentation (Harris et al., 2020) - mixes images using Fourier-space masks.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class FMix<T> : ImageMixingAugmenterBase<T>
{
    public double DecayPower { get; }

    public FMix(double alpha = 1.0, double decayPower = 3.0,
        double probability = 0.5) : base(probability, alpha)
    {
        DecayPower = decayPower;
    }

    /// <summary>
    /// Mixes two images using a Fourier-space generated mask.
    /// </summary>
    public ImageTensor<T> ApplyFMix(ImageTensor<T> image1, ImageTensor<T> image2,
        Vector<T>? labels1, Vector<T>? labels2, AugmentationContext<T> context)
    {
        if (image1.Height != image2.Height || image1.Width != image2.Width)
            image2 = new Resize<T>(image1.Height, image1.Width).Apply(image2, context);

        // Sample lambda from Beta(alpha, alpha) distribution
        double lambda = SampleLambda(context);

        // Generate low-frequency mask with target lambda
        var mask = GenerateFourierMask(image1.Height, image1.Width, lambda, context);

        // Recompute actual lambda from mask area
        long totalPixels = (long)image1.Height * image1.Width;
        LastMixingLambda = NumOps.Divide(Engine.TensorSum(mask), NumOps.FromDouble(totalPixels));

        var result = image1.Clone();

        // result = image1 * m + image2 * (1 - m) for the channels both images have
        var engine = Engine;
        int channels = Math.Min(image1.Channels, image2.Channels);
        var m = engine.Reshape(mask, [1, image1.Height, image1.Width]);
        var planar1 = image1.ToPlanarTensor();
        var mixed = engine.TensorAdd(
            engine.TensorMultiply(ImageTensor<T>.LeadingChannels(planar1, channels), m),
            engine.TensorMultiply(ImageTensor<T>.LeadingChannels(image2.ToPlanarTensor(), channels), engine.ScalarMinusTensor(NumOps.One, m)));
        result.SetFromPlanarTensor(ImageTensor<T>.ReplaceLeadingBlock(planar1, mixed));

        if (labels1 is not null && labels2 is not null)
        {
            var args = new LabelMixingEventArgs<T>(
                labels1, labels2, LastMixingLambda,
                context.SampleIndex, -1, MixingStrategy.Custom);
            RaiseLabelMixing(args);
        }

        return result;
    }

    private Tensor<T> GenerateFourierMask(int height, int width, double targetLambda, AugmentationContext<T> context)
    {
        var engine = Engine;
        int freqH = Math.Max(2, height / 8);
        int freqW = Math.Max(2, width / 8);

        // Low-frequency noise grid: N(0, 1) * max(dist, 0.01)^(-DecayPower) with
        // dist = sqrt((y^2 + x^2) / (freqH^2 + freqW^2))
        var gy = engine.Reshape(engine.TensorArange<T>(0, freqH, 1), [freqH, 1]);
        var gx = engine.Reshape(engine.TensorArange<T>(0, freqW, 1), [1, freqW]);
        int[] gridShape = [freqH, freqW];
        var radiusSq = engine.TensorAdd(
            engine.TensorBroadcastTo(engine.TensorMultiply(gy, gy), gridShape),
            engine.TensorBroadcastTo(engine.TensorMultiply(gx, gx), gridShape));
        var dist = engine.TensorSqrt(engine.TensorDivideScalar(radiusSq, NumOps.FromDouble((double)freqH * freqH + (double)freqW * freqW)));
        var decay = engine.TensorPower(engine.TensorMax(dist, NumOps.FromDouble(0.01)), NumOps.FromDouble(-DecayPower));
        var lowFreq = engine.TensorMultiply(AugmentationRandom<T>.StandardNormal(context, gridShape), decay);

        // Upsample to full resolution via bilinear interpolation at (y / height * (freqH - 1), x / width * (freqW - 1))
        var rowPositions = engine.TensorMultiplyScalar(
            engine.TensorDivideScalar(engine.TensorArange<T>(0, height, 1), NumOps.FromDouble(height)), NumOps.FromDouble(freqH - 1));
        var colPositions = engine.TensorMultiplyScalar(
            engine.TensorDivideScalar(engine.TensorArange<T>(0, width, 1), NumOps.FromDouble(width)), NumOps.FromDouble(freqW - 1));
        var mask = ImageTensor<T>.InterpolateGrid(lowFreq, rowPositions, colPositions, smoothstep: false);

        // Threshold to binary mask at the targetLambda percentile
        var sorted = engine.TensorSort(engine.Reshape(mask, [height * width]), 0, false).Values;
        int thresholdIdx = Math.Max(0, Math.Min(height * width - 1, (int)((1 - targetLambda) * height * width)));
        T threshold = sorted.GetFlat(thresholdIdx);
        return engine.TensorGreaterThan(mask, threshold);
    }

    /// <inheritdoc />
    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        return data.Clone();
    }

    /// <inheritdoc />
    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters();
        p["decay_power"] = DecayPower;
        return p;
    }
}
