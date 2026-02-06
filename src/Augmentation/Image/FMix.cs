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
        double maskSum = 0;
        long totalPixels = (long)image1.Height * image1.Width;
        for (int y = 0; y < image1.Height; y++)
            for (int x = 0; x < image1.Width; x++)
                maskSum += mask[y, x];
        lambda = maskSum / totalPixels;
        LastMixingLambda = NumOps.FromDouble(lambda);

        var result = image1.Clone();

        for (int y = 0; y < image1.Height; y++)
            for (int x = 0; x < image1.Width; x++)
            {
                double m = mask[y, x];
                for (int c = 0; c < Math.Min(image1.Channels, image2.Channels); c++)
                {
                    double v1 = NumOps.ToDouble(image1.GetPixel(y, x, c));
                    double v2 = NumOps.ToDouble(image2.GetPixel(y, x, c));
                    result.SetPixel(y, x, c, NumOps.FromDouble(v1 * m + v2 * (1 - m)));
                }
            }

        if (labels1 is not null && labels2 is not null)
        {
            var args = new LabelMixingEventArgs<T>(
                labels1, labels2, LastMixingLambda,
                context.SampleIndex, -1, MixingStrategy.Custom);
            RaiseLabelMixing(args);
        }

        return result;
    }

    private double[,] GenerateFourierMask(int height, int width, double targetLambda, AugmentationContext<T> context)
    {
        // Generate random low-frequency pattern
        var mask = new double[height, width];
        int freqH = Math.Max(2, height / 8);
        int freqW = Math.Max(2, width / 8);

        // Low-frequency noise grid
        var lowFreq = new double[freqH, freqW];
        for (int y = 0; y < freqH; y++)
            for (int x = 0; x < freqW; x++)
            {
                double dist = Math.Sqrt(((double)y * y + (double)x * x) / ((double)freqH * freqH + (double)freqW * freqW));
                double decay = Math.Pow(Math.Max(dist, 0.01), -DecayPower);
                lowFreq[y, x] = context.SampleGaussian(0, 1) * decay;
            }

        // Upsample to full resolution via bilinear interpolation
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
            {
                double fy = (double)y / height * (freqH - 1);
                double fx = (double)x / width * (freqW - 1);
                int y0 = (int)fy; int x0 = (int)fx;
                int y1 = Math.Min(y0 + 1, freqH - 1);
                int x1 = Math.Min(x0 + 1, freqW - 1);
                double ty = fy - y0; double tx = fx - x0;

                mask[y, x] = lowFreq[y0, x0] * (1 - ty) * (1 - tx) +
                             lowFreq[y1, x0] * ty * (1 - tx) +
                             lowFreq[y0, x1] * (1 - ty) * tx +
                             lowFreq[y1, x1] * ty * tx;
            }

        // Threshold to binary mask at the targetLambda percentile
        var values = new double[height * width];
        int idx = 0;
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
                values[idx++] = mask[y, x];
        Array.Sort(values);
        int thresholdIdx = Math.Max(0, Math.Min(values.Length - 1, (int)((1 - targetLambda) * values.Length)));
        double threshold = values[thresholdIdx];

        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
                mask[y, x] = mask[y, x] > threshold ? 1.0 : 0.0;

        return mask;
    }

    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        return data.Clone();
    }

    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters();
        p["decay_power"] = DecayPower;
        return p;
    }
}
