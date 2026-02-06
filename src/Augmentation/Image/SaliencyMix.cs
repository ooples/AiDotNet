using AiDotNet.Augmentation;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Augmentation.Image;

/// <summary>
/// SaliencyMix - saliency-guided image mixing that pastes salient regions.
/// Uses simple gradient-based saliency estimation.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class SaliencyMix<T> : ImageMixingAugmenterBase<T>
{
    public SaliencyMix(double alpha = 1.0, double probability = 0.5) : base(probability, alpha)
    {
    }

    /// <summary>
    /// Mixes two images using saliency-guided selection of the paste region.
    /// </summary>
    public ImageTensor<T> ApplySaliencyMix(ImageTensor<T> image1, ImageTensor<T> image2,
        Vector<T>? labels1, Vector<T>? labels2, AugmentationContext<T> context)
    {
        if (image1.Height != image2.Height || image1.Width != image2.Width)
            image2 = new Resize<T>(image1.Height, image1.Width).Apply(image2, context);

        var result = image1.Clone();

        // Compute simple saliency map of image2 using gradient magnitude
        var saliency = ComputeSaliency(image2);

        // Find most salient region
        int bestY = 0, bestX = 0;
        double bestSaliency = double.MinValue;
        double lambda = SampleLambda(context);
        LastMixingLambda = NumOps.FromDouble(lambda);

        int cropH = (int)(image1.Height * Math.Sqrt(1 - lambda));
        int cropW = (int)(image1.Width * Math.Sqrt(1 - lambda));
        cropH = Math.Max(1, Math.Min(cropH, image1.Height));
        cropW = Math.Max(1, Math.Min(cropW, image1.Width));

        // Search for most salient location (stride for efficiency)
        int stride = Math.Max(1, Math.Min(cropH, cropW) / 4);
        for (int y = 0; y <= image2.Height - cropH; y += stride)
            for (int x = 0; x <= image2.Width - cropW; x += stride)
            {
                double s = 0;
                for (int dy = 0; dy < cropH; dy += stride)
                    for (int dx = 0; dx < cropW; dx += stride)
                        if (y + dy < image2.Height && x + dx < image2.Width)
                            s += saliency[y + dy, x + dx];

                if (s > bestSaliency) { bestSaliency = s; bestY = y; bestX = x; }
            }

        // Paste most salient region from image2 onto image1
        int pasteY = context.GetRandomInt(0, Math.Max(1, image1.Height - cropH));
        int pasteX = context.GetRandomInt(0, Math.Max(1, image1.Width - cropW));

        for (int y = 0; y < cropH; y++)
            for (int x = 0; x < cropW; x++)
            {
                if (pasteY + y >= image1.Height || pasteX + x >= image1.Width) continue;
                if (bestY + y >= image2.Height || bestX + x >= image2.Width) continue;
                for (int c = 0; c < Math.Min(image1.Channels, image2.Channels); c++)
                    result.SetPixel(pasteY + y, pasteX + x, c,
                        image2.GetPixel(bestY + y, bestX + x, c));
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

    private static double[,] ComputeSaliency(ImageTensor<T> image)
    {
        var saliency = new double[image.Height, image.Width];

        for (int y = 1; y < image.Height - 1; y++)
            for (int x = 1; x < image.Width - 1; x++)
            {
                double grad = 0;
                for (int c = 0; c < image.Channels; c++)
                {
                    double dx = NumOps.ToDouble(image.GetPixel(y, x + 1, c)) -
                                NumOps.ToDouble(image.GetPixel(y, x - 1, c));
                    double dy = NumOps.ToDouble(image.GetPixel(y + 1, x, c)) -
                                NumOps.ToDouble(image.GetPixel(y - 1, x, c));
                    grad += Math.Sqrt(dx * dx + dy * dy);
                }
                saliency[y, x] = grad;
            }

        return saliency;
    }

    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        return data.Clone();
    }

    public override IDictionary<string, object> GetParameters()
    {
        return base.GetParameters();
    }
}
