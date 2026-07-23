using AiDotNet.Augmentation;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Augmentation.Image;

/// <summary>
/// SnapMix (Huang et al., 2021) - semantically proportional mixing for fine-grained recognition.
/// Uses class activation map (CAM) inspired saliency for proportional label mixing.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class SnapMix<T> : ImageMixingAugmenterBase<T>
{
    public SnapMix(double alpha = 5.0, double probability = 0.5) : base(probability, alpha)
    {
    }

    /// <summary>
    /// Mixes two images using saliency-proportional mixing.
    /// </summary>
    public ImageTensor<T> ApplySnapMix(
        ImageTensor<T> image1, ImageTensor<T> image2,
        Vector<T>? labels1, Vector<T>? labels2, AugmentationContext<T> context)
    {
        if (image1.Height != image2.Height || image1.Width != image2.Width)
            image2 = new Resize<T>(image1.Height, image1.Width).Apply(image2, context);

        var result = image1.Clone();
        double betaLambda = SampleLambda(context);

        // CutMix-style region selection
        int cutH = (int)(image1.Height * Math.Sqrt(1 - betaLambda));
        int cutW = (int)(image1.Width * Math.Sqrt(1 - betaLambda));
        cutH = Math.Max(1, Math.Min(cutH, image1.Height));
        cutW = Math.Max(1, Math.Min(cutW, image1.Width));

        int y1 = context.GetRandomInt(0, Math.Max(1, image1.Height - cutH));
        int x1 = context.GetRandomInt(0, Math.Max(1, image1.Width - cutW));

        // Compute saliency weights for proportional label mixing
        double salTotal1 = 0, salPaste = 0;
        for (int y = 0; y < image1.Height; y++)
            for (int x = 0; x < image1.Width; x++)
            {
                double s = 0;
                for (int c = 0; c < image1.Channels; c++)
                    s += NumOps.ToDouble(image1.GetPixel(y, x, c));
                salTotal1 += s;

                if (y >= y1 && y < y1 + cutH && x >= x1 && x < x1 + cutW)
                    salPaste += s;
            }

        // Paste region from image2
        for (int y = y1; y < y1 + cutH && y < image1.Height; y++)
            for (int x = x1; x < x1 + cutW && x < image1.Width; x++)
                for (int c = 0; c < Math.Min(image1.Channels, image2.Channels); c++)
                    result.SetPixel(y, x, c, image2.GetPixel(y, x, c));

        // Lambda proportional to saliency
        double lambda = salTotal1 > 1e-10 ? 1.0 - salPaste / salTotal1 : betaLambda;
        LastMixingLambda = NumOps.FromDouble(lambda);

        if (labels1 is not null && labels2 is not null)
        {
            var args = new LabelMixingEventArgs<T>(
                labels1, labels2, LastMixingLambda,
                context.SampleIndex, -1, MixingStrategy.Custom);
            RaiseLabelMixing(args);
        }

        return result;
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
