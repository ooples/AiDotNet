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

        // Compute saliency weights for proportional label mixing: total intensity of image1 and
        // the intensity inside the pasted region
        var engine = Engine;
        var planar1 = image1.ToPlanarTensor();
        int pasteH = Math.Max(0, Math.Min(y1 + cutH, image1.Height) - y1);
        int pasteW = Math.Max(0, Math.Min(x1 + cutW, image1.Width) - x1);
        T salTotal1 = engine.TensorSum(planar1);
        T salPaste = NumOps.Zero;
        if (pasteH > 0 && pasteW > 0)
        {
            salPaste = engine.TensorSum(engine.TensorSlice(planar1, [0, y1, x1], [image1.Channels, pasteH, pasteW]));

            // Paste region from image2
            int channels = Math.Min(image1.Channels, image2.Channels);
            var patch = engine.TensorSlice(image2.ToPlanarTensor(), [0, y1, x1], [channels, pasteH, pasteW]);
            result.SetFromPlanarTensor(engine.TensorSetSlice(result.ToPlanarTensor(), patch, [0, y1, x1]));
        }

        // Lambda proportional to saliency
        LastMixingLambda = NumOps.GreaterThan(salTotal1, NumOps.FromDouble(1e-10))
            ? NumOps.Subtract(NumOps.One, NumOps.Divide(salPaste, salTotal1))
            : NumOps.FromDouble(betaLambda);

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
