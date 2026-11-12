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

        double lambda = SampleLambda(context);
        LastMixingLambda = NumOps.FromDouble(lambda);

        int cropH = (int)(image1.Height * Math.Sqrt(1 - lambda));
        int cropW = (int)(image1.Width * Math.Sqrt(1 - lambda));
        cropH = Math.Max(1, Math.Min(cropH, image1.Height));
        cropW = Math.Max(1, Math.Min(cropW, image1.Width));

        // Search for most salient location (stride for efficiency): every candidate (y, x) on the stride
        // grid scores the saliency summed at (y + dy, x + dx) for stride-spaced offsets inside the crop
        // and the image; the first (row-major) best candidate wins
        var engine = Engine;
        int stride = Math.Max(1, Math.Min(cropH, cropW) / 4);
        var rowSampler = WindowSampler(engine, image2.Height, cropH, stride);
        var colSampler = WindowSampler(engine, image2.Width, cropW, stride);
        var scores = engine.TensorMatMul(engine.TensorMatMul(rowSampler, saliency), engine.TensorTranspose(colSampler));
        int candidatesX = colSampler.Shape[0];
        int best = engine.TensorArgMax(engine.Reshape(scores, [scores.Length]), 0).GetFlat(0);
        int bestY = best / candidatesX * stride;
        int bestX = best % candidatesX * stride;

        // Paste most salient region from image2 onto image1
        int pasteY = context.GetRandomInt(0, Math.Max(1, image1.Height - cropH));
        int pasteX = context.GetRandomInt(0, Math.Max(1, image1.Width - cropW));

        int copyH = Math.Min(cropH, Math.Min(image1.Height - pasteY, image2.Height - bestY));
        int copyW = Math.Min(cropW, Math.Min(image1.Width - pasteX, image2.Width - bestX));
        if (copyH > 0 && copyW > 0)
        {
            int channels = Math.Min(image1.Channels, image2.Channels);
            var patch = engine.TensorSlice(image2.ToPlanarTensor(), [0, bestY, bestX], [channels, copyH, copyW]);
            result.SetFromPlanarTensor(engine.TensorSetSlice(result.ToPlanarTensor(), patch, [0, pasteY, pasteX]));
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

    private static Tensor<T> ComputeSaliency(ImageTensor<T> image)
    {
        // Gradient magnitude summed over channels (central differences), 0 on the border
        return ImageTensor<T>.GradientMagnitude(image.ToPlanarTensor());
    }

    /// <summary>
    /// [candidates, size] matrix: candidate i starts at i * stride (for starts 0 .. size - window) and
    /// samples positions start + k * stride for k * stride &lt; window, clipped to the axis.
    /// </summary>
    private static Tensor<T> WindowSampler(IEngine engine, int size, int window, int stride)
    {
        int candidates = (size - window) / stride + 1;
        int[] shape = [candidates, size];
        var start = engine.TensorBroadcastTo(
            engine.Reshape(engine.TensorMultiplyScalar(engine.TensorArange<int>(0, candidates, 1), stride), [candidates, 1]), shape);
        var position = engine.TensorBroadcastTo(engine.Reshape(engine.TensorArange<int>(0, size, 1), [1, size]), shape);
        var offset = engine.TensorSubtract(position, start);
        var inWindow = engine.TensorMultiply(
            engine.ScalarMinusTensor(1, engine.TensorLessThan(offset, 0)), engine.TensorLessThan(offset, window));
        var onStride = engine.TensorEquals(
            engine.TensorSubtract(offset, engine.TensorMultiplyScalar(engine.TensorDivideScalar(offset, stride), stride)), 0);
        return engine.TensorMultiply(inWindow, onStride).Cast<T>();
    }

    /// <inheritdoc />
    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        return data.Clone();
    }

    /// <inheritdoc />
    public override IDictionary<string, object> GetParameters()
    {
        return base.GetParameters();
    }
}
