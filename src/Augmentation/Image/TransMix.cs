using AiDotNet.Augmentation;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Augmentation.Image;

/// <summary>
/// TransMix (Chen et al., 2022) - attention-guided mixing for Vision Transformers.
/// Creates patch-level mixing masks based on attention-like scores.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class TransMix<T> : ImageMixingAugmenterBase<T>
{
    public int PatchSize { get; }

    public TransMix(double alpha = 1.0, int patchSize = 16,
        double probability = 0.5) : base(probability, alpha)
    {
        if (patchSize < 1) throw new ArgumentOutOfRangeException(nameof(patchSize), "PatchSize must be at least 1.");
        PatchSize = patchSize;
    }

    /// <summary>
    /// Mixes two images using patch-level attention-guided masks.
    /// </summary>
    public ImageTensor<T> ApplyTransMix(ImageTensor<T> image1, ImageTensor<T> image2,
        Vector<T>? labels1, Vector<T>? labels2, AugmentationContext<T> context)
    {
        if (image1.Height != image2.Height || image1.Width != image2.Width)
            image2 = new Resize<T>(image1.Height, image1.Width).Apply(image2, context);

        var result = image1.Clone();
        double lambda = SampleLambda(context);
        LastMixingLambda = NumOps.FromDouble(lambda);

        int numPatchesH = Math.Max(1, image1.Height / PatchSize);
        int numPatchesW = Math.Max(1, image1.Width / PatchSize);
        int totalPatches = numPatchesH * numPatchesW;

        // Generate attention-like scores for each patch
        var scores = new double[totalPatches];
        for (int i = 0; i < totalPatches; i++)
            scores[i] = context.GetRandomDouble(0, 1);

        // Select patches to mix based on lambda
        int patchesToMix = (int)(totalPatches * (1 - lambda));
        var indices = new int[totalPatches];
        for (int i = 0; i < totalPatches; i++) indices[i] = i;

        // Sort ascending by score; lowest-scored patches are selected for mixing
        Array.Sort(scores, indices);

        var mixMask = new bool[totalPatches];
        for (int i = 0; i < patchesToMix && i < totalPatches; i++)
            mixMask[indices[i]] = true;

        // Apply mixing
        for (int py = 0; py < numPatchesH; py++)
            for (int px = 0; px < numPatchesW; px++)
            {
                int patchIdx = py * numPatchesW + px;
                if (!mixMask[patchIdx]) continue;

                int startY = py * PatchSize;
                int startX = px * PatchSize;
                int endY = Math.Min(startY + PatchSize, image1.Height);
                int endX = Math.Min(startX + PatchSize, image1.Width);

                for (int y = startY; y < endY; y++)
                    for (int x = startX; x < endX; x++)
                        for (int c = 0; c < Math.Min(image1.Channels, image2.Channels); c++)
                            result.SetPixel(y, x, c, image2.GetPixel(y, x, c));
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

    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        return data.Clone();
    }

    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters();
        p["patch_size"] = PatchSize;
        return p;
    }
}
