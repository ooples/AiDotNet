using AiDotNet.Augmentation;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Augmentation.Image;

/// <summary>
/// TokenMix - token-level mixing for transformer architectures.
/// Randomly selects and replaces individual patches (tokens).
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class TokenMix<T> : ImageMixingAugmenterBase<T>
{
    public int PatchSize { get; }
    public double MinMixRatio { get; }
    public double MaxMixRatio { get; }

    public TokenMix(int patchSize = 16, double minMixRatio = 0.2, double maxMixRatio = 0.8,
        double probability = 0.5) : base(probability)
    {
        if (patchSize < 1) throw new ArgumentOutOfRangeException(nameof(patchSize), "PatchSize must be at least 1.");
        if (minMixRatio < 0) throw new ArgumentOutOfRangeException(nameof(minMixRatio), "MinMixRatio must be non-negative.");
        if (maxMixRatio < minMixRatio) throw new ArgumentOutOfRangeException(nameof(maxMixRatio), "MaxMixRatio must be >= MinMixRatio.");
        if (maxMixRatio > 1.0) throw new ArgumentOutOfRangeException(nameof(maxMixRatio), "MaxMixRatio must be <= 1.0.");
        PatchSize = patchSize; MinMixRatio = minMixRatio; MaxMixRatio = maxMixRatio;
    }

    /// <summary>
    /// Mixes two images at the token (patch) level.
    /// </summary>
    public ImageTensor<T> ApplyTokenMix(ImageTensor<T> image1, ImageTensor<T> image2,
        Vector<T>? labels1, Vector<T>? labels2, AugmentationContext<T> context)
    {
        if (image1.Height != image2.Height || image1.Width != image2.Width)
            image2 = new Resize<T>(image1.Height, image1.Width).Apply(image2, context);

        var result = image1.Clone();
        double mixRatio = context.GetRandomDouble(MinMixRatio, MaxMixRatio);

        int numPatchesH = Math.Max(1, image1.Height / PatchSize);
        int numPatchesW = Math.Max(1, image1.Width / PatchSize);
        int totalPatches = numPatchesH * numPatchesW;
        int patchesToReplace = (int)(totalPatches * mixRatio);

        // Randomly select patches to replace
        var patchIndices = new int[totalPatches];
        for (int i = 0; i < totalPatches; i++) patchIndices[i] = i;

        // Fisher-Yates shuffle
        for (int i = totalPatches - 1; i > 0; i--)
        {
            int j = context.GetRandomInt(0, i + 1);
            (patchIndices[i], patchIndices[j]) = (patchIndices[j], patchIndices[i]);
        }

        for (int i = 0; i < patchesToReplace; i++)
        {
            int patchIdx = patchIndices[i];
            int py = patchIdx / numPatchesW;
            int px = patchIdx % numPatchesW;

            int startY = py * PatchSize;
            int startX = px * PatchSize;
            int endY = Math.Min(startY + PatchSize, image1.Height);
            int endX = Math.Min(startX + PatchSize, image1.Width);

            for (int y = startY; y < endY; y++)
                for (int x = startX; x < endX; x++)
                    for (int c = 0; c < Math.Min(image1.Channels, image2.Channels); c++)
                        result.SetPixel(y, x, c, image2.GetPixel(y, x, c));
        }

        // Lambda = proportion of image1 remaining
        double lambda = 1.0 - mixRatio;
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
        var p = base.GetParameters();
        p["patch_size"] = PatchSize;
        p["min_mix_ratio"] = MinMixRatio; p["max_mix_ratio"] = MaxMixRatio;
        return p;
    }
}
