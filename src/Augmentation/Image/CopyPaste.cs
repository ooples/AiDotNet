namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Copy-Paste augmentation (Ghiasi et al., 2020) - copies random regions from a source
/// image and pastes them onto the target image.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class CopyPaste<T> : ImageAugmenterBase<T>
{
    public int MinPatches { get; }
    public int MaxPatches { get; }
    public double MinPatchScale { get; }
    public double MaxPatchScale { get; }

    public CopyPaste(int minPatches = 1, int maxPatches = 3,
        double minPatchScale = 0.05, double maxPatchScale = 0.25,
        double probability = 0.5) : base(probability)
    {
        MinPatches = minPatches; MaxPatches = maxPatches;
        MinPatchScale = minPatchScale; MaxPatchScale = maxPatchScale;
    }

    /// <summary>
    /// Copies patches from source image and pastes onto target.
    /// </summary>
    public ImageTensor<T> ApplyCopyPaste(ImageTensor<T> target, ImageTensor<T> source,
        AugmentationContext<T> context)
    {
        var result = target.Clone();
        int numPatches = context.GetRandomInt(MinPatches, MaxPatches + 1);

        for (int p = 0; p < numPatches; p++)
        {
            double scale = context.GetRandomDouble(MinPatchScale, MaxPatchScale);
            int patchH = Math.Max(1, (int)(source.Height * scale));
            int patchW = Math.Max(1, (int)(source.Width * scale));

            // Source region
            int srcY = context.GetRandomInt(0, Math.Max(1, source.Height - patchH));
            int srcX = context.GetRandomInt(0, Math.Max(1, source.Width - patchW));

            // Target position
            int dstY = context.GetRandomInt(0, Math.Max(1, target.Height - patchH));
            int dstX = context.GetRandomInt(0, Math.Max(1, target.Width - patchW));

            for (int y = 0; y < patchH; y++)
                for (int x = 0; x < patchW; x++)
                {
                    if (srcY + y >= source.Height || srcX + x >= source.Width) continue;
                    if (dstY + y >= target.Height || dstX + x >= target.Width) continue;
                    for (int c = 0; c < Math.Min(target.Channels, source.Channels); c++)
                        result.SetPixel(dstY + y, dstX + x, c,
                            source.GetPixel(srcY + y, srcX + x, c));
                }
        }

        return result;
    }

    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        // Self copy-paste as fallback
        return ApplyCopyPaste(data, data, context);
    }

    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters();
        p["min_patches"] = MinPatches; p["max_patches"] = MaxPatches;
        p["min_patch_scale"] = MinPatchScale; p["max_patch_scale"] = MaxPatchScale;
        return p;
    }
}
