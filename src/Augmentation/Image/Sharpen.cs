namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Applies a sharpening filter using a 3x3 kernel.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class Sharpen<T> : ImageAugmenterBase<T>
{
    public double MinAlpha { get; }
    public double MaxAlpha { get; }
    public double MinLightness { get; }
    public double MaxLightness { get; }

    public Sharpen(double minAlpha = 0.2, double maxAlpha = 0.5,
        double minLightness = 0.5, double maxLightness = 1.0,
        double probability = 0.5) : base(probability)
    {
        MinAlpha = minAlpha; MaxAlpha = maxAlpha;
        MinLightness = minLightness; MaxLightness = maxLightness;
    }

    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        var result = data.Clone();
        double maxVal = data.IsNormalized ? 1.0 : 255.0;
        double alpha = context.GetRandomDouble(MinAlpha, MaxAlpha);
        double lightness = context.GetRandomDouble(MinLightness, MaxLightness);

        // Sharpening kernel blended with identity
        // identity = [0,0,0; 0,1,0; 0,0,0]
        // sharpen  = [-1,-1,-1; -1,8+L,-1; -1,-1,-1] / (L)
        // Effective kernel = (1-alpha)*identity + alpha*sharpen

        for (int c = 0; c < data.Channels; c++)
        {
            for (int y = 0; y < data.Height; y++)
                for (int x = 0; x < data.Width; x++)
                {
                    double original = NumOps.ToDouble(data.GetPixel(y, x, c));

                    // Compute Laplacian
                    double laplacian = 0;
                    for (int ky = -1; ky <= 1; ky++)
                        for (int kx = -1; kx <= 1; kx++)
                        {
                            int ny = Math.Max(0, Math.Min(data.Height - 1, y + ky));
                            int nx = Math.Max(0, Math.Min(data.Width - 1, x + kx));
                            double neighbor = NumOps.ToDouble(data.GetPixel(ny, nx, c));
                            if (ky == 0 && kx == 0)
                                laplacian += neighbor * (8 + lightness);
                            else
                                laplacian -= neighbor;
                        }
                    laplacian /= (8 + lightness);

                    double sharpened = original * (1 - alpha) + laplacian * alpha;
                    result.SetPixel(y, x, c, NumOps.FromDouble(Math.Max(0, Math.Min(maxVal, sharpened))));
                }
        }

        return result;
    }

    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters();
        p["min_alpha"] = MinAlpha; p["max_alpha"] = MaxAlpha;
        p["min_lightness"] = MinLightness; p["max_lightness"] = MaxLightness;
        return p;
    }
}
