namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Simulates K-space motion artifacts in MRI by introducing phase shifts.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class KSpaceMotion<T> : ImageAugmenterBase<T>
{
    public double MinDisplacement { get; }
    public double MaxDisplacement { get; }
    public int NumTransforms { get; }

    public KSpaceMotion(double minDisplacement = 1.0, double maxDisplacement = 5.0,
        int numTransforms = 3, double probability = 0.5) : base(probability)
    {
        MinDisplacement = minDisplacement; MaxDisplacement = maxDisplacement;
        NumTransforms = numTransforms;
    }

    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        var result = data.Clone();
        double maxVal = data.IsNormalized ? 1.0 : 255.0;

        // Simulate motion by applying random shifts to horizontal lines
        // (approximation of k-space corruption)
        int linesPerTransform = data.Height / (NumTransforms + 1);

        for (int t = 0; t < NumTransforms; t++)
        {
            double displacement = context.GetRandomDouble(MinDisplacement, MaxDisplacement);
            if (context.GetRandomBool()) displacement = -displacement;
            int shift = (int)Math.Round(displacement);

            int startLine = context.GetRandomInt(0, data.Height - linesPerTransform);
            int endLine = Math.Min(startLine + linesPerTransform, data.Height);

            for (int y = startLine; y < endLine; y++)
                for (int x = 0; x < data.Width; x++)
                {
                    int srcX = x - shift;
                    if (srcX < 0 || srcX >= data.Width) continue;

                    for (int c = 0; c < data.Channels; c++)
                        result.SetPixel(y, x, c, data.GetPixel(y, srcX, c));
                }
        }

        return result;
    }

    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters();
        p["min_displacement"] = MinDisplacement; p["max_displacement"] = MaxDisplacement;
        p["num_transforms"] = NumTransforms;
        return p;
    }
}
