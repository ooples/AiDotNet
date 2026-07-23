namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Applies a random tone curve transformation to adjust image tonality.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class RandomToneCurve<T> : ImageAugmenterBase<T>
{
    public double Scale { get; }
    public int NumControlPoints { get; }

    public RandomToneCurve(double scale = 0.1, int numControlPoints = 4,
        double probability = 0.5) : base(probability)
    {
        if (numControlPoints < 0) throw new ArgumentOutOfRangeException(nameof(numControlPoints), "Must be non-negative.");
        Scale = scale; NumControlPoints = numControlPoints;
    }

    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        var result = data.Clone();
        double maxVal = data.IsNormalized ? 1.0 : 255.0;

        // Generate random control points for the tone curve
        var controlX = new double[NumControlPoints + 2];
        var controlY = new double[NumControlPoints + 2];
        controlX[0] = 0; controlY[0] = 0;
        controlX[NumControlPoints + 1] = 1; controlY[NumControlPoints + 1] = 1;

        for (int i = 1; i <= NumControlPoints; i++)
        {
            controlX[i] = (double)i / (NumControlPoints + 1);
            controlY[i] = controlX[i] + context.GetRandomDouble(-Scale, Scale);
            controlY[i] = Math.Max(0, Math.Min(1, controlY[i]));
        }

        // Build lookup using linear interpolation between control points
        for (int y = 0; y < data.Height; y++)
            for (int x = 0; x < data.Width; x++)
                for (int c = 0; c < data.Channels; c++)
                {
                    double val = NumOps.ToDouble(data.GetPixel(y, x, c));
                    double normalized = val / maxVal;

                    // Find segment
                    int seg = 0;
                    for (int i = 1; i < controlX.Length; i++)
                    {
                        if (normalized <= controlX[i]) { seg = i - 1; break; }
                        seg = i - 1;
                    }

                    double t = (controlX[seg + 1] - controlX[seg]) > 1e-10
                        ? (normalized - controlX[seg]) / (controlX[seg + 1] - controlX[seg])
                        : 0;
                    double mapped = controlY[seg] + t * (controlY[seg + 1] - controlY[seg]);

                    result.SetPixel(y, x, c,
                        NumOps.FromDouble(Math.Max(0, Math.Min(maxVal, mapped * maxVal))));
                }

        return result;
    }

    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters();
        p["scale"] = Scale; p["num_control_points"] = NumControlPoints;
        return p;
    }
}
