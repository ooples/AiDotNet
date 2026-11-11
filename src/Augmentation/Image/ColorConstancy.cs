namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Applies color constancy correction using the Gray World assumption or Max-RGB method.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class ColorConstancy<T> : ImageAugmenterBase<T>
{
    public ColorConstancyMethod Method { get; }
    public double BlendFactor { get; }

    public ColorConstancy(ColorConstancyMethod method = ColorConstancyMethod.GrayWorld,
        double blendFactor = 1.0, double probability = 0.5) : base(probability)
    {
        Method = method; BlendFactor = blendFactor;
    }

    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        if (data.Channels < 3) return data.Clone();

        var result = data.Clone();
        double maxVal = data.IsNormalized ? 1.0 : 255.0;
        var engine = Engine;
        var image = data.ToPlanarTensor();
        var rgb = engine.TensorSlice(image, [0, 0, 0], [3, data.Height, data.Width]);
        var ones = ImageTensor<T>.Constant(engine.ReduceMean(rgb, [1, 2], true), 1.0);
        Tensor<T> gainTensor;
        if (Method == ColorConstancyMethod.GrayWorld)
        {
            var channelMean = engine.ReduceMean(rgb, [1, 2], true);
            T overallMean = engine.TensorMean(channelMean);
            var valid = engine.TensorGreaterThan(channelMean, NumOps.FromDouble(1e-10));
            gainTensor = engine.TensorWhere(valid, engine.TensorDivide(engine.TensorMultiplyScalar(ones, overallMean), channelMean), ones);
        }
        else // MaxRGB
        {
            var channelMax = engine.TensorMax(engine.ReduceMax(rgb, [1, 2], true), NumOps.Zero);
            T overallMax = engine.TensorMaxValue(channelMax);
            var valid = engine.TensorGreaterThan(channelMax, NumOps.FromDouble(1e-10));
            gainTensor = engine.TensorWhere(valid, engine.TensorDivide(engine.TensorMultiplyScalar(ones, overallMax), channelMax), ones);
        }

        // Blend gains toward 1.0
        gainTensor = engine.TensorAddScalar(
            engine.TensorMultiplyScalar(engine.TensorSubtractScalar(gainTensor, NumOps.One), NumOps.FromDouble(BlendFactor)), NumOps.One);

        var balanced = engine.TensorClamp(engine.TensorMultiply(rgb, gainTensor), NumOps.Zero, NumOps.FromDouble(maxVal));
        result.SetFromPlanarTensor(ImageTensor<T>.ReplaceLeadingChannels(image,
            engine.TensorSliceAxis(balanced, 0, 0), engine.TensorSliceAxis(balanced, 0, 1), engine.TensorSliceAxis(balanced, 0, 2)));

        return result;
    }

    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters();
        p["method"] = Method.ToString(); p["blend_factor"] = BlendFactor;
        return p;
    }
}

/// <summary>
/// Color constancy correction method.
/// </summary>
public enum ColorConstancyMethod
{
    /// <summary>Gray World assumption - adjusts so channel means are equal.</summary>
    GrayWorld,
    /// <summary>Max-RGB - adjusts so channel maximums are equal.</summary>
    MaxRgb
}
