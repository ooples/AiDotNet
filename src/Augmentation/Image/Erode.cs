namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Morphological erosion - shrinks bright regions using a structuring element.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class Erode<T> : ImageAugmenterBase<T>
{
    public int KernelSize { get; }
    public int Iterations { get; }

    public Erode(int kernelSize = 3, int iterations = 1,
        double probability = 0.5) : base(probability)
    {
        if (kernelSize < 1 || kernelSize % 2 == 0) throw new ArgumentOutOfRangeException(nameof(kernelSize));
        KernelSize = kernelSize; Iterations = iterations;
    }

    protected override ImageTensor<T> ApplyAugmentation(ImageTensor<T> data, AugmentationContext<T> context)
    {
        var result = data;
        int half = KernelSize / 2;

        for (int iter = 0; iter < Iterations; iter++)
        {
            var input = result.Clone();
            result = input.Clone();

            // Minimum over the edge-replicated neighbourhood
            var eroded = Engine.TensorAmin(ImageTensor<T>.Neighbourhoods(input.ToPlanarTensor(), 2 * half + 1), [3], false);
            result.SetFromPlanarTensor(eroded);
        }

        return result;
    }

    public override IDictionary<string, object> GetParameters()
    {
        var p = base.GetParameters();
        p["kernel_size"] = KernelSize; p["iterations"] = Iterations;
        return p;
    }
}
