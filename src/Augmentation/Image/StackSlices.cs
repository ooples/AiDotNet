namespace AiDotNet.Augmentation.Image;

/// <summary>
/// Stacks adjacent 2D slices to create multi-channel input (useful for 3D volume processing).
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class StackSlices<T> : ImageAugmenterBase<T>
{
    public int NumSlices { get; }

    public StackSlices(int numSlices = 3, double probability = 1.0) : base(probability)
    {
        if (numSlices < 1) throw new ArgumentOutOfRangeException(nameof(numSlices));
        NumSlices = numSlices;
    }

    /// <summary>
    /// Stacks multiple slices into a single multi-channel image.
    /// </summary>
    public ImageTensor<T> ApplyStackSlices(ImageTensor<T>[] slices)
    {
        if (slices.Length == 0) throw new ArgumentException("At least one slice required");

        int height = slices[0].Height;
        int width = slices[0].Width;
        int totalChannels = 0;
        for (int i = 0; i < slices.Length; i++)
            totalChannels += slices[i].Channels;

        var result = new ImageTensor<T>(height: height, width: width, channels: totalChannels);
        int channelOffset = 0;

        for (int s = 0; s < slices.Length; s++)
        {
            var slice = slices[s];
            if (slice.Height != height || slice.Width != width)
                slice = new Resize<T>(height, width).Apply(slice, new AugmentationContext<T>());

            for (int c = 0; c < slice.Channels; c++)
                for (int y = 0; y < height; y++)
                    for (int x = 0; x < width; x++)
                        result.SetPixel(y, x, channelOffset + c, slice.GetPixel(y, x, c));

            channelOffset += slice.Channels;
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
        p["num_slices"] = NumSlices;
        return p;
    }
}
