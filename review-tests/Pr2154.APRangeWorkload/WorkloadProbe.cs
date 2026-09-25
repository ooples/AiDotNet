global using AiDotNet.Tensors.Interfaces;
global using AiDotNet.Tensors.Helpers;

using AiDotNet.Augmentation.Image;

// Only compiled into the isolated counter executable. The timed production assembly has no
// counter, callback, subclassed geometry, or numeric-provider mutation.
internal static class WorkloadProbe
{
    internal static long IoUCalls { get; set; }

    internal static double CountIoU<T>(BoundingBox<T> prediction, BoundingBox<T> candidate) where T : struct
    {
        IoUCalls++;
        return prediction.IoU(candidate);
    }
}
