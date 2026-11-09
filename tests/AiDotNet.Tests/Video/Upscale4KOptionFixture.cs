using AiDotNet.Video.Enhancement;
using AiDotNet.Video.Options;

namespace AiDotNet.Tests.Video;

internal static class Upscale4KOptionFixture
{
    internal static void PrepareOptions(Upscale4KAgentOptions options)
    {
        // NIQE is unbounded above, so its 1 - NIQE/10 quality can be negative. Place the
        // cutoff at the measured quality of the existing pyiqa reference image; ordinary
        // numeric perturbations can then cross a real, scorable decision boundary.
        options.RollbackThreshold = 1 - Niqe.Score(Upscale4KAgentPaperTests.ReferenceImage(1, 200, 300), 200, 300) / 10;
    }

    internal static long Observe<T>(Upscale4KAgent<T> model)
    {
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        var image = Upscale4KAgentPaperTests.ReferenceImage(1, 200, 300);
        var clip = new Tensor<T>(new[] { 1, 3, 200, 300 });
        for (int i = 0; i < image.Length; i++) clip[i] = ops.FromDouble(image[i]);
        // A threshold affects the success decision recorded for a step. The release's
        // fully expanded SR-only plan keeps the same best descendant even after failure,
        // so waveform/pixel output alone cannot distinguish the reflection outcomes.
        return model.MeetsRollbackThreshold(model.QualityScore(clip)) ? 1 : 0;
    }
}
