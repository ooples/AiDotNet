using AiDotNet.VisionLanguage.Grounding;

namespace AiDotNet.Tests.VisionLanguage;

internal static class GroundedSam2OptionFixture
{
    internal static long Observe<T>(GroundedSAM2<T> model, Tensor<T> input)
    {
        var image = new Tensor<T>(input.Shape.ToArray());
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        for (int i = 0; i < image.Length; i++) image[i] = ops.FromDouble(0.4 + 0.2 * Math.Sin(i * 0.17));
        var segments = model.Segment(image, "cat .");
        if (segments.Count == 0) throw new InvalidOperationException("The SAM decoder probe requires a detector box.");
        return ModelFamilyTests.Base.OptionSensitivityHarness.Fingerprint(segments.SelectMany(segment =>
            new[] { segment.MaskScore }.Concat(segment.Mask.Select(ops.ToDouble))));
    }
}
