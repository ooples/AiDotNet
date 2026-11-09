using AiDotNet.Video.Enhancement;
using AiDotNet.Video.Options;

namespace AiDotNet.Tests.Video;

internal static class MgldVsrOptionFixture
{
    internal static bool Handles(string option) => option == "MinimumSide" || IsDecoderOption(option);

    private static bool IsDecoderOption(string option) => option is "TrainingFusionWeight" or "DecoderLearningRate"
        or "DifferenceWeight" or "DiscriminatorWeight" or "EdgeWeight" or "DiscriminatorStart";

    internal static void Prepare(MGLDVSROptions options, string option)
    {
        if (!Handles(option)) return;
        // Keep frozen sampling stable; decoder warping still uses the real SPyNet path.
        options.GuidanceScale = 0;
        options.FlowIterations = 1;
        if (option == "MinimumSide")
        {
            options.MinimumSide = 32;
        }
        if (!IsDecoderOption(option)) return;
        // These settings act in decoder fine-tuning, after the frozen diffusion stage.
        options.DiffusionIterations = 0;
        if (option is "DiscriminatorWeight" or "DiscriminatorStart") options.DiscriminatorStart = 0;
    }

    internal static long Observe<T>(MGLDVSR<T> model, string option, Func<long> observeOutput)
    {
        if (option == "MinimumSide") return observeOutput();
        var numeric = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        // Observe the trained components directly so output clamping cannot hide their updates.
        return TtsPaperNumerics.Combine(model.DecoderParameters.Concat(model.DiscriminatorParameters)
            .Select(tensor => TtsPaperNumerics.Fingerprint(tensor.Select(numeric.ToDouble))).ToArray());
    }
}
