using AiDotNet.Interfaces;
using AiDotNet.Diffusion.Video.VideoEditing;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

// Foundation-scale video-editing diffusion. Every test fits the 120s [Fact(Timeout)] in
// isolation (verified — heaviest is Predict ~24s, the 10-step training probe ~27s), but the
// forward is slow enough that it trips the wall-clock envelope under parallel contention
// (#1305). Serialize it onto dedicated cores like the other foundation-scale diffusion models
// (#1622 L4) so it stays green in the default gate rather than excluding it to HeavyTimeout.
[Xunit.Collection("FoundationScaleSerial")]
public class FlowVidModelTests : DiffusionModelTestBase<float>
{
    protected override int[] InputShape => [1, 4, 16, 16];
    protected override int[] OutputShape => [1, 4, 16, 16];

    protected override IDiffusionModel<float> CreateModel()
        => new FlowVidModel<float>(seed: 42);
}
