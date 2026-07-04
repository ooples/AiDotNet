using AiDotNet.Interfaces;
using AiDotNet.Diffusion.Video.VideoEditing;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

// Foundation-scale-at-default: the model's full-scale default config has a Training peak (weights +
// gradients + Adam state + activations) that OOMs the 16 GB CI runner (fits only on a larger box).
// Moved to the HeavyTimeout nightly lane so the default PR-gate shard fits and passes (#1706/#1305).
[Xunit.Trait("Category", "HeavyTimeout")]
public class InstructVid2VidModelTests : DiffusionModelTestBase<float>
{
    protected override int[] InputShape => [1, 4, 16, 16];
    protected override int[] OutputShape => [1, 4, 16, 16];

    protected override IDiffusionModel<float> CreateModel()
        => new InstructVid2VidModel<float>(seed: 42);
}
