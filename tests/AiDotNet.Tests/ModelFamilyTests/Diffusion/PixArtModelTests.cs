using AiDotNet.Interfaces;
using AiDotNet.Diffusion.TextToImage;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.Diffusion;

// Full-scale PixArt-α (600M params) in float (per the research-paper default / PR #1605). At full
// scale a forward is ~32-49s; with float + the O(1) copy-on-write Clone this fits the 120s envelope
// without shrinking the model.
public class PixArtModelTests : DiffusionModelTestBase<float>
{
    protected override int[] InputShape => [1, 4, 32, 32];
    protected override int[] OutputShape => [1, 4, 32, 32];

    protected override IDiffusionModel<float> CreateModel()
        => new PixArtModel<float>(seed: 42);
}
