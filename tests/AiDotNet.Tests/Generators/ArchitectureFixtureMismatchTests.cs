using AiDotNet.Generators;
using Xunit;

namespace AiDotNet.Tests.Generators;

/// <summary>
/// Pins ADNTEST002's detection, and pins that it has NO family exemption.
/// </summary>
/// <remarks>
/// <para>
/// The rule was disabled for years behind a claim that most of its firings were not defects. They
/// were. Re-derived over the emitted fixtures the population was eleven models, every one of them
/// pinning an <c>InputType.TwoDimensional</c> architecture - whose <c>GetInputShape()</c> is a
/// rank-2 <c>[H, W]</c> - while its fixture fed a rank-3 mel spectrogram.
/// </para>
/// <para>
/// An exemption for those was written first, on the theory that <c>inputHeight</c>/<c>inputWidth</c>
/// mean spectrogram bins rather than image axes for an audio model, and it was wrong. A pinned
/// architecture drives <c>ResolveLazyLayerShapes</c>, so a lazy layer resolves its weights from the
/// declared shape while the forward runs on the fixture - the failure already recorded against the
/// sequence-labeling NER pin, where 128-vs-100 threw "Matrix dimensions incompatible". These tests
/// exist so that exemption cannot come back.
/// </para>
/// </remarks>
public class ArchitectureFixtureMismatchTests
{
    /// <summary>The SlimSAM defect the rule exists to catch: a 32x32 pin against a 128px fixture.</summary>
    private const string VisionMismatchFixture = @"
    protected override object Model => new SlimSAM<double>(new NeuralNetworkArchitecture<double>(
        inputType: InputType.ThreeDimensional, inputHeight: 32, inputWidth: 32, inputDepth: 3, outputSize: 4));
    protected override int[] InputShape => new[] { 3, 128, 128 };
";

    /// <summary>A vocoder before the fix: a rank-2 64x32 pin against an 80-bin mel fixture.</summary>
    private const string AudioMismatchFixture = @"
    protected override object Model => new HiFiGAN<double>(new NeuralNetworkArchitecture<double>(
        inputType: InputType.TwoDimensional, inputHeight: 64, inputWidth: 32, inputDepth: 1, outputSize: 4));
    protected override int[] InputShape => new[] { 1, 80, 1 };
";

    /// <summary>The same vocoder after the fix: the pin declares the mel geometry it is fed.</summary>
    private const string AudioFixedFixture = @"
    protected override object Model => new HiFiGAN<double>(new NeuralNetworkArchitecture<double>(
        inputType: InputType.ThreeDimensional, inputHeight: 80, inputWidth: 1, inputDepth: 1, outputSize: 4));
    protected override int[] InputShape => new[] { 1, 80, 1 };
";

    [Fact]
    public void VisionModel_WithDisagreeingPin_IsReported()
    {
        bool detected = ArchitectureFixtureMismatch.TryDetect(
            VisionMismatchFixture, out int ah, out int aw, out int fh, out int fw);

        Assert.True(detected);
        Assert.Equal(32, ah);
        Assert.Equal(32, aw);
        Assert.Equal(128, fh);
        Assert.Equal(128, fw);
    }

    [Fact]
    public void AudioModel_WithDisagreeingPin_IsAlsoReported()
    {
        // The exemption regression test. An audio fixture gets no free pass: this shape of
        // disagreement is the eleven-model defect the rule was disabled for missing.
        bool detected = ArchitectureFixtureMismatch.TryDetect(
            AudioMismatchFixture, out int ah, out int aw, out int fh, out int fw);

        Assert.True(detected);
        Assert.Equal(64, ah);
        Assert.Equal(32, aw);
        Assert.Equal(80, fh);
        Assert.Equal(1, fw);
    }

    [Fact]
    public void AudioModel_WhosePinDeclaresItsMelGeometry_IsSilent()
    {
        // The rule goes quiet because the generator was fixed, not because the rule stopped asking.
        Assert.False(ArchitectureFixtureMismatch.TryDetect(
            AudioFixedFixture, out _, out _, out _, out _));
    }

    [Fact]
    public void VisionModel_WhoseFixtureAgrees_IsNotReported()
    {
        const string agreeing = @"
    protected override object Model => new Probe<double>(new NeuralNetworkArchitecture<double>(
        inputHeight: 32, inputWidth: 32, inputDepth: 3, outputSize: 4));
    protected override int[] InputShape => new[] { 3, 32, 32 };
";
        Assert.False(ArchitectureFixtureMismatch.TryDetect(
            agreeing, out _, out _, out _, out _));
    }

    [Fact]
    public void FourElementFixture_IsNotComparedAgainstItsPrefix()
    {
        // A [B, C, H, W] fixture must not match on its first three entries: that compared [C, H]
        // against the pin and reported nonsense such as "AVID fixture [.., 3, 32]".
        const string batched = @"
    protected override object Model => new Probe<double>(new NeuralNetworkArchitecture<double>(
        inputHeight: 64, inputWidth: 64, inputDepth: 3, outputSize: 4));
    protected override int[] InputShape => new[] { 1, 3, 64, 64 };
";
        Assert.False(ArchitectureFixtureMismatch.TryDetect(
            batched, out _, out _, out _, out _));
    }

    [Fact]
    public void FixtureWithNoArchitecturePin_IsNotReported()
    {
        // A parameterless model has no inputHeight/inputWidth to compare against; there is nothing
        // to contradict, so silence here is correct rather than an exemption.
        const string noPin = @"
    protected override object Model => new Probe<double>();
    protected override int[] InputShape => new[] { 3, 32, 32 };
";
        Assert.False(ArchitectureFixtureMismatch.TryDetect(
            noPin, out _, out _, out _, out _));
    }
}
