using AiDotNet.Audio.VoiceActivity;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Audio;

/// <summary>
/// The frame loops must put the AUDIO into the tensor they score.
/// </summary>
/// <remarks>
/// <para>
/// <c>GetFrameProbabilities</c> and <c>DetectSpeechSegments</c> each built a per-frame tensor and then
/// assigned the samples into <c>frameTensor.ToVector()</c>. That method allocates and copies, so the
/// write landed in a throwaway vector and the tensor handed to the model stayed ALL ZEROS: every frame
/// was scored as silence and the result did not depend on the input at all.
/// </para>
/// <para>
/// <c>PreprocessAudio</c> in the same file already carries a comment describing this exact defect and
/// writes through <c>Data.Span</c> instead; these two loops were surviving instances of it. The
/// assertion here is the property a caller cares about — different audio must produce different frame
/// probabilities — which is false for any implementation that scores a zero-filled tensor.
/// </para>
/// </remarks>
public class SileroVadFramePayloadTests
{
    private const int FrameSize = 64;

    private static SileroVad<double> CreateVad() =>
        new SileroVad<double>(
            new NeuralNetworkArchitecture<double>(inputFeatures: FrameSize, outputSize: 1),
            sampleRate: 16000,
            frameSize: FrameSize,
            convFilters: 4,
            lstmHiddenDim: 4,
            numLstmLayers: 1);

    private static Tensor<double> Signal(int frames, double amplitude, int seed)
    {
        var t = new Tensor<double>([frames * FrameSize]);
        var span = t.Data.Span;
        var rng = new System.Random(seed);
        for (int i = 0; i < span.Length; i++)
            span[i] = amplitude * ((rng.NextDouble() * 2.0) - 1.0);
        return t;
    }

    [Fact]
    public void FrameProbabilities_DependOnTheAudio()
    {
        using var vad = CreateVad();

        var quiet = vad.GetFrameProbabilities(Signal(frames: 4, amplitude: 0.0, seed: 1));
        var loud = vad.GetFrameProbabilities(Signal(frames: 4, amplitude: 0.9, seed: 2));

        Assert.Equal(quiet.Length, loud.Length);
        Assert.True(quiet.Length > 0, "no frames were scored, so the assertion below would be vacuous");

        bool anyDifferent = false;
        for (int i = 0; i < quiet.Length; i++)
            if (quiet[i] != loud[i]) { anyDifferent = true; break; }

        Assert.True(anyDifferent,
            "every frame probability was identical for silence and for a loud signal, so the model is "
            + "not seeing the audio. This is the ToVector()-copy defect: the samples were written into a "
            + "copy and the tensor scored by the model stayed all zeros.");
    }

    [Fact]
    public void FrameProbabilities_AreDeterministicForTheSameAudio()
    {
        using var vad = CreateVad();

        var audio = Signal(frames: 3, amplitude: 0.7, seed: 11);
        var first = vad.GetFrameProbabilities(audio);
        var second = vad.GetFrameProbabilities(audio);

        Assert.Equal(first.Length, second.Length);
        for (int i = 0; i < first.Length; i++)
            Assert.True(first[i] == second[i],
                $"frame {i} scored {first[i]} then {second[i]} for identical audio; the per-frame tensor "
                + "is now disposed after each frame, and a disposed tensor must not be observable here.");
    }
}
