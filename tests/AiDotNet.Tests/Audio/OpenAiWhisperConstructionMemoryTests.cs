using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Audio.Whisper;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;
using Xunit;

namespace AiDotNet.Tests.Audio;

/// <summary>
/// Constructing the openai-whisper network must cost about its weights, not a second copy of them.
/// </summary>
/// <remarks>
/// <para>
/// ModelInputMeetsFirstLayerTests' slice 4 host was killed at 12 GB while building WhisperSpeech. That model contains a
/// whisper-medium network (764M parameters), and InitializeLikeReference returned a fresh <c>double[]</c> for every
/// tensor. The token table went through a <c>Vector</c> and then <c>ToArray</c>, so it was copied twice. Every
/// construction therefore allocated 8 to 16 bytes of large-object garbage per parameter on top of the weights. Measured
/// with float weights: 3.6 GB live, 11.1 GB private bytes, and the private bytes fell back to 4.0 GB after an
/// aggressive GC. None of that memory was rooted, but the collector kept the freed regions committed, so the sweep's
/// non-aggressive between-model collections never gave it back.
/// </para>
/// <para>
/// The initializer now streams its values into the layers through <c>TorchTensorSource.Generated</c>, the same way
/// LlamaNetwork does. These tests pin the three properties that change has to keep: the drawn values are identical,
/// construction allocates about its weights, and repeated build-and-drop retains nothing.
/// </para>
/// </remarks>
public class OpenAiWhisperConstructionMemoryTests
{
    private static WhisperDimensions Dims(int width, int layers, int vocabulary) =>
        new(Mels: 8, AudioContext: 6, AudioState: width, AudioHeads: 2, AudioLayers: layers, Vocabulary: vocabulary,
            TextContext: 5, TextState: width, TextHeads: 2, TextLayers: layers);

    /// <summary>The allocating initializer this replaced, kept verbatim as the reference for the values it drew.</summary>
    private static Func<string, int[], double[]> AllocatingReference(WhisperDimensions d, Random random) => (name, shape) =>
    {
        int count = shape.Aggregate(1, (a, s) => a * s);
        var values = new double[count];
        if (name == "encoder.positional_embedding") return OpenAiWhisperNetwork<double>.Sinusoids(d.AudioContext, d.AudioState);
        if (name.EndsWith("ln.weight", StringComparison.Ordinal) || name.EndsWith("ln_post.weight", StringComparison.Ordinal))
            return Enumerable.Repeat(1.0, count).ToArray();
        if (name.EndsWith("ln.bias", StringComparison.Ordinal) || name.EndsWith("ln_post.bias", StringComparison.Ordinal) || name == "decoder.positional_embedding")
            return values;
        if (name == "decoder.token_embedding.weight")
            return AiDotNet.TextToSpeech.CodecBased.TorchParameters.Gaussian<double>(random, count, 1.0).ToArray();
        int fanIn = shape.Length >= 2 ? shape.Skip(1).Aggregate(1, (a, s) => a * s) : 0;
        if (shape.Length == 1)
        {
            fanIn = name.StartsWith("encoder.conv1", StringComparison.Ordinal) ? d.Mels * 3
                : name.StartsWith("encoder.conv2", StringComparison.Ordinal) ? d.AudioState * 3
                : name.Contains(".mlp.2.") ? 4 * (name.StartsWith("encoder", StringComparison.Ordinal) ? d.AudioState : d.TextState)
                : name.StartsWith("encoder", StringComparison.Ordinal) ? d.AudioState : d.TextState;
        }
        double bound = 1.0 / Math.Sqrt(fanIn);
        for (int i = 0; i < count; i++) values[i] = bound * (2 * random.NextDouble() - 1);
        return values;
    };

    [Fact]
    public void StreamedInitialization_DrawsExactlyTheValuesTheAllocatingFormDrew()
    {
        // An odd vocabulary times an odd width gives the token table an odd count, which covers the Box-Muller pair
        // that is cut short at the end of the tensor.
        var d = Dims(width: 15, layers: 2, vocabulary: 37) with { AudioHeads = 3, TextHeads = 3 };
        var engine = AiDotNetEngine.Current;
        // Both networks are built alike and then re-initialized from the same seed, one through each form, so
        // anything a load leaves untouched is the same on both sides.
        var streamedLayers = new List<LayerBase<double>>();
        var streamed = new OpenAiWhisperNetwork<double>(engine, streamedLayers, d, new Random(99));
        streamed.InitializeLikeReference(new Random(7));

        var referenceLayers = new List<LayerBase<double>>();
        var reference = new OpenAiWhisperNetwork<double>(engine, referenceLayers, d, new Random(99));
        reference.LoadTorchWeights(AllocatingReference(d, new Random(7)));

        Assert.Equal(referenceLayers.Count, streamedLayers.Count);
        int compared = 0;
        for (int i = 0; i < streamedLayers.Count; i++)
        {
            // The loaded tensors themselves: a linear holder's weights and a convolution's kernel.
            double[] expected, actual;
            if (streamedLayers[i] is BiasFreeLinearLayer<double> s && referenceLayers[i] is BiasFreeLinearLayer<double> r)
            {
                actual = s.Weights.AsSpan().ToArray();
                expected = r.Weights.AsSpan().ToArray();
            }
            else if (streamedLayers[i] is NormedConv1DLayer<double> sc && referenceLayers[i] is NormedConv1DLayer<double> rc)
            {
                actual = sc.Kernel().AsSpan().ToArray();
                expected = rc.Kernel().AsSpan().ToArray();
            }
            else continue;
            compared++;
            Assert.Equal(expected.Length, actual.Length);
            for (int j = 0; j < expected.Length; j++)
                Assert.True(expected[j].Equals(actual[j]),
                    $"layer {i} ({streamedLayers[i].GetType().Name}) value {j} of {actual.Length}: {actual[j]} != {expected[j]}");
        }
        Assert.Equal(streamedLayers.Count, compared);
    }

#if NET8_0_OR_GREATER
    [Fact]
    public void Construction_AllocatesAboutItsWeights_NotASecondCopy()
    {
        // 13.3M of the parameters are the 51,865-token table, the tensor that used to be copied twice.
        var d = Dims(width: 256, layers: 4, vocabulary: 51865);
        var engine = AiDotNetEngine.Current;
        _ = new OpenAiWhisperNetwork<float>(engine, new List<LayerBase<float>>(), Dims(16, 1, 10), new Random(0)); // JIT and statics

        long before = GC.GetTotalAllocatedBytes(precise: true);
        var layers = new List<LayerBase<float>>();
        var network = new OpenAiWhisperNetwork<float>(engine, layers, d, new Random(0));
        long allocated = GC.GetTotalAllocatedBytes(precise: true) - before;
        GC.KeepAlive(network);

        long weightBytes = layers.Sum(l => (long)l.ParameterCount) * sizeof(float);
        // The allocating initializer cost 3x the float weights (12 bytes a parameter, 20 for the token table). The
        // streamed one costs the weights plus small per-layer bookkeeping.
        Assert.True(allocated < weightBytes * 3 / 2,
            $"constructing a {weightBytes / 1e6:F0} MB whisper network allocated {allocated / 1e6:F0} MB "
            + $"({(double)allocated / weightBytes:F2}x its weights); initialization is materializing whole tensors again.");
    }
#endif

    [Fact]
    public void RepeatedConstruction_RetainsNothing()
    {
        var d = Dims(width: 256, layers: 4, vocabulary: 51865);
        var engine = AiDotNetEngine.Current;

        long Build()
        {
            var layers = new List<LayerBase<float>>();
            _ = new OpenAiWhisperNetwork<float>(engine, layers, d, new Random(0));
            long bytes = layers.Sum(l => (long)l.ParameterCount) * sizeof(float);
            foreach (var layer in layers) layer.Dispose();
            return bytes;
        }

        long weightBytes = Build();
        long baseline = GC.GetTotalMemory(forceFullCollection: true);
        const int Rounds = 5;
        for (int i = 0; i < Rounds; i++) Build();
        long retained = GC.GetTotalMemory(forceFullCollection: true) - baseline;

        // A leak of one model per round would retain 5x the weights. Allow a tenth of one model for noise.
        Assert.True(retained < weightBytes / 10,
            $"{Rounds} build-and-dispose rounds of a {weightBytes / 1e6:F0} MB network retained {retained / 1e6:F0} MB.");
    }
}
