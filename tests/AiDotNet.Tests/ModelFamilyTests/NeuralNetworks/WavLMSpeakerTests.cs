using AiDotNet.Audio.Speaker;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

/// <summary>
/// Manual test scaffold for WavLMSpeaker (Chen et al. 2022, "WavLM: Large-Scale
/// Self-Supervised Pre-Training for Full Stack Speech Processing",
/// arXiv:2110.13900) speaker-embedding head. The auto-generator is told to skip
/// WavLMSpeaker (<c>ExcludedClassNames</c>) so this hand-written scaffold is
/// authoritative.
/// </summary>
/// <remarks>
/// <b>Why a reduced-scale config:</b> WavLMSpeaker's production defaults are a
/// 12-layer / 768-dim / 3072-FFN transformer encoder — a WavLM-base-scale model
/// whose forward+backward+optimizer step, times the training invariants'
/// iterations, exceeds the 120/180s CI budget on CPU. These invariants validate
/// the <i>architecture's code paths</i> (feature encoder, transformer stack,
/// attentive-stat pooling, embedding projection, deterministic inference,
/// backprop, optimizer step, clone) — not paper-scale numerical behaviour. A
/// 2-layer / 64-dim config exercises every path in seconds while keeping the
/// architecture faithful.
/// </remarks>
public class WavLMSpeakerTests : SpeakerRecognitionTestBase
{
    protected override int[] InputShape => [1, 64, 32];

    protected override INeuralNetworkModel<double> CreateNetwork()
    {
        var options = new WavLMSpeakerOptions
        {
            HiddenDim = 64,
            NumLayers = 2,
            NumAttentionHeads = 4,
            FeedForwardDim = 128,
            FeatureEncoderDim = 64,
            EmbeddingDim = 64,
            DropoutRate = 0.0,
        };
        var architecture = new NeuralNetworkArchitecture<double>(
            inputFeatures: options.NumMels, outputSize: options.EmbeddingDim);
        return new WavLMSpeaker<double>(architecture, options);
    }
}
