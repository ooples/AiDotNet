using AiDotNet.NeuralNetworks;
using AiDotNet.SpeechRecognition.Foundation;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

/// <summary>
/// Manual test scaffold for Data2VecASR (Baevski et al. 2022, "data2vec: A General
/// Framework for Self-Supervised Learning in Speech, Vision and Language",
/// arXiv:2202.03555) with a CTC recognition head. The auto-generator is told to
/// skip Data2VecASR (<c>ExcludedClassNames</c>) so this hand-written scaffold is
/// authoritative.
/// </summary>
/// <remarks>
/// <b>Why a reduced-scale config:</b> Data2VecASR's production defaults are a
/// wav2vec2/data2vec-base-scale encoder (768-dim × 12 transformer layers × 12
/// heads × 3072-FFN). One forward+backward+optimizer step at that scale, times the
/// MoreData invariant's iterations, exceeds the 120s CI budget on CPU (the test
/// timed out). These invariants validate the <i>architecture's code paths</i>
/// (feature projection, transformer encoder stack, CTC head, deterministic
/// inference, backprop, optimizer step, clone) — not paper-scale numerical
/// behaviour. A 2-layer / 64-dim config exercises every path in seconds while
/// keeping the architecture faithful.
/// </remarks>
public class Data2VecASRTests : AudioNNModelTestBase
{
    protected override int[] InputShape => [1, 16, 32];

    protected override AiDotNet.Interfaces.INeuralNetworkModel<double> CreateNetwork()
    {
        var options = new Data2VecASROptions
        {
            EncoderDim = 64,
            NumEncoderLayers = 2,
            NumAttentionHeads = 4,
            NumMels = 32,
            VocabSize = 64,
            DropoutRate = 0.0,
        };
        var architecture = new NeuralNetworkArchitecture<double>(
            inputFeatures: options.NumMels, outputSize: options.VocabSize);
        return new Data2VecASR<double>(architecture, options);
    }
}
