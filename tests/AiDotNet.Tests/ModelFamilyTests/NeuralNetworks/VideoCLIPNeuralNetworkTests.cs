using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks;
using AiDotNet.Optimizers;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

/// <summary>
/// Tests for VideoCLIPNeuralNetwork per Xu et al. (2021)
/// "VideoCLIP: Contrastive Pre-training for Zero-shot Video-Text Understanding" (EMNLP 2021).
///
/// Paper architecture (Table 1): ViT-B/16 vision encoder, 12-layer temporal transformer,
/// 12-layer text encoder, 512-dim joint embedding space, 8 video frames at 1 FPS,
/// 77 max text tokens, 49408 BPE vocab (CLIP tokenizer).
///
/// Test uses scaled-down dims while preserving architectural ratios:
///   - imageSize=32, patchSize=16 → 4 patches (paper: 224/16=196 patches)
///   - embeddingDim=64 (paper: 512) — maintains power-of-2 dim for attention heads
///   - visionHiddenDim=128 (paper: 768) — 2x embedding per paper's ViT-B design
///   - textHiddenDim=64 (paper: 512) — matches embedding dim per CLIP text encoder
///   - numHeads=4 (paper: 12) — preserves multi-head structure, head_dim=16
///   - 2 encoder layers each (paper: 12) — minimal depth to test gradient flow
///   - 4 frames (paper: 8) — temporal aggregation test
///   - LR=1e-4 (paper §4: AdamW with 1e-5 to 3e-4 and warm-up)
/// </summary>
public class VideoCLIPNeuralNetworkTests : NeuralNetworkModelTestBase
{
    // Paper input: [numFrames, channels, height, width] — video as frame sequence
    // 4 frames × 3 RGB channels × 32×32 (scaled from 224×224)
    protected override int[] InputShape => [4, 3, 32, 32];

    // Paper output: [1, embeddingDim] — joint video-text embedding space
    protected override int[] OutputShape => [1, 64];

    // Contrastive learning on L2-normalized embeddings has a bounded loss landscape
    // (cosine similarity in [-1, 1]). The optimizer oscillates near the minimum rather
    // than monotonically converging, which is expected per paper §4 training dynamics.
    protected override double MoreDataTolerance => 0.05;

    protected override INeuralNetworkModel<double> CreateNetwork()
    {
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputHeight: 32,
            inputWidth: 32,
            inputDepth: 3,
            outputSize: 64);

        // Paper §4: AdamW with LR 1e-5 to 3e-4 and cosine warm-up schedule. Use
        // the peak of that range (3e-4) for a static-LR test — the warm-up's
        // purpose is to ramp UP to the peak, so steady-state training without
        // warm-up should use the peak, not the midpoint.
        var optimizerOptions = new AdamOptimizerOptions<double, Tensor<double>, Tensor<double>>
        {
            InitialLearningRate = 3e-4
        };

        // Paper §3 defines training in unit-norm embedding space via cosine
        // similarity (the InfoNCE numerator is exp(cos_sim/τ)). VideoCLIP's
        // forward returns an L2-normalized embedding [1, embeddingDim], so the
        // paper-faithful single-pair training signal is "drive cosine(output,
        // target) toward 1" — exactly what CosineSimilarityLoss computes
        // (1 − cos(o, t)). The model's default constructor sets
        // CrossEntropyWithLogitsLoss as a generic fallback, which is wrong for
        // a unit-norm output (it routes the embedding through softmax and
        // computes class-CE against a continuous target, producing a ~136
        // baseline that barely moves regardless of training success — the loss
        // formula plateau, not a gradient bug). Override here so the test
        // measures actual embedding alignment.
        var model = new VideoCLIPNeuralNetwork<double>(
            architecture,
            imageSize: 32,              // Paper: 224 (ViT-B/16 input resolution)
            channels: 3,                // Paper: 3 RGB channels
            patchSize: 16,              // Paper: 16 (ViT-B/16 patch size)
            vocabularySize: 49408,      // Paper: 49408 BPE tokens (CLIP tokenizer)
            maxSequenceLength: 77,      // Paper: 77 (CLIP text encoder max length)
            embeddingDimension: 64,     // Paper: 512 (joint embedding space)
            visionHiddenDim: 128,       // Paper: 768 (ViT-B hidden dim)
            textHiddenDim: 64,          // Paper: 512 (CLIP text encoder hidden dim)
            numFrameEncoderLayers: 2,   // Paper: 12 (ViT-B depth)
            numTemporalLayers: 2,       // Paper: 4 (temporal transformer depth)
            numTextLayers: 2,           // Paper: 12 (CLIP text encoder depth)
            numHeads: 4,                // Paper: 12 (ViT-B attention heads)
            numFrames: 4,               // Paper: 8 (sampled frames per video)
            frameRate: 1.0,             // Paper: 1 FPS sampling
            temporalAggregation: TemporalAggregationType.TemporalTransformer,
            optimizer: new AdamOptimizer<double, Tensor<double>, Tensor<double>>(null, optimizerOptions),
            lossFunction: new CosineSimilarityLoss<double>());

        return model;
    }
}
