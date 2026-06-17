using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.VisionLanguage.Proprietary;
using AiDotNet.Tests.ModelFamilyTests.Base;

namespace AiDotNet.Tests.ModelFamilyTests.NeuralNetworks;

/// <summary>
/// Manual test scaffold for GrokVision (xAI's proprietary multimodal model).
/// xAI has not published a research paper or whitepaper detailing Grok-1.5V's
/// exact architecture, so <see cref="GrokVisionOptions"/> defaults are the
/// best-estimate production values (VisionDim=1024, DecoderDim=8192,
/// 32 vision + 64 decoder layers, 64 heads). The auto-generated test
/// scaffold's <c>[3, 128, 128]</c> raw-image input hits an immediate shape
/// mismatch because the model's first layer is
/// <c>LayerNormalizationLayer</c> + <c>MultiHeadAttentionLayer(VisionDim)</c>
/// — it expects post-patch-embedding token tensors of shape
/// <c>[batch, num_tokens, VisionDim]</c>, NOT raw pixels. Override the
/// input shape only; keep production-scale model defaults so weight
/// streaming (NeuralNetworkBase.TryAutoEnableWeightStreaming) engages on
/// the layer-by-layer parameter budget, matching production deployment.
/// </summary>
public class GrokVisionTests : NeuralNetworkModelTestBase<float>
{
    // [batch=1, num_tokens=4, vision_dim=1024]. vision_dim must equal
    // GrokVisionOptions.VisionDim (1024) so the first MultiHeadAttention's
    // weight matrix matches; num_tokens kept small so attention's per-step
    // intermediate tensors stay bounded.
    protected override int[] InputShape => [1, 4, 1024];

    // Decoder emits DecoderDim (8192) per token.
    protected override int[] OutputShape => [1, 4, 8192];

    protected override INeuralNetworkModel<float> CreateNetwork()
        => new GrokVision<float>(
            architecture: new NeuralNetworkArchitecture<float>(
                inputType: InputType.OneDimensional,
                taskType: NeuralNetworkTaskType.TextGeneration,
                inputSize: 1024,
                outputSize: 8192),
            options: new GrokVisionOptions());
}
