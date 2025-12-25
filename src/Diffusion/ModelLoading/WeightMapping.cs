using System.Text.RegularExpressions;

namespace AiDotNet.Diffusion.ModelLoading;

/// <summary>
/// Maps weight names between different model formats.
/// </summary>
/// <remarks>
/// <para>
/// Different ML frameworks and model releases use different naming conventions.
/// This class provides mappings to translate between them.
/// </para>
/// <para>
/// <b>For Beginners:</b> Model weights have names like paths in a file system.
///
/// For example, in Stable Diffusion:
/// - HuggingFace: "model.diffusion_model.input_blocks.0.0.weight"
/// - Our model: "unet.inputConv.weight"
///
/// This class translates between these naming conventions so we can
/// load weights from any source into our models.
/// </para>
/// </remarks>
public class WeightMapping
{
    /// <summary>
    /// Direct name-to-name mappings.
    /// </summary>
    private readonly Dictionary<string, string> _mappings;

    /// <summary>
    /// Pattern-based mappings (regex -> replacement).
    /// </summary>
    private readonly List<(Regex Pattern, string Replacement)> _patternMappings;

    /// <summary>
    /// Initializes a new instance with custom mappings.
    /// </summary>
    /// <param name="mappings">Direct name mappings.</param>
    public WeightMapping(Dictionary<string, string>? mappings = null)
    {
        _mappings = mappings ?? new Dictionary<string, string>();
        _patternMappings = new List<(Regex, string)>();
    }

    /// <summary>
    /// Adds a direct name mapping.
    /// </summary>
    /// <param name="sourceName">Source weight name.</param>
    /// <param name="targetName">Target weight name.</param>
    public void AddMapping(string sourceName, string targetName)
    {
        _mappings[sourceName] = targetName;
    }

    /// <summary>
    /// Adds a pattern-based mapping.
    /// </summary>
    /// <param name="pattern">Regex pattern to match.</param>
    /// <param name="replacement">Replacement pattern.</param>
    public void AddPatternMapping(string pattern, string replacement)
    {
        _patternMappings.Add((new Regex(pattern, RegexOptions.Compiled), replacement));
    }

    /// <summary>
    /// Maps a source weight name to the target name.
    /// </summary>
    /// <param name="sourceName">The source weight name.</param>
    /// <returns>The mapped target name, or null if no mapping exists.</returns>
    public string? Map(string sourceName)
    {
        // Try direct mapping first
        if (_mappings.TryGetValue(sourceName, out var mapped))
        {
            return mapped;
        }

        // Try pattern mappings
        foreach (var (pattern, replacement) in _patternMappings)
        {
            if (pattern.IsMatch(sourceName))
            {
                return pattern.Replace(sourceName, replacement);
            }
        }

        return null;
    }

    /// <summary>
    /// Creates weight mapping for Stable Diffusion v1.x VAE.
    /// </summary>
    /// <returns>Weight mapping configured for SD v1.x VAE.</returns>
    public static WeightMapping CreateStableDiffusionV1VAE()
    {
        var mapping = new WeightMapping();

        // Encoder mappings
        mapping.AddMapping("encoder.conv_in.weight", "vae.inputConv.weight");
        mapping.AddMapping("encoder.conv_in.bias", "vae.inputConv.bias");

        // Decoder mappings
        mapping.AddMapping("decoder.conv_out.weight", "vae.outputConv.weight");
        mapping.AddMapping("decoder.conv_out.bias", "vae.outputConv.bias");

        // Latent projections
        mapping.AddMapping("quant_conv.weight", "vae.quantConv.weight");
        mapping.AddMapping("quant_conv.bias", "vae.quantConv.bias");
        mapping.AddMapping("post_quant_conv.weight", "vae.postQuantConv.weight");
        mapping.AddMapping("post_quant_conv.bias", "vae.postQuantConv.bias");

        // Pattern mappings for encoder blocks
        // encoder.down.{level}.block.{block}.norm1.weight -> vae.encoder.down{level}.res{block}.norm1.gamma
        mapping.AddPatternMapping(
            @"encoder\.down\.(\d+)\.block\.(\d+)\.norm1\.weight",
            "vae.encoder.down$1.res$2.norm1.gamma");
        mapping.AddPatternMapping(
            @"encoder\.down\.(\d+)\.block\.(\d+)\.norm1\.bias",
            "vae.encoder.down$1.res$2.norm1.beta");
        mapping.AddPatternMapping(
            @"encoder\.down\.(\d+)\.block\.(\d+)\.norm2\.weight",
            "vae.encoder.down$1.res$2.norm2.gamma");
        mapping.AddPatternMapping(
            @"encoder\.down\.(\d+)\.block\.(\d+)\.norm2\.bias",
            "vae.encoder.down$1.res$2.norm2.beta");
        mapping.AddPatternMapping(
            @"encoder\.down\.(\d+)\.block\.(\d+)\.conv1\.weight",
            "vae.encoder.down$1.res$2.conv1.weight");
        mapping.AddPatternMapping(
            @"encoder\.down\.(\d+)\.block\.(\d+)\.conv1\.bias",
            "vae.encoder.down$1.res$2.conv1.bias");
        mapping.AddPatternMapping(
            @"encoder\.down\.(\d+)\.block\.(\d+)\.conv2\.weight",
            "vae.encoder.down$1.res$2.conv2.weight");
        mapping.AddPatternMapping(
            @"encoder\.down\.(\d+)\.block\.(\d+)\.conv2\.bias",
            "vae.encoder.down$1.res$2.conv2.bias");

        // Downsample convolutions
        mapping.AddPatternMapping(
            @"encoder\.down\.(\d+)\.downsample\.conv\.weight",
            "vae.encoder.down$1.downsample.weight");
        mapping.AddPatternMapping(
            @"encoder\.down\.(\d+)\.downsample\.conv\.bias",
            "vae.encoder.down$1.downsample.bias");

        // Pattern mappings for decoder blocks
        mapping.AddPatternMapping(
            @"decoder\.up\.(\d+)\.block\.(\d+)\.norm1\.weight",
            "vae.decoder.up$1.res$2.norm1.gamma");
        mapping.AddPatternMapping(
            @"decoder\.up\.(\d+)\.block\.(\d+)\.norm1\.bias",
            "vae.decoder.up$1.res$2.norm1.beta");
        mapping.AddPatternMapping(
            @"decoder\.up\.(\d+)\.block\.(\d+)\.norm2\.weight",
            "vae.decoder.up$1.res$2.norm2.gamma");
        mapping.AddPatternMapping(
            @"decoder\.up\.(\d+)\.block\.(\d+)\.norm2\.bias",
            "vae.decoder.up$1.res$2.norm2.beta");
        mapping.AddPatternMapping(
            @"decoder\.up\.(\d+)\.block\.(\d+)\.conv1\.weight",
            "vae.decoder.up$1.res$2.conv1.weight");
        mapping.AddPatternMapping(
            @"decoder\.up\.(\d+)\.block\.(\d+)\.conv1\.bias",
            "vae.decoder.up$1.res$2.conv1.bias");
        mapping.AddPatternMapping(
            @"decoder\.up\.(\d+)\.block\.(\d+)\.conv2\.weight",
            "vae.decoder.up$1.res$2.conv2.weight");
        mapping.AddPatternMapping(
            @"decoder\.up\.(\d+)\.block\.(\d+)\.conv2\.bias",
            "vae.decoder.up$1.res$2.conv2.bias");

        // Upsample convolutions
        mapping.AddPatternMapping(
            @"decoder\.up\.(\d+)\.upsample\.conv\.weight",
            "vae.decoder.up$1.upsample.weight");
        mapping.AddPatternMapping(
            @"decoder\.up\.(\d+)\.upsample\.conv\.bias",
            "vae.decoder.up$1.upsample.bias");

        // Skip connection convolutions (nin_shortcut)
        mapping.AddPatternMapping(
            @"encoder\.down\.(\d+)\.block\.(\d+)\.nin_shortcut\.weight",
            "vae.encoder.down$1.res$2.skip.weight");
        mapping.AddPatternMapping(
            @"encoder\.down\.(\d+)\.block\.(\d+)\.nin_shortcut\.bias",
            "vae.encoder.down$1.res$2.skip.bias");
        mapping.AddPatternMapping(
            @"decoder\.up\.(\d+)\.block\.(\d+)\.nin_shortcut\.weight",
            "vae.decoder.up$1.res$2.skip.weight");
        mapping.AddPatternMapping(
            @"decoder\.up\.(\d+)\.block\.(\d+)\.nin_shortcut\.bias",
            "vae.decoder.up$1.res$2.skip.bias");

        return mapping;
    }

    /// <summary>
    /// Creates weight mapping for Stable Diffusion v1.x UNet.
    /// </summary>
    /// <returns>Weight mapping configured for SD v1.x UNet.</returns>
    public static WeightMapping CreateStableDiffusionV1UNet()
    {
        var mapping = new WeightMapping();

        // Time embedding
        mapping.AddMapping("time_embed.0.weight", "unet.timeEmbed.linear1.weight");
        mapping.AddMapping("time_embed.0.bias", "unet.timeEmbed.linear1.bias");
        mapping.AddMapping("time_embed.2.weight", "unet.timeEmbed.linear2.weight");
        mapping.AddMapping("time_embed.2.bias", "unet.timeEmbed.linear2.bias");

        // Input blocks pattern mappings
        mapping.AddPatternMapping(
            @"input_blocks\.(\d+)\.0\.in_layers\.0\.weight",
            "unet.down$1.norm1.gamma");
        mapping.AddPatternMapping(
            @"input_blocks\.(\d+)\.0\.in_layers\.0\.bias",
            "unet.down$1.norm1.beta");
        mapping.AddPatternMapping(
            @"input_blocks\.(\d+)\.0\.in_layers\.2\.weight",
            "unet.down$1.conv1.weight");
        mapping.AddPatternMapping(
            @"input_blocks\.(\d+)\.0\.in_layers\.2\.bias",
            "unet.down$1.conv1.bias");

        // Output blocks pattern mappings
        mapping.AddPatternMapping(
            @"output_blocks\.(\d+)\.0\.in_layers\.0\.weight",
            "unet.up$1.norm1.gamma");
        mapping.AddPatternMapping(
            @"output_blocks\.(\d+)\.0\.in_layers\.0\.bias",
            "unet.up$1.norm1.beta");

        // Middle block
        mapping.AddMapping("middle_block.0.in_layers.0.weight", "unet.mid.block1.norm1.gamma");
        mapping.AddMapping("middle_block.0.in_layers.0.bias", "unet.mid.block1.norm1.beta");
        mapping.AddMapping("middle_block.2.in_layers.0.weight", "unet.mid.block2.norm1.gamma");
        mapping.AddMapping("middle_block.2.in_layers.0.bias", "unet.mid.block2.norm1.beta");

        return mapping;
    }

    /// <summary>
    /// Creates weight mapping for SDXL VAE.
    /// </summary>
    /// <returns>Weight mapping configured for SDXL VAE.</returns>
    public static WeightMapping CreateSDXLVAE()
    {
        // SDXL VAE uses similar structure to SD v1.x but with some differences
        var mapping = CreateStableDiffusionV1VAE();

        // Add any SDXL-specific overrides here
        // SDXL uses fp16 by default and has some additional layers

        return mapping;
    }

    /// <summary>
    /// Creates weight mapping for CLIP text encoder.
    /// </summary>
    /// <returns>Weight mapping configured for CLIP text encoder.</returns>
    public static WeightMapping CreateCLIPTextEncoder()
    {
        var mapping = new WeightMapping();

        // Token embedding
        mapping.AddMapping(
            "text_model.embeddings.token_embedding.weight",
            "textEncoder.tokenEmbedding.weight");

        // Position embedding
        mapping.AddMapping(
            "text_model.embeddings.position_embedding.weight",
            "textEncoder.positionEmbedding.weight");

        // Transformer blocks
        mapping.AddPatternMapping(
            @"text_model\.encoder\.layers\.(\d+)\.self_attn\.q_proj\.weight",
            "textEncoder.layers.$1.selfAttn.toQ.weight");
        mapping.AddPatternMapping(
            @"text_model\.encoder\.layers\.(\d+)\.self_attn\.k_proj\.weight",
            "textEncoder.layers.$1.selfAttn.toK.weight");
        mapping.AddPatternMapping(
            @"text_model\.encoder\.layers\.(\d+)\.self_attn\.v_proj\.weight",
            "textEncoder.layers.$1.selfAttn.toV.weight");
        mapping.AddPatternMapping(
            @"text_model\.encoder\.layers\.(\d+)\.self_attn\.out_proj\.weight",
            "textEncoder.layers.$1.selfAttn.toOut.weight");

        // Layer norms
        mapping.AddPatternMapping(
            @"text_model\.encoder\.layers\.(\d+)\.layer_norm1\.weight",
            "textEncoder.layers.$1.norm1.gamma");
        mapping.AddPatternMapping(
            @"text_model\.encoder\.layers\.(\d+)\.layer_norm1\.bias",
            "textEncoder.layers.$1.norm1.beta");
        mapping.AddPatternMapping(
            @"text_model\.encoder\.layers\.(\d+)\.layer_norm2\.weight",
            "textEncoder.layers.$1.norm2.gamma");
        mapping.AddPatternMapping(
            @"text_model\.encoder\.layers\.(\d+)\.layer_norm2\.bias",
            "textEncoder.layers.$1.norm2.beta");

        // MLP layers
        mapping.AddPatternMapping(
            @"text_model\.encoder\.layers\.(\d+)\.mlp\.fc1\.weight",
            "textEncoder.layers.$1.mlp.fc1.weight");
        mapping.AddPatternMapping(
            @"text_model\.encoder\.layers\.(\d+)\.mlp\.fc1\.bias",
            "textEncoder.layers.$1.mlp.fc1.bias");
        mapping.AddPatternMapping(
            @"text_model\.encoder\.layers\.(\d+)\.mlp\.fc2\.weight",
            "textEncoder.layers.$1.mlp.fc2.weight");
        mapping.AddPatternMapping(
            @"text_model\.encoder\.layers\.(\d+)\.mlp\.fc2\.bias",
            "textEncoder.layers.$1.mlp.fc2.bias");

        // Final layer norm
        mapping.AddMapping("text_model.final_layer_norm.weight", "textEncoder.finalNorm.gamma");
        mapping.AddMapping("text_model.final_layer_norm.bias", "textEncoder.finalNorm.beta");

        return mapping;
    }

    /// <summary>
    /// Gets all direct mappings.
    /// </summary>
    public IReadOnlyDictionary<string, string> DirectMappings => _mappings;

    /// <summary>
    /// Gets the count of direct mappings.
    /// </summary>
    public int DirectMappingCount => _mappings.Count;

    /// <summary>
    /// Gets the count of pattern mappings.
    /// </summary>
    public int PatternMappingCount => _patternMappings.Count;
}
