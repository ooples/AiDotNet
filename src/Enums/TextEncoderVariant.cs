namespace AiDotNet.Enums;

/// <summary>
/// Variants for SigLIP text encoder models.
/// </summary>
public enum SigLIPVariant
{
    /// <summary>SigLIP Base: 768-dim, 12 layers, 12 heads.</summary>
    Base,
    /// <summary>SigLIP Large: 1024-dim, 24 layers, 16 heads.</summary>
    Large,
    /// <summary>SigLIP So400M: 1152-dim, 27 layers, 16 heads.</summary>
    So400M
}

/// <summary>
/// Variants for SigLIP 2 text encoder models.
/// </summary>
public enum SigLIP2Variant
{
    /// <summary>SigLIP 2 Base: 768-dim, 12 layers, 12 heads.</summary>
    Base,
    /// <summary>SigLIP 2 Large: 1024-dim, 24 layers, 16 heads.</summary>
    Large,
    /// <summary>SigLIP 2 So400M: 1152-dim, 27 layers, 16 heads.</summary>
    So400M
}

/// <summary>
/// Variants for CLIP text encoder models.
/// </summary>
public enum CLIPVariant
{
    /// <summary>CLIP ViT-L/14: 768-dim, 12 layers. Used in SD 1.x, encoder 1 of SDXL/SD3/FLUX.</summary>
    ViTL14,
    /// <summary>CLIP ViT-H/14: 1024-dim, 24 layers. Used in SD 2.x.</summary>
    ViTH14,
    /// <summary>CLIP ViT-bigG/14: 1280-dim, 32 layers. Encoder 2 of SDXL/SD3.</summary>
    ViTBigG14
}

/// <summary>
/// Variants for distilled T5 text encoder models.
/// </summary>
public enum DistilledT5Variant
{
    /// <summary>Small: 512-dim hidden, 6 layers.</summary>
    Small,
    /// <summary>Base: 768-dim hidden, 12 layers.</summary>
    Base,
    /// <summary>Large: 1024-dim hidden, 24 layers.</summary>
    Large
}

/// <summary>
/// Variants for Gemma text encoder models.
/// </summary>
public enum GemmaVariant
{
    /// <summary>Gemma 2B: 2048-dim, 18 layers.</summary>
    TwoB,
    /// <summary>Gemma 7B: 3072-dim, 28 layers.</summary>
    SevenB
}

/// <summary>
/// Variants for Qwen2 text encoder models.
/// </summary>
public enum Qwen2Variant
{
    /// <summary>Qwen2 1.5B: 1536-dim, 28 layers.</summary>
    OnePointFiveB,
    /// <summary>Qwen2 7B: 4096-dim, 32 layers.</summary>
    SevenB
}

/// <summary>
/// Variants for MMDiT noise predictor architectures.
/// </summary>
public enum MMDiTXVariant
{
    /// <summary>SD3.5 Medium: 24 joint layers, 2048 hidden.</summary>
    Medium,
    /// <summary>SD3.5 Large: 38 joint layers, 2560 hidden.</summary>
    Large,
    /// <summary>SD3.5 Large Turbo: 38 joint layers with distillation.</summary>
    LargeTurbo
}

/// <summary>
/// Variants for FLUX noise predictor architectures.
/// </summary>
public enum FluxPredictorVariant
{
    /// <summary>FLUX.1 Dev: 19 double-stream + 38 single-stream blocks.</summary>
    Dev,
    /// <summary>FLUX.1 Schnell: Same architecture, distilled for 1-4 steps.</summary>
    Schnell,
    /// <summary>FLUX.2: Next generation with improved architecture.</summary>
    V2
}

/// <summary>
/// Variants for FLUX model.
/// </summary>
public enum FluxVariant
{
    /// <summary>FLUX.1 [dev]: Open-weight, guidance-distilled, non-commercial license.</summary>
    Dev,
    /// <summary>FLUX.1 [schnell]: Fast 1-4 step generation, Apache 2.0 license.</summary>
    Schnell,
    /// <summary>FLUX.1 [pro]: Best quality, API-only.</summary>
    Pro
}

/// <summary>
/// Variants for HiDream-I1 model.
/// </summary>
public enum HiDreamVariant
{
    /// <summary>HiDream-I1 Full: 17B parameters, highest quality.</summary>
    Full,
    /// <summary>HiDream-I1 Dev: 12B parameters, good balance of quality and speed.</summary>
    Dev,
    /// <summary>HiDream-I1 Fast: 8B parameters, optimized for speed.</summary>
    Fast
}

/// <summary>
/// Versions for the Kandinsky model family.
/// </summary>
public enum KandinskyVersion
{
    /// <summary>Kandinsky 2.2: 512x512 native resolution, CLIP ViT-G/14 prior.</summary>
    V2_2,
    /// <summary>Kandinsky 3.0: 1024x1024 native resolution, improved architecture.</summary>
    V3_0
}

/// <summary>
/// Variants for the PixArt model family.
/// </summary>
public enum PixArtVariant
{
    /// <summary>PixArt-α: 600M params, 1024px default resolution.</summary>
    Alpha,
    /// <summary>PixArt-Σ: Similar to α, 512px default resolution.</summary>
    Sigma,
    /// <summary>PixArt-δ: Smaller 256px variant.</summary>
    Delta,
    /// <summary>PixArt-XL: Larger variant with more parameters, 1024px.</summary>
    XL
}

/// <summary>
/// Variants for the SANA model family.
/// </summary>
public enum SANAVariant
{
    /// <summary>SANA 0.6B: Default 600M parameter linear DiT.</summary>
    Small,
    /// <summary>SANA 1.6B: Larger 1.6B parameter variant for higher quality.</summary>
    Large
}

/// <summary>
/// Variants for the Stable Diffusion 3 model family.
/// </summary>
public enum SD3Variant
{
    /// <summary>SD3 Medium: 2B MMDiT parameters, 24 layers.</summary>
    Medium,
    /// <summary>SD3.5 Large: 8B MMDiT parameters, 38 layers.</summary>
    Large,
    /// <summary>SD3.5 Large Turbo: 8B distilled for 4-step generation.</summary>
    LargeTurbo
}
