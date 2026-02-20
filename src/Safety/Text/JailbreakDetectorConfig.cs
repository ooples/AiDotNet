namespace AiDotNet.Safety.Text;

/// <summary>
/// Configuration for jailbreak and prompt injection detection modules.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Use this to configure how aggressively the jailbreak detector
/// works. Higher sensitivity catches more attack attempts but may also flag legitimate
/// prompts. Lower sensitivity is more lenient.
/// </para>
/// </remarks>
public class JailbreakDetectorConfig
{
    /// <summary>Detection sensitivity (0.0 = lenient, 1.0 = strict). Default: 0.5.</summary>
    public double? Sensitivity { get; set; }

    /// <summary>Whether to check for encoding attacks (Base64, ROT13, Unicode). Default: true.</summary>
    public bool? DetectEncodingAttacks { get; set; }

    /// <summary>Whether to check for multi-turn escalation attacks. Default: true.</summary>
    public bool? DetectMultiTurnAttacks { get; set; }

    /// <summary>Whether to check for character injection (emoji/Unicode smuggling). Default: true.</summary>
    public bool? DetectCharacterInjection { get; set; }

    internal double EffectiveSensitivity => Sensitivity ?? 0.5;
    internal bool EffectiveDetectEncodingAttacks => DetectEncodingAttacks ?? true;
    internal bool EffectiveDetectMultiTurnAttacks => DetectMultiTurnAttacks ?? true;
    internal bool EffectiveDetectCharacterInjection => DetectCharacterInjection ?? true;
}
