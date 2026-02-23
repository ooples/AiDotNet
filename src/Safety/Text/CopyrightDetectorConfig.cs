namespace AiDotNet.Safety.Text;

/// <summary>
/// Configuration for copyright and memorization detection modules.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Use this to configure copyright detection. You can set
/// the n-gram size for overlap detection and the threshold for flagging content
/// as potentially copyrighted or memorized.
/// </para>
/// </remarks>
public class CopyrightDetectorConfig
{
    /// <summary>N-gram size for overlap detection. Default: 50.</summary>
    public int? NgramSize { get; set; }

    /// <summary>Memorization score threshold (0.0-1.0). Default: 0.5.</summary>
    public double? Threshold { get; set; }

    /// <summary>Minimum text length to analyze. Default: 100.</summary>
    public int? MinTextLength { get; set; }

    internal int EffectiveNgramSize => NgramSize ?? 50;
    internal double EffectiveThreshold => Threshold ?? 0.5;
    internal int EffectiveMinTextLength => MinTextLength ?? 100;
}
