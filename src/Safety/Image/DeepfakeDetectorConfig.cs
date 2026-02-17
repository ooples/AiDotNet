namespace AiDotNet.Safety.Image;

/// <summary>
/// Configuration for deepfake and AI-generated image detection modules.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Use this to configure how the deepfake detector works.
/// You can set the detection threshold and choose which analysis methods to enable.
/// </para>
/// </remarks>
public class DeepfakeDetectorConfig
{
    /// <summary>Detection threshold (0.0-1.0). Default: 0.5.</summary>
    public double? Threshold { get; set; }

    /// <summary>Whether to use frequency domain analysis. Default: true.</summary>
    public bool? FrequencyAnalysis { get; set; }

    /// <summary>Whether to use consistency analysis. Default: true.</summary>
    public bool? ConsistencyAnalysis { get; set; }

    /// <summary>Whether to use provenance/metadata analysis. Default: true.</summary>
    public bool? ProvenanceAnalysis { get; set; }

    internal double EffectiveThreshold => Threshold ?? 0.5;
    internal bool EffectiveFrequencyAnalysis => FrequencyAnalysis ?? true;
    internal bool EffectiveConsistencyAnalysis => ConsistencyAnalysis ?? true;
    internal bool EffectiveProvenanceAnalysis => ProvenanceAnalysis ?? true;
}
