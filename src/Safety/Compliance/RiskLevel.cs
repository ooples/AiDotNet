namespace AiDotNet.Safety.Compliance;

/// <summary>
/// EU AI Act risk classification levels (Articles 5, 6, 50, 52).
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> The EU AI Act classifies AI systems into risk levels.
/// Unacceptable systems are banned. High-risk systems face strict requirements.
/// Limited-risk systems need transparency. Minimal-risk systems are largely unregulated.
/// </para>
/// </remarks>
public enum RiskLevel
{
    /// <summary>Banned practices (social scoring, real-time biometric surveillance). Article 5.</summary>
    Unacceptable,

    /// <summary>Strict requirements (safety, transparency, human oversight). Article 6.</summary>
    High,

    /// <summary>Transparency requirements (chatbots, deepfakes). Article 50/52.</summary>
    Limited,

    /// <summary>Largely unregulated (spam filters, video games).</summary>
    Minimal
}
