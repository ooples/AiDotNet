namespace AiDotNet.Enums;

/// <summary>
/// A family of adversarial prompts used when red teaming the safety pipeline.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Red teaming means attacking your own system on purpose to find its weak spots.
/// Each category draws prompts from one of the library's safety benchmarks.</para>
/// </remarks>
public enum RedTeamingCategory
{
    /// <summary>Instruction-override and role-play jailbreaks (the jailbreak benchmark).</summary>
    Jailbreaking,

    /// <summary>Toxic, violent or hateful requests (the toxicity benchmark).</summary>
    HarmfulContent,

    /// <summary>Fabricated or contradictory claims (the hallucination benchmark).</summary>
    Misinformation,

    /// <summary>Prompts carrying or soliciting personal data (the PII benchmark).</summary>
    PrivacyViolation,

    /// <summary>Stereotyping and demographic-bias probes (the bias benchmark).</summary>
    BiasExploration
}
