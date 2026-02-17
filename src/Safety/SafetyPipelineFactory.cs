using AiDotNet.Safety.Text;

namespace AiDotNet.Safety;

/// <summary>
/// Factory that constructs a <see cref="SafetyPipeline{T}"/> from a <see cref="SafetyConfig"/>.
/// </summary>
/// <remarks>
/// <para>
/// This factory reads the SafetyConfig and instantiates the appropriate safety modules
/// based on which features are enabled. It is called internally by the AiModelBuilder
/// during the build process.
/// </para>
/// <para>
/// <b>For Beginners:</b> You don't need to use this factory directly. When you call
/// <c>ConfigureSafety(...)</c> on the AiModelBuilder, this factory automatically creates
/// the right safety modules based on your settings.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public static class SafetyPipelineFactory<T>
{
    /// <summary>
    /// Creates a safety pipeline from the given configuration.
    /// </summary>
    /// <param name="config">The safety configuration. If null, defaults are used.</param>
    /// <returns>A configured safety pipeline with the appropriate modules registered.</returns>
    public static SafetyPipeline<T> Create(SafetyConfig? config = null)
    {
        var effectiveConfig = config ?? new SafetyConfig();
        var pipeline = new SafetyPipeline<T>(effectiveConfig);

        if (!effectiveConfig.EffectiveEnabled)
        {
            return pipeline;
        }

        // Register text safety modules
        RegisterTextModules(pipeline, effectiveConfig);

        // Future: RegisterImageModules, RegisterAudioModules, RegisterVideoModules

        return pipeline;
    }

    private static void RegisterTextModules(SafetyPipeline<T> pipeline, SafetyConfig config)
    {
        // Toxicity detection
        if (config.Text.EffectiveToxicityDetection)
        {
            pipeline.AddModule(new RuleBasedToxicityDetector<T>(config.Text.EffectiveToxicityThreshold));
        }

        // PII detection
        if (config.Text.EffectivePIIDetection)
        {
            pipeline.AddModule(new RegexPIIDetector<T>());
        }

        // Jailbreak detection
        if (config.Text.EffectiveJailbreakDetection)
        {
            pipeline.AddModule(new PatternJailbreakDetector<T>(config.Text.EffectiveJailbreakSensitivity));
        }
    }
}
