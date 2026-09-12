using AiDotNet.Configuration;

namespace AiDotNet;

public partial class AiModelBuilder<T, TInput, TOutput>
{
    /// <summary>Creates a builder from an already parsed configuration without reading a file or resolving variables again.</summary>
    /// <param name="config">The parsed configuration, applied through the same Configure methods as the YAML constructor.</param>
    /// <returns>A new builder containing the configured sections.</returns>
    /// <remarks>
    /// This lets preflight and run setup share one parsed document. Individual Configure methods retain their
    /// existing snapshot and caller-owned service contracts; the factory does not promise to clone arbitrary services.
    /// </remarks>
    public static AiModelBuilder<T, TInput, TOutput> FromConfiguration(YamlModelConfig config)
    {
        if (config is null) throw new ArgumentNullException(nameof(config));
        var builder = new AiModelBuilder<T, TInput, TOutput>();
        YamlConfigApplier<T, TInput, TOutput>.Apply(config, builder);
        return builder;
    }
}
