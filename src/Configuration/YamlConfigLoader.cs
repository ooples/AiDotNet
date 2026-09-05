using YamlDotNet.Core;
using YamlDotNet.Serialization;
using YamlDotNet.Serialization.NamingConventions;

namespace AiDotNet.Configuration;

/// <summary>
/// Loads and deserializes YAML configuration files into strongly-typed configuration objects.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> This class reads a YAML file from disk (or a YAML string)
/// and converts it into a structured C# object that the builder or trainer can use. YAML uses
/// camelCase property names (e.g., <c>timeSeriesModel</c>, <c>gpuAcceleration</c>).</para>
///
/// <para>
/// Every document passes through <see cref="YamlVariableResolver"/> first, so <c>${NAME}</c> anywhere in the file is
/// replaced by the environment variable of that name and <c>${NAME:-fallback}</c> supplies a default. That is what
/// keeps an API key or a machine-specific path out of a file you want to commit; a reference with neither a value nor
/// a fallback fails immediately and names the variable.
/// </para>
///
/// <para><b>Example usage:</b></para>
/// <code>
/// // Load AiModelBuilder config
/// var config = YamlConfigLoader.LoadFromFile("model-config.yaml");
///
/// // Load training recipe config
/// var recipe = YamlConfigLoader.LoadFromFile&lt;TrainingRecipeConfig&gt;("training-recipe.yaml");
///
/// // Load from string
/// var config2 = YamlConfigLoader.LoadFromString&lt;TrainingRecipeConfig&gt;(yamlContent);
/// </code>
/// </remarks>
public static class YamlConfigLoader
{
    /// <summary>
    /// Loads a YAML configuration file from disk and deserializes it into a <see cref="YamlModelConfig"/>.
    /// </summary>
    /// <param name="filePath">The absolute or relative path to the YAML file.</param>
    /// <returns>A deserialized <see cref="YamlModelConfig"/> instance.</returns>
    /// <exception cref="ArgumentException">Thrown when <paramref name="filePath"/> is null or whitespace.</exception>
    /// <exception cref="FileNotFoundException">Thrown when the file does not exist.</exception>
    public static YamlModelConfig LoadFromFile(string filePath)
    {
        if (string.IsNullOrWhiteSpace(filePath))
        {
            throw new ArgumentException("Config file path cannot be null or empty.", nameof(filePath));
        }

        if (!File.Exists(filePath))
        {
            throw new FileNotFoundException($"YAML config file not found: {filePath}", filePath);
        }

        string yamlContent = File.ReadAllText(filePath);
        return LoadFromString(yamlContent);
    }

    /// <summary>
    /// Deserializes a YAML string into a <see cref="YamlModelConfig"/>.
    /// </summary>
    /// <param name="yamlContent">The YAML content as a string.</param>
    /// <returns>A deserialized <see cref="YamlModelConfig"/> instance.</returns>
    /// <exception cref="ArgumentException">Thrown when <paramref name="yamlContent"/> is null or whitespace.</exception>
    public static YamlModelConfig LoadFromString(string yamlContent)
    {
        if (string.IsNullOrWhiteSpace(yamlContent))
        {
            throw new ArgumentException("YAML content cannot be null or empty.", nameof(yamlContent));
        }

        IDeserializer deserializer = CreateDeserializer();

        YamlModelConfig? config;
        try
        {
            config = deserializer.Deserialize<YamlModelConfig>(YamlVariableResolver.Resolve(yamlContent));
        }
        catch (YamlException ex)
        {
            throw new ArgumentException("YAML deserialization failed.", nameof(yamlContent), ex);
        }

        return config ?? new YamlModelConfig();
    }

    /// <summary>
    /// Loads a YAML configuration file from disk and deserializes it into the specified type.
    /// </summary>
    /// <typeparam name="TConfig">The configuration type to deserialize into. Must have a parameterless constructor.</typeparam>
    /// <param name="filePath">The absolute or relative path to the YAML file.</param>
    /// <returns>A deserialized instance of <typeparamref name="TConfig"/>.</returns>
    /// <exception cref="ArgumentException">Thrown when <paramref name="filePath"/> is null or whitespace.</exception>
    /// <exception cref="FileNotFoundException">Thrown when the file does not exist.</exception>
    public static TConfig LoadFromFile<TConfig>(string filePath) where TConfig : new()
    {
        if (string.IsNullOrWhiteSpace(filePath))
        {
            throw new ArgumentException("Config file path cannot be null or empty.", nameof(filePath));
        }

        if (!File.Exists(filePath))
        {
            throw new FileNotFoundException($"YAML config file not found: {filePath}", filePath);
        }

        string yamlContent = File.ReadAllText(filePath);
        return LoadFromString<TConfig>(yamlContent);
    }

    /// <summary>
    /// Deserializes a YAML string into the specified type.
    /// </summary>
    /// <typeparam name="TConfig">The configuration type to deserialize into. Must have a parameterless constructor.</typeparam>
    /// <param name="yamlContent">The YAML content as a string.</param>
    /// <returns>A deserialized instance of <typeparamref name="TConfig"/>.</returns>
    /// <exception cref="ArgumentException">Thrown when <paramref name="yamlContent"/> is null or whitespace.</exception>
    public static TConfig LoadFromString<TConfig>(string yamlContent) where TConfig : new()
    {
        if (string.IsNullOrWhiteSpace(yamlContent))
        {
            throw new ArgumentException("YAML content cannot be null or empty.", nameof(yamlContent));
        }

        IDeserializer deserializer = CreateDeserializer();

        TConfig? config;
        try
        {
            config = deserializer.Deserialize<TConfig>(YamlVariableResolver.Resolve(yamlContent));
        }
        catch (YamlException ex)
        {
            throw new ArgumentException("YAML deserialization failed.", nameof(yamlContent), ex);
        }

        return config ?? new TConfig();
    }

    /// <summary>Builds the deserializer every load shares, including the converters hand-written types need.</summary>
    /// <returns>A configured deserializer.</returns>
    /// <remarks>
    /// Immutable configuration types validate their values in a constructor, which the general object mapper cannot
    /// call, so each one contributes a converter here. Registering them in one place is what keeps a file loaded
    /// through any of the four entry points from behaving differently.
    /// </remarks>
    private static IDeserializer CreateDeserializer() => new DeserializerBuilder()
        .WithNamingConvention(CamelCaseNamingConvention.Instance)
        .WithTypeConverter(new EvolutionDescriptorYamlConverter())
        .IgnoreUnmatchedProperties()
        .Build();

    /// <summary>Builds a serializer that writes what <see cref="CreateDeserializer"/> can read back.</summary>
    /// <returns>A configured serializer.</returns>
    /// <remarks>
    /// Round-tripping is the property that makes a configuration file trustworthy: a run's settings can be written
    /// out, committed, and loaded again to reproduce that run. The serializer therefore has to share the
    /// deserializer's naming convention and converters rather than being built ad hoc at each call site.
    /// </remarks>
    private static ISerializer CreateSerializer() => new SerializerBuilder()
        .WithNamingConvention(CamelCaseNamingConvention.Instance)
        .WithTypeConverter(new EvolutionDescriptorYamlConverter())
        .Build();

    /// <summary>Writes a configuration object as YAML that <see cref="LoadFromString"/> reads back unchanged.</summary>
    /// <typeparam name="TConfig">The configuration type to serialize.</typeparam>
    /// <param name="config">The configuration to write.</param>
    /// <returns>The YAML text.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="config"/> is <c>null</c>.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> Use this to capture the exact settings a run used, so the run can be repeated
    /// later or reviewed by someone else. Note that any <c>${NAME}</c> reference in the original file has already
    /// been replaced by its value at load time, so a file written this way contains the resolved values - keep it out
    /// of source control if those values include a secret.</para>
    /// </remarks>
    public static string SaveToString<TConfig>(TConfig config)
    {
        if (config is null) throw new ArgumentNullException(nameof(config));
        return CreateSerializer().Serialize(config);
    }
}
