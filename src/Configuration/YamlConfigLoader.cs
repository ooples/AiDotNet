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
/// <para><b>Example usage:</b></para>
/// <code>
/// // Load AiModelBuilder config
/// var config = YamlConfigLoader.LoadFromFile("model-config.yaml");
///
/// // Load training recipe config
/// var recipe = YamlConfigLoader.LoadFromFile&lt;TrainingRecipeConfig&gt;("training-recipe.yaml");
///
/// // Load from string
/// var config = YamlConfigLoader.LoadFromString&lt;TrainingRecipeConfig&gt;(yamlContent);
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

        var deserializer = new DeserializerBuilder()
            .WithNamingConvention(CamelCaseNamingConvention.Instance)
            .IgnoreUnmatchedProperties()
            .Build();

        YamlModelConfig? config;
        try
        {
            config = deserializer.Deserialize<YamlModelConfig>(yamlContent);
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

        var deserializer = new DeserializerBuilder()
            .WithNamingConvention(CamelCaseNamingConvention.Instance)
            .IgnoreUnmatchedProperties()
            .Build();

        TConfig? config;
        try
        {
            config = deserializer.Deserialize<TConfig>(yamlContent);
        }
        catch (YamlException ex)
        {
            throw new ArgumentException("YAML deserialization failed.", nameof(yamlContent), ex);
        }

        return config ?? new TConfig();
    }
}
