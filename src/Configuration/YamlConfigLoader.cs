using YamlDotNet.Serialization;
using YamlDotNet.Serialization.NamingConventions;

namespace AiDotNet.Configuration;

/// <summary>
/// Loads and validates YAML configuration files into <see cref="YamlModelConfig"/> instances.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> This class reads a YAML file from disk (or a YAML string)
/// and converts it into a structured C# object that the builder can use. YAML uses
/// camelCase property names (e.g., <c>timeSeriesModel</c>, <c>gpuAcceleration</c>).</para>
///
/// <para><b>Example usage:</b></para>
/// <code>
/// var config = YamlConfigLoader.LoadFromFile("training-recipe.yaml");
/// // or
/// var config = YamlConfigLoader.LoadFromString(yamlContent);
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

        var config = deserializer.Deserialize<YamlModelConfig>(yamlContent);

        return config ?? new YamlModelConfig();
    }
}
