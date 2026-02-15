namespace AiDotNet.Configuration;

/// <summary>
/// Marks an interface or abstract base class as discoverable by the YAML configuration system.
/// The source generator will automatically find all concrete implementations and register them
/// in the YAML type registry under the specified section name.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> Place this attribute on any interface or abstract class whose
/// implementations should be configurable via YAML files. The generator will scan the assembly
/// for all concrete types that implement/extend the marked type and make them available
/// for YAML-based configuration.</para>
/// <para>
/// This provides the same YAML discoverability as adding a <c>Configure*()</c> method to
/// <c>AiModelBuilder</c>, but without requiring a builder method. Use this when the type
/// is already used through other APIs and doesn't need its own Configure method.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// [YamlConfigurable("ChatModel")]
/// public interface IChatModel&lt;T&gt; : ILanguageModel&lt;T&gt; { }
///
/// // In YAML:
/// // chatModel:
/// //   type: AnthropicChatModel
/// //   params:
/// //     apiKey: "..."
/// </code>
/// </example>
[AttributeUsage(AttributeTargets.Interface | AttributeTargets.Class, AllowMultiple = false, Inherited = false)]
public sealed class YamlConfigurableAttribute : Attribute
{
    /// <summary>
    /// Gets the YAML section name for this type family.
    /// </summary>
    public string SectionName { get; }

    /// <summary>
    /// Initializes a new instance of the <see cref="YamlConfigurableAttribute"/> class.
    /// </summary>
    /// <param name="sectionName">
    /// The section name used in YAML configuration files and the type registry.
    /// For example, "ChatModel" would create a <c>chatModel:</c> section in YAML.
    /// </param>
    public YamlConfigurableAttribute(string sectionName)
    {
        SectionName = sectionName;
    }
}
