using System.Reflection;
using AiDotNet.Configuration;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Configuration;

/// <summary>
/// Regression guard for the YamlConfigSourceGenerator accessibility bug: the implementation finder
/// only rejected NESTED non-public types, so a TOP-LEVEL internal type in any referenced assembly
/// that implemented a broad interface (e.g. <c>FormattedLogValues : IReadOnlyList&lt;...&gt;</c> in
/// Microsoft.Extensions.Logging matching an <c>IReadOnlyList&lt;T&gt;</c> Configure parameter) was
/// emitted into the generated registry as a <c>typeof(...)</c> reference the registry could not
/// compile against (CS0122). The generator now checks the full containment chain is public; this
/// test pins the contract on whatever the registry actually emitted.
/// </summary>
public class YamlTypeRegistryAccessibilityTests
{
    [Fact]
    public void Every_registered_implementation_type_is_publicly_visible()
    {
        var registryType = typeof(YamlModelConfig).Assembly
            .GetTypes()
            .FirstOrDefault(t => t.Name.StartsWith("YamlTypeRegistry", StringComparison.Ordinal));

        Assert.True(registryType is not null, "Generated YamlTypeRegistry type not found in the AiDotNet assembly.");

        // Walk every static field/property of the registry that holds Type values (the per-section
        // name → implementation maps) and assert each registered Type is visible to external code.
        var offenders = new List<string>();
        foreach (var field in registryType!.GetFields(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static))
        {
            object? value;
            try
            {
                value = field.GetValue(null);
            }
            catch (Exception)
            {
                continue; // a throwing initializer would itself fail other registry tests
            }

            if (value is not System.Collections.IDictionary dict)
            {
                continue;
            }

            foreach (var entryValue in dict.Values)
            {
                if (entryValue is Type implType && !implType.IsVisible)
                {
                    offenders.Add($"{field.Name}: {implType.FullName}");
                }
            }
        }

        Assert.True(offenders.Count == 0,
            "Generated YamlTypeRegistry references non-public types (generator accessibility filter regressed): " +
            string.Join("; ", offenders));
    }
}
