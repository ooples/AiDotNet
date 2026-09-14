using System.Reflection;
using System.Text;
using AiDotNet.Configuration;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using Newtonsoft.Json.Serialization;

namespace AiDotNet.Evolve.Cli;

/// <summary>Preserves configuration structure while replacing private/runtime bindings with explicit placeholders.</summary>
internal static class RunConfigurationTemplate
{
    internal sealed record Export(byte[] Json, string[] ProhibitedValues);

    internal static Export Create(YamlModelConfig config, bool includeProgramContent)
    {
        var serializer = JsonSerializer.Create(new JsonSerializerSettings
        {
            NullValueHandling = NullValueHandling.Ignore, ReferenceLoopHandling = ReferenceLoopHandling.Error,
            ContractResolver = new ConfigurationContractResolver()
        });
        var root = JObject.FromObject(config, serializer);
        var prohibited = new HashSet<string>(StringComparer.Ordinal);
        CollectPrivateValues(root, prohibited);
        var bindings = new List<string>();
        Redact(root, string.Empty, prohibited, bindings, includeProgramContent);
        if (root["ProgramEvolution"] is JObject program && config.ProgramEvolution is { } options)
        {
            foreach (var (name, present) in new[]
            {
                ("CustomVariation", options.CustomVariation is not null),
                ("CustomFitnessEvaluator", options.CustomFitnessEvaluator is not null),
                ("ResourceAccounting", options.ResourceAccounting is not null)
            })
            {
                if (!present) continue;
                program[name] = new JObject { ["RequiresBinding"] = true, ["Kind"] = "caller-owned runtime service" };
                bindings.Add("/ProgramEvolution/" + name);
            }
        }
        byte[] json = RunRecord.Json(new
        {
            SchemaVersion = 2, Configuration = root, RequiredBindings = bindings,
            ProgramContentIncluded = includeProgramContent,
            Scope = "configuration template; restore required private/runtime bindings before replay; placeholders must never be passed to a provider"
        });
        if (json.Length > 128 * 1024) throw new InvalidDataException("Configuration evidence exceeds its metadata bound.");
        return new Export(json, prohibited.ToArray());
    }

    private static bool IsPrivate(string name) =>
        !name.EndsWith("DirectoryName", StringComparison.OrdinalIgnoreCase) &&
        (name.Equals("Params", StringComparison.OrdinalIgnoreCase) ||
        name.Equals("RunId", StringComparison.OrdinalIgnoreCase) ||
        name.IndexOf("password", StringComparison.OrdinalIgnoreCase) >= 0 ||
        name.IndexOf("secret", StringComparison.OrdinalIgnoreCase) >= 0 ||
        name.IndexOf("credential", StringComparison.OrdinalIgnoreCase) >= 0 ||
        name.IndexOf("apikey", StringComparison.OrdinalIgnoreCase) >= 0 ||
        name.IndexOf("api_key", StringComparison.OrdinalIgnoreCase) >= 0 ||
        name.IndexOf("authorization", StringComparison.OrdinalIgnoreCase) >= 0 ||
        name.EndsWith("Token", StringComparison.OrdinalIgnoreCase) ||
        name.EndsWith("Key", StringComparison.OrdinalIgnoreCase) ||
        name.IndexOf("environment", StringComparison.OrdinalIgnoreCase) >= 0 ||
        name.EndsWith("Path", StringComparison.OrdinalIgnoreCase) ||
        name.IndexOf("Directory", StringComparison.OrdinalIgnoreCase) >= 0);

    private static bool ContainsCredentials(string name) => IsPrivate(name) &&
        !name.Equals("RunId", StringComparison.OrdinalIgnoreCase) &&
        !name.EndsWith("Path", StringComparison.OrdinalIgnoreCase) &&
        name.IndexOf("Directory", StringComparison.OrdinalIgnoreCase) < 0;

    private static bool IsProgramContent(string name) =>
        name is "SeedPrograms" or "ProgramSources" or "EvaluatorScript" or "TestCases" or "SystemMessage" or "TaskDescription" ||
        name.IndexOf("Prompt", StringComparison.OrdinalIgnoreCase) >= 0;

    private static void CollectPrivateValues(JToken item, HashSet<string> values)
    {
        if (item is JObject map)
            foreach (var property in map.Properties())
            {
                if (ContainsCredentials(property.Name)) CollectStrings(property.Value, values);
                else CollectPrivateValues(property.Value, values);
            }
        else if (item is JArray array)
            foreach (var child in array) CollectPrivateValues(child, values);
    }

    private static void CollectStrings(JToken item, HashSet<string> values)
    {
        if (item.Type == JTokenType.String && item.Value<string>() is { Length: > 0 } text) values.Add(text);
        if (item is JContainer container)
            foreach (var child in container.Children()) CollectStrings(child, values);
    }

    private static void Redact(JToken item, string path, HashSet<string> prohibited, List<string> bindings, bool includeContent)
    {
        if (item is JObject map)
            foreach (var property in map.Properties().ToArray())
            {
                string childPath = path + "/" + property.Name.Replace("~", "~0").Replace("/", "~1");
                bool containsKnownPrivateValue = property.Value.Type == JTokenType.String &&
                    prohibited.Any(value => property.Value.Value<string>()!.Contains(value, StringComparison.Ordinal));
                if (IsPrivate(property.Name) || (!includeContent && IsProgramContent(property.Name)) || containsKnownPrivateValue)
                {
                    bindings.Add(childPath);
                    // Do not publish a password/API-key hash that permits offline guessing of weak credentials.
                    property.Value = IsPrivate(property.Name) || containsKnownPrivateValue
                        ? new JObject { ["RequiresBinding"] = true }
                        : new JObject { ["RequiresBinding"] = true, ["ValueSha256"] = RunRecord.Hash(property.Value.ToString(Formatting.None)) };
                }
                else Redact(property.Value, childPath, prohibited, bindings, includeContent);
            }
        else if (item is JArray array)
            for (int i = 0; i < array.Count; i++) Redact(array[i], path + "/" + i, prohibited, bindings, includeContent);
    }

    private sealed class ConfigurationContractResolver : DefaultContractResolver
    {
        protected override JsonProperty CreateProperty(MemberInfo member, MemberSerialization serialization)
        {
            var property = base.CreateProperty(member, serialization);
            // Runtime objects are caller-owned services, not a portable configuration graph. Never invoke their getters.
            if (member.Name is "CustomVariation" or "CustomFitnessEvaluator" or "ResourceAccounting")
                property.ShouldSerialize = _ => false;
            return property;
        }
    }
}
