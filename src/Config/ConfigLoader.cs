namespace AiDotNet.Config;

using System.IO;
using Newtonsoft.Json;

public static class ConfigLoader
{
    public static T LoadJson<T>(string path)
    {
        var json = File.ReadAllText(path);
        var settings = new JsonSerializerSettings
        {
            MissingMemberHandling = MissingMemberHandling.Ignore,
            NullValueHandling = NullValueHandling.Include
        };
        return JsonConvert.DeserializeObject<T>(json, settings)!;
    }

    public static T LoadYaml<T>(string path)
    {
        // Placeholder: YAML support can be added via a compatible library for net462.
        // For now, throw explicit guidance.
        throw new System.NotSupportedException("YAML parsing not enabled in this build. Use JSON or add a YAML parser compatible with net462.");
    }
}
