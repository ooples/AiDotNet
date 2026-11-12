using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>Preserves saved native component dimensions over conflicting legacy shared aliases.</summary>
internal sealed class LlasaOptionsConverter : JsonConverter
{
    public override bool CanWrite => false;
    public override bool CanConvert(Type objectType) => typeof(LlasaOptions).IsAssignableFrom(objectType);

    public override object? ReadJson(JsonReader reader, Type objectType, object? existingValue, JsonSerializer serializer)
    {
        if (reader.TokenType == JsonToken.Null) return null;
        var data = JObject.Load(reader);
        foreach (var pair in new[]
        {
            (Native: "Heads", Alias: "NumHeads"),
            (Native: "Layers", Alias: "NumDecoderLayers"),
            (Native: "SemanticLayer", Alias: "NumEncoderLayers"),
        })
            if (data.GetValue(pair.Native, StringComparison.OrdinalIgnoreCase) is not null)
                foreach (var alias in data.Properties().Where(p => p.Name.Equals(pair.Alias, StringComparison.OrdinalIgnoreCase)).ToArray())
                    alias.Remove();
        var options = existingValue ?? serializer.ContractResolver.ResolveContract(objectType).DefaultCreator?.Invoke()
            ?? throw new JsonSerializationException($"Cannot construct {objectType.FullName} options.");
        using var normalized = data.CreateReader();
        serializer.Populate(normalized, options);
        return options;
    }

    public override void WriteJson(JsonWriter writer, object? value, JsonSerializer serializer)
        => throw new NotSupportedException("Serialization uses the default options contract.");
}
