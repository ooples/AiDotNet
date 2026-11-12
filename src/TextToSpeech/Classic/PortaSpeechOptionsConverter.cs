using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace AiDotNet.TextToSpeech.Classic;

/// <summary>Preserves native architecture fields over conflicting saved legacy aliases.</summary>
internal sealed class PortaSpeechOptionsConverter : JsonConverter
{
    public override bool CanWrite => false;
    public override bool CanConvert(Type objectType) => typeof(PortaSpeechOptions).IsAssignableFrom(objectType);

    public override object? ReadJson(JsonReader reader, Type objectType, object? existingValue, JsonSerializer serializer)
    {
        if (reader.TokenType == JsonToken.Null) return null;
        var data = JObject.Load(reader);
        foreach (var pair in new[]
        {
            (Native: "HiddenDim", Alias: "EncoderDim"),
            (Native: "GeneratorChannels", Alias: "DecoderDim"),
            (Native: "GeneratorDecoderLayers", Alias: "NumDecoderLayers"),
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
