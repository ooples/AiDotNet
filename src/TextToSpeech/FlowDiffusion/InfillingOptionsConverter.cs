using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace AiDotNet.TextToSpeech.FlowDiffusion;

/// <summary>Native backbone fields take precedence over obsolete independent shared layer counts.</summary>
internal sealed class InfillingOptionsConverter : JsonConverter
{
    public override bool CanWrite => false;
    public override bool CanConvert(Type objectType) => typeof(F5TTSOptions).IsAssignableFrom(objectType)
        || typeof(E2TTSOptions).IsAssignableFrom(objectType);

    public override object? ReadJson(JsonReader reader, Type objectType, object? existingValue, JsonSerializer serializer)
    {
        if (reader.TokenType == JsonToken.Null) return null;
        var data = JObject.Load(reader);
        bool f5 = typeof(F5TTSOptions).IsAssignableFrom(objectType);
        bool nativeDepth = data.GetValue("NumLayers", StringComparison.OrdinalIgnoreCase) is not null;
        bool nativeText = data.GetValue("TextConvLayers", StringComparison.OrdinalIgnoreCase) is not null;
        foreach (var property in data.Properties().Where(p =>
            p.Name.Equals("NumDecoderLayers", StringComparison.OrdinalIgnoreCase) && nativeDepth
            || p.Name.Equals("NumEncoderLayers", StringComparison.OrdinalIgnoreCase) && (f5 ? nativeText : nativeDepth)).ToArray())
            property.Remove();
        var options = existingValue ?? serializer.ContractResolver.ResolveContract(objectType).DefaultCreator?.Invoke()
            ?? throw new JsonSerializationException($"Cannot construct {objectType.FullName} options.");
        using var normalized = data.CreateReader();
        serializer.Populate(normalized, options);
        return options;
    }

    public override void WriteJson(JsonWriter writer, object? value, JsonSerializer serializer)
        => throw new NotSupportedException("Serialization uses the default options contract.");
}
