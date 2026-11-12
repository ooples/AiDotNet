using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace AiDotNet.TextToSpeech.Latest;

/// <summary>Preserves released backbone dimensions when loading old shared option metadata.</summary>
internal sealed class OuteOptionsConverter : JsonConverter
{
    public override bool CanWrite => false;
    public override bool CanConvert(Type objectType) => typeof(OuteTTSOptions).IsAssignableFrom(objectType);

    public override object? ReadJson(JsonReader reader, Type objectType, object? existingValue, JsonSerializer serializer)
    {
        if (reader.TokenType == JsonToken.Null) return null;
        var data = JObject.Load(reader);
        PreferNative("NumHeads", "Heads");
        PreferNative("NumDecoderLayers", "Layers");
        // Historical full configs contained the unused generic six-encoder default. Explicit partial requests remain invalid.
        var encoder = data.GetValue("NumEncoderLayers", StringComparison.OrdinalIgnoreCase);
        if (data.GetValue("Layers", StringComparison.OrdinalIgnoreCase) is not null
            && encoder?.Type == JTokenType.Integer && (int)encoder == 6)
            encoder.Replace(new JValue(0));
        var options = existingValue ?? serializer.ContractResolver.ResolveContract(objectType).DefaultCreator?.Invoke()
            ?? throw new JsonSerializationException($"Cannot construct {objectType.FullName} options.");
        using var normalized = data.CreateReader();
        serializer.Populate(normalized, options);
        return options;

        void PreferNative(string alias, string native)
        {
            if (data.GetValue(native, StringComparison.OrdinalIgnoreCase) is null) return;
            foreach (var property in data.Properties().Where(p => p.Name.Equals(alias, StringComparison.OrdinalIgnoreCase)).ToArray())
                property.Remove();
        }
    }

    public override void WriteJson(JsonWriter writer, object? value, JsonSerializer serializer)
        => throw new NotSupportedException("Serialization uses the default options contract.");
}
