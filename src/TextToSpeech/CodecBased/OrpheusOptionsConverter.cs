using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>Preserves native release dimensions when restoring older shared option metadata.</summary>
internal sealed class OrpheusOptionsConverter : JsonConverter<OrpheusTTSOptions>
{
    public override bool CanWrite => false;

    public override OrpheusTTSOptions? ReadJson(JsonReader reader, Type objectType, OrpheusTTSOptions? existingValue,
        bool hasExistingValue, JsonSerializer serializer)
    {
        if (reader.TokenType == JsonToken.Null) return null;
        var data = JObject.Load(reader);
        PreferNative("NumHeads", "Heads");
        PreferNative("NumDecoderLayers", "Layers");
        // Older full configurations carried the generic six-encoder default, which never built an encoder.
        // A partial configuration explicitly requesting an encoder remains invalid.
        var encoder = data.GetValue("NumEncoderLayers", StringComparison.OrdinalIgnoreCase);
        if (data.GetValue("Layers", StringComparison.OrdinalIgnoreCase) is not null
            && encoder?.Type == JTokenType.Integer && (int)encoder == 6)
            encoder.Replace(new JValue(0));
        var options = hasExistingValue && existingValue is not null ? existingValue
            : serializer.ContractResolver.ResolveContract(objectType).DefaultCreator?.Invoke() as OrpheusTTSOptions
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

    public override void WriteJson(JsonWriter writer, OrpheusTTSOptions? value, JsonSerializer serializer) =>
        throw new NotSupportedException("Serialization uses the default options contract.");
}
