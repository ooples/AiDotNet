using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace AiDotNet.TextToSpeech.EndToEnd;

/// <summary>Restores native VITS geometry before applying shared option aliases.</summary>
internal sealed class VitsOptionsConverter : JsonConverter
{
    public override bool CanWrite => false;
    public override bool CanConvert(Type objectType) => typeof(VitsModelOptions).IsAssignableFrom(objectType);

    public override object? ReadJson(JsonReader reader, Type objectType, object? existingValue, JsonSerializer serializer)
    {
        if (reader.TokenType == JsonToken.Null) return null;
        var data = JObject.Load(reader);
        void Remove(string name)
        {
            foreach (var property in data.Properties().Where(p => p.Name.Equals(name, StringComparison.OrdinalIgnoreCase)).ToArray())
                property.Remove();
        }
        // Earlier full options stored an unrelated decoder-layer count (usually six).
        if (data.GetValue("UpsampleRates", StringComparison.OrdinalIgnoreCase) is not null) Remove("NumDecoderLayers");
        if (typeof(YourTTSOptions).IsAssignableFrom(objectType)
            && data.GetValue("SpeakerEncoderDim", StringComparison.OrdinalIgnoreCase) is not null)
            Remove("SpeakerEmbeddingDim");
        // Older Piper options stored an independent AddBlank=false beside the native InterspersePad=true.
        if (typeof(PiperOptions).IsAssignableFrom(objectType)
            && data.GetValue("InterspersePad", StringComparison.OrdinalIgnoreCase) is not null)
            Remove("AddBlank");
        var options = existingValue ?? serializer.ContractResolver.ResolveContract(objectType).DefaultCreator?.Invoke()
            ?? throw new JsonSerializationException($"Cannot construct {objectType.FullName} options.");
        using var normalized = data.CreateReader();
        serializer.Populate(normalized, options);
        return options;
    }

    public override void WriteJson(JsonWriter writer, object? value, JsonSerializer serializer) =>
        throw new NotSupportedException("Serialization uses the default options contract.");
}
