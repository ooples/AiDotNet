using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace AiDotNet.TextToSpeech.Classic;

/// <summary>Restores native mel geometry before applying its shared decoder alias.</summary>
internal sealed class AcousticMelOptionsConverter : JsonConverter
{
    public override bool CanWrite => false;

    public override bool CanConvert(Type objectType) => typeof(AcousticModelOptions).IsAssignableFrom(objectType);

    public override object? ReadJson(JsonReader reader, Type objectType, object? existingValue, JsonSerializer serializer)
    {
        if (reader.TokenType == JsonToken.Null) return null;
        var data = JObject.Load(reader);
        // Earlier versions stored an independent, often stale DecoderDim beside the actual MelChannels.
        // Native geometry wins in either property order; an alias-only configuration still works.
        if (data.GetValue("MelChannels", StringComparison.OrdinalIgnoreCase) is not null)
            foreach (var property in data.Properties().Where(p => p.Name.Equals("DecoderDim", StringComparison.OrdinalIgnoreCase)).ToArray())
                property.Remove();
        var options = existingValue ?? serializer.ContractResolver.ResolveContract(objectType).DefaultCreator?.Invoke()
            ?? throw new JsonSerializationException($"Cannot construct {objectType.FullName} options.");
        using var normalized = data.CreateReader();
        serializer.Populate(normalized, options);
        return options;
    }

    public override void WriteJson(JsonWriter writer, object? value, JsonSerializer serializer) =>
        throw new NotSupportedException("Serialization uses the default options contract.");
}
