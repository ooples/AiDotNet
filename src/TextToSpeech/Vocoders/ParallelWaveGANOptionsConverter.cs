using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace AiDotNet.TextToSpeech.Vocoders;

/// <summary>Retains explicit native conditioning scales over legacy shared hops in either JSON order.</summary>
internal sealed class ParallelWaveGANOptionsConverter : JsonConverter
{
    public override bool CanWrite => false;
    public override bool CanConvert(Type objectType) => typeof(ParallelWaveGANOptions).IsAssignableFrom(objectType);

    public override object? ReadJson(JsonReader reader, Type objectType, object? existingValue, JsonSerializer serializer)
    {
        if (reader.TokenType == JsonToken.Null) return null;
        var data = JObject.Load(reader);
        if (data.GetValue("UpsampleRates", StringComparison.OrdinalIgnoreCase) is not null)
            foreach (var alias in data.Properties().Where(p => p.Name.Equals("HopSize", StringComparison.OrdinalIgnoreCase)).ToArray())
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
