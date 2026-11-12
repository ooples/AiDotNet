using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace AiDotNet.TextToSpeech.Vocoders;

/// <summary>Retains canonical native denoiser dimensions and strides over legacy shared aliases.</summary>
internal sealed class PriorGradOptionsConverter : JsonConverter
{
    public override bool CanWrite => false;
    public override bool CanConvert(Type objectType) => typeof(PriorGradOptions).IsAssignableFrom(objectType);

    public override object? ReadJson(JsonReader reader, Type objectType, object? existingValue, JsonSerializer serializer)
    {
        if (reader.TokenType == JsonToken.Null) return null;
        var data = JObject.Load(reader);
        foreach (var pair in new[] { ("UpsampleStrides", "HopSize"), ("ResChannels", "HiddenDim") })
            if (data.GetValue(pair.Item1, StringComparison.OrdinalIgnoreCase) is not null)
                foreach (var alias in data.Properties().Where(p => p.Name.Equals(pair.Item2, StringComparison.OrdinalIgnoreCase)).ToArray())
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
