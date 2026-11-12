using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>Loads canonical codec settings before shared aliases, preserving explicit native fields in either order.</summary>
internal sealed class SpearOptionsConverter : JsonConverter
{
    public override bool CanWrite => false;
    public override bool CanConvert(Type objectType) => typeof(SPEARTTSOptions).IsAssignableFrom(objectType);

    public override object? ReadJson(JsonReader reader, Type objectType, object? existingValue, JsonSerializer serializer)
    {
        if (reader.TokenType == JsonToken.Null) return null;
        var data = JObject.Load(reader);
        var codec = data.GetValue("SoundStream", StringComparison.OrdinalIgnoreCase) as JObject;
        var aliases = new JObject();
        foreach (var name in new[] { "SampleRate", "HopSize", "CodebookSize", "NumCodebooks" })
        {
            var property = data.Properties().FirstOrDefault(p => p.Name.Equals(name, StringComparison.OrdinalIgnoreCase));
            if (property is null) continue;
            string native = name == "HopSize" ? "Ratios" : name == "NumCodebooks" ? "TargetBandwidthKbps" : name;
            if (codec?.GetValue(native, StringComparison.OrdinalIgnoreCase) is null) aliases[name] = property.Value.DeepClone();
            property.Remove();
        }
        var options = existingValue as SPEARTTSOptions
            ?? serializer.ContractResolver.ResolveContract(objectType).DefaultCreator?.Invoke() as SPEARTTSOptions
            ?? throw new JsonSerializationException($"Cannot construct {objectType.FullName} options.");
        using (var normalized = data.CreateReader()) serializer.Populate(normalized, options);
        if (options.SoundStream is null) throw new JsonSerializationException("SPEAR-TTS requires a SoundStream configuration.");
        using (var normalizedAliases = aliases.CreateReader()) serializer.Populate(normalizedAliases, options);
        // Changing a geometry alias preserves the selected levels; an explicit native bandwidth remains authoritative.
        if (codec?.GetValue("TargetBandwidthKbps", StringComparison.OrdinalIgnoreCase) is { } bandwidth)
            options.SoundStream.TargetBandwidthKbps = bandwidth.ToObject<double>(serializer);
        if (codec?.GetValue("NumQuantizers", StringComparison.OrdinalIgnoreCase) is { } capacity)
        {
            int maximum = capacity.ToObject<int>(serializer);
            if (aliases["NumCodebooks"] is { } selected && selected.ToObject<int>(serializer) > maximum)
                throw new JsonSerializationException("The selected SPEAR-TTS levels exceed the explicitly configured codec capacity.");
            options.SoundStream.NumQuantizers = maximum;
        }
        return options;
    }

    public override void WriteJson(JsonWriter writer, object? value, JsonSerializer serializer)
        => throw new NotSupportedException("Serialization uses the default options contract.");
}
