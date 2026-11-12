using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace AiDotNet.TextToSpeech.Classic;

/// <summary>Restores the native geometry of the two classical flow/diffusion acoustic models.</summary>
internal sealed class ClassicFlowOptionsConverter : JsonConverter
{
    public override bool CanWrite => false;

    public override bool CanConvert(Type objectType)
        => typeof(GradTTSOptions).IsAssignableFrom(objectType) || typeof(GlowTTSOptions).IsAssignableFrom(objectType);

    public override object? ReadJson(JsonReader reader, Type objectType, object? existingValue, JsonSerializer serializer)
    {
        if (reader.TokenType == JsonToken.Null) return null;
        var data = JObject.Load(reader);
        void NativeWins(string native, string alias)
        {
            if (data.GetValue(native, StringComparison.OrdinalIgnoreCase) is null) return;
            foreach (var property in data.Properties().Where(p => p.Name.Equals(alias, StringComparison.OrdinalIgnoreCase)).ToArray())
                property.Remove();
        }
        NativeWins("HiddenDim", "EncoderDim");
        if (typeof(GlowTTSOptions).IsAssignableFrom(objectType))
        {
            NativeWins("DecoderHiddenChannels", "DecoderDim");
            NativeWins("NumFlowBlocks", "NumDecoderLayers");
        }
        else NativeWins("DecoderDimMultipliers", "NumDecoderLayers");
        var options = existingValue ?? serializer.ContractResolver.ResolveContract(objectType).DefaultCreator?.Invoke()
            ?? throw new JsonSerializationException($"Cannot construct {objectType.FullName} options.");
        using var normalized = data.CreateReader();
        serializer.Populate(normalized, options);
        return options;
    }

    public override void WriteJson(JsonWriter writer, object? value, JsonSerializer serializer)
        => throw new NotSupportedException("Serialization uses the default options contract.");
}
