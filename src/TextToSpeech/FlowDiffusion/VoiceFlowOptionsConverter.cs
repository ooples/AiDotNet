using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace AiDotNet.TextToSpeech.FlowDiffusion;

/// <summary>Preserves the independently configured encoder and flow-decoder widths in legacy options.</summary>
internal sealed class VoiceFlowOptionsConverter : JsonConverter
{
    public override bool CanWrite => false;
    public override bool CanConvert(Type objectType) => typeof(VoiceFlowOptions).IsAssignableFrom(objectType);

    public override object? ReadJson(JsonReader reader, Type objectType, object? existingValue, JsonSerializer serializer)
    {
        if (reader.TokenType == JsonToken.Null) return null;
        var data = JObject.Load(reader);
        // Old DecoderDim aliased the encoder's HiddenDim. FlowDim was the true U-Net width.
        // Full legacy configurations retain the native width; alias-only configurations still work.
        if (data.GetValue("FlowDim", StringComparison.OrdinalIgnoreCase) is not null)
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
