using AiDotNet.TextToSpeech.CodecBased;

namespace AiDotNet.TextToSpeech.VoiceCloning;

/// <summary>Base options for voice cloning TTS models.</summary>
public class VoiceCloningOptions : CodecTtsOptions
{
    public VoiceCloningOptions()
    {
        SpeakerEmbeddingDim = 256;
        NumEncoderLayers = 6;
        NumHeads = 8;
        DropoutRate = 0.1;
    }

    public double MinReferenceDurationSec { get; set; } = 3.0;
}
