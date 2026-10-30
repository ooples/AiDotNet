namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>Options for <see cref="FishSpeech{T}"/> (Fish Speech 1.4).</summary>
/// <remarks>
/// <para>The defaults are Fish Speech 1.4's (<c>fishaudio/fish-speech-1.4</c>): the same Dual-AR sizes as 1.5 over a
/// 32,000-token byte-level BPE vocabulary with one <c>&lt;|semantic|&gt;</c> token, a 4,096-token context and dropout 0.1;
/// the same Firefly-GAN. Text goes through the Chinese number normalizer as the 1.4 API does by default. Training follows
/// the release's fine-tuning recipe (<see cref="FishSpeechRecipe.ReleaseFineTune"/>).</para>
/// <para><b>For Beginners:</b> These settings size Fish Speech 1.4 and control how it speaks and trains; the defaults
/// reproduce the released model.</para>
/// </remarks>
public class FishSpeechOptions : FishSpeechFamilyOptions
{
    /// <summary>Creates the released Fish Speech 1.4 configuration.</summary>
    public FishSpeechOptions()
    {
        VocabSize = 32000;
        MaxSequenceLength = 4096;
        DropoutRate = 0.1;
        NormalizeChineseText = true;
        ApplyRecipe(FishSpeechRecipe.ReleaseFineTune);
    }
}
