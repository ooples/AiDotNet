namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>Options for <see cref="FishSpeechV15{T}"/>.</summary>
/// <remarks>
/// <para>The defaults are Fish Speech 1.5's (<c>fishaudio/fish-speech-1.5</c>): a Dual-AR transformer of 24 slow and 4 fast
/// layers, 1024 wide, 16 query and 2 key-value heads, a 102,048-token tiktoken vocabulary whose last 1,024 tokens are the
/// semantic tokens, an 8,192-token context; Firefly-GAN at 44.1 kHz with 8 codebooks of 1,024 codes at about 21 frames a
/// second. Training follows the paper's recipe (<see cref="FishSpeechRecipe.Paper"/>); the release's fine-tuning
/// recipe is <see cref="FishSpeechFamilyOptions.ApplyRecipe"/>'s other choice.</para>
/// <para><b>For Beginners:</b> These settings size Fish Speech 1.5 and control how it speaks and trains; the defaults
/// reproduce the released model.</para>
/// </remarks>
public class FishSpeechV15Options : FishSpeechFamilyOptions
{
    /// <summary>Creates the released Fish Speech 1.5 configuration.</summary>
    public FishSpeechV15Options()
    {
        VocabSize = 102048;
        MaxSequenceLength = 8192;
        DropoutRate = 0.0;
        NormalizeChineseText = false;
        ApplyRecipe(FishSpeechRecipe.Paper);
    }
}
