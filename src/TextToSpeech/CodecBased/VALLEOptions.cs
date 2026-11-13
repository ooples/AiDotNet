using AiDotNet.Audio.Generation;

namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>Options for VALL-E (Wang et al. 2023, "Neural Codec Language Models are Zero-Shot Text to Speech
/// Synthesizers").</summary>
/// <remarks>
/// <para>
/// The defaults are the paper's (§5.1): EnCodec at 6 kbps (eight codebooks of 1024 codes, 75 frames a second, 24 kHz),
/// an AR and a NAR Transformer each of 12 layers, 16 heads, width 1024, feed-forward 4096 and dropout 0.1, trained with
/// AdamW warmed up over 32,000 updates to 5e-4 and decayed linearly over 800,000 updates; the NAR's training prompt is
/// a random 3-second segment of the same utterance. The paper converts text with a Kaldi ASR model's phoneme
/// alignments; this model reads espeak-style phonemes (<see cref="FrontEnd.EnglishG2P"/>), as the reference
/// reproduction (lifeiteng/vall-e) does, and takes the rest of what the paper leaves open from it: the phoneme table
/// (512 embedding rows), sinusoidal positions with a learned scale in the AR model, pre-norm layers, the NAR heads
/// sharing the acoustic embeddings, and sampling with temperature 1 and no top-k cut.
/// </para>
/// <para><b>For Beginners:</b> These options set the sizes of VALL-E's two networks, its codec and how it is trained
/// and sampled. The defaults reproduce the paper.</para>
/// </remarks>
public class VALLEOptions : VallEModelOptions
{
    /// <summary>Creates the paper's options.</summary>
    public VALLEOptions()
    {
    }

    /// <summary>Creates a copy of <paramref name="other"/>.</summary>
    public VALLEOptions(VALLEOptions other)
        : base(other ?? throw new ArgumentNullException(nameof(other)))
    {
        PromptSeconds = other.PromptSeconds;
    }


    /// <summary>Length of the NAR's training prompt, in seconds (3, §5.1).</summary>
    public double PromptSeconds { get; set; } = 3.0;


    /// <summary>EnCodec's released 24 kHz model at 6 kbps: eight codebooks.</summary>
    public static EnCodecOptions CreatePaperCodec()
    {
        var codec = EnCodecOptions.OfficialCheckpoint24kHz();
        codec.TargetBandwidthKbps = 6.0;
        return codec;
    }
}
