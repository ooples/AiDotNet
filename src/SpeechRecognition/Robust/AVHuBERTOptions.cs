using AiDotNet.Models.Options;
using AiDotNet.Onnx;

namespace AiDotNet.SpeechRecognition.Robust;

/// <summary>Options for AV-HuBERT: audio-visual masked multimodal cluster prediction, with a CTC projection.</summary>
/// <remarks>
/// <para>The defaults are the BASE model of Shi et al. 2022 (section 4, appendix B) with the released pre-training
/// configuration where the paper is silent. For LARGE set EncoderDim 1024, NumEncoderLayers 24, NumAttentionHeads 16 and
/// FeedForwardDim 4096.</para>
/// <para><b>For Beginners:</b> These options set the size of AV-HuBERT's encoder, how often pre-training hides parts of
/// the audio and the lip video, and how often it drops a whole modality.</para>
/// </remarks>
public class AVHuBERTOptions : ModelOptions
{
    /// <summary>Initializes a new instance with default values.</summary>
    public AVHuBERTOptions() { }

    /// <summary>Initializes a new instance by copying from another instance.</summary>
    /// <param name="other">The options instance to copy from.</param>
    /// <exception cref="ArgumentNullException">Thrown when other is null.</exception>
    public AVHuBERTOptions(AVHuBERTOptions other)
    {
        if (other == null)
            throw new ArgumentNullException(nameof(other));

        SampleRate = other.SampleRate;
        MaxAudioLengthSeconds = other.MaxAudioLengthSeconds;
        FilterbankBins = other.FilterbankBins;
        LipSize = other.LipSize;
        EncoderDim = other.EncoderDim;
        NumEncoderLayers = other.NumEncoderLayers;
        NumAttentionHeads = other.NumAttentionHeads;
        FeedForwardDim = other.FeedForwardDim;
        DropoutRate = other.DropoutRate;
        LayerDrop = other.LayerDrop;
        ModalityDropout = other.ModalityDropout;
        AudioDropout = other.AudioDropout;
        AudioMaskProbability = other.AudioMaskProbability;
        AudioMaskLength = other.AudioMaskLength;
        VideoMaskProbability = other.VideoMaskProbability;
        VideoMaskLength = other.VideoMaskLength;
        FinalDim = other.FinalDim;
        NumClusters = other.NumClusters;
        LogitTemperature = other.LogitTemperature;
        FeaturePenaltyWeight = other.FeaturePenaltyWeight;
        VocabSize = other.VocabSize;
        MaxTextLength = other.MaxTextLength;
        ModelPath = other.ModelPath;
        OnnxOptions = new OnnxModelOptions(other.OnnxOptions);
        Language = other.Language;
    }

    /// <summary>Audio sample rate in Hz.</summary>
    public int SampleRate { get; set; } = 16000;

    /// <summary>Longest audio a transcription covers, in seconds; longer input is truncated.</summary>
    public int MaxAudioLengthSeconds { get; set; } = 30;

    /// <summary>Log filterbank bins per 10 ms frame; four frames are stacked into one 104-dimensional 25 Hz frame (the released audio features).</summary>
    public int FilterbankBins { get; set; } = 26;

    /// <summary>Side of the grayscale mouth crop in pixels (appendix B.2: 88 x 88).</summary>
    public int LipSize { get; set; } = 88;

    /// <summary>Transformer width (section 4: BASE 768, LARGE 1024).</summary>
    public int EncoderDim { get; set; } = 768;

    /// <summary>Transformer layers (section 4: BASE 12, LARGE 24).</summary>
    public int NumEncoderLayers { get; set; } = 12;

    /// <summary>Attention heads (section 4: BASE 12, LARGE 16).</summary>
    public int NumAttentionHeads { get; set; } = 12;

    /// <summary>Feed-forward width (section 4: BASE 3072, LARGE 4096).</summary>
    public int FeedForwardDim { get; set; } = 3072;

    /// <summary>Dropout in the Transformer and on the fused input (appendix B.3: 0.1).</summary>
    public double DropoutRate { get; set; } = 0.1;

    /// <summary>Probability of skipping each Transformer layer in training (0.05 in the released pre-training configuration; appendix B.3 states 0.1).</summary>
    public double LayerDrop { get; set; } = 0.05;

    /// <summary>Probability p_m complement: with probability ModalityDropout one modality is dropped for the whole sequence (appendix B.4: p_m = 0.5).</summary>
    public double ModalityDropout { get; set; } = 0.5;

    /// <summary>When a modality is dropped, the probability p_a that it is the audio (appendix B.4: 0.5).</summary>
    public double AudioDropout { get; set; } = 0.5;

    /// <summary>Proportion of audio frames that start a masked span in pre-training (appendix B.4: p = 8 per 10 frames).</summary>
    public double AudioMaskProbability { get; set; } = 0.8;

    /// <summary>Masked audio span length in frames (appendix B.4: 10).</summary>
    public int AudioMaskLength { get; set; } = 10;

    /// <summary>Proportion of video frames that start a masked span in pre-training (0.3 in the released configuration; appendix B.4 states 6 per 5 frames as p/l).</summary>
    public double VideoMaskProbability { get; set; } = 0.3;

    /// <summary>Masked video span length in frames (appendix B.4: 5).</summary>
    public int VideoMaskLength { get; set; } = 5;

    /// <summary>Width of the projection compared with the cluster embeddings (256 in the released configuration).</summary>
    public int FinalDim { get; set; } = 256;

    /// <summary>Cluster targets of the masked prediction (appendix B.4: 2000 in the fifth iteration).</summary>
    public int NumClusters { get; set; } = 2000;

    /// <summary>Temperature of the cosine cluster logits (fairseq HuBERT's logit_temp: 0.1).</summary>
    public double LogitTemperature { get; set; } = 0.1;

    /// <summary>Weight of the mean squared fused features in the pre-training loss (the released configuration's loss_weights: 10).</summary>
    public double FeaturePenaltyWeight { get; set; } = 10.0;

    /// <summary>CTC vocabulary size, blank (token 0) included (appendix B.4: 46 phone units for CTC); tokens above 0 decode as Unicode code points.</summary>
    public int VocabSize { get; set; } = 46;

    /// <summary>Most tokens a transcription decodes.</summary>
    public int MaxTextLength { get; set; } = 512;

    /// <summary>Path to an exported ONNX model, when one is used.</summary>
    public string? ModelPath { get; set; }

    /// <summary>ONNX runtime options.</summary>
    public OnnxModelOptions OnnxOptions { get; set; } = new();

    /// <summary>Language reported for transcriptions.</summary>
    public string Language { get; set; } = "en";

    // The network these options describe.
    internal AvHubertConfiguration ToConfiguration() => new()
    {
        SampleRate = SampleRate, FilterbankBins = FilterbankBins, LipSize = LipSize, EncoderDim = EncoderDim,
        Layers = NumEncoderLayers, Heads = NumAttentionHeads, FeedForwardDim = FeedForwardDim, Dropout = DropoutRate,
        LayerDrop = LayerDrop, ModalityDropout = ModalityDropout, AudioDropout = AudioDropout,
        AudioMaskProbability = AudioMaskProbability, AudioMaskLength = AudioMaskLength,
        VideoMaskProbability = VideoMaskProbability, VideoMaskLength = VideoMaskLength, FinalDim = FinalDim,
        Clusters = NumClusters, LogitTemperature = LogitTemperature, FeaturePenaltyWeight = FeaturePenaltyWeight,
        VocabSize = VocabSize,
    };
}
