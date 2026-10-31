namespace AiDotNet.TextToSpeech.CodecBased;

/// <summary>Which training recipe a Fish Speech model's optimizer follows.</summary>
public enum FishSpeechRecipe
{
    /// <summary>The paper's pre-training (Liao et al. 2024, Appendix A): AdamW (0.9, 0.98, 1e-8) at 5e-4, weight decay
    /// 0.01, cosine decay to a tenth after 2,000 warm-up steps over 500,000 steps.</summary>
    Paper,

    /// <summary>The release's fine-tuning (<c>text2semantic_finetune.yaml</c>): AdamW (0.9, 0.95, 1e-5) at 1e-4, no weight
    /// decay, constant after 10 warm-up steps.</summary>
    ReleaseFineTune,
}

/// <summary>
/// Options shared by the Fish Speech models: the Dual-AR transformer, the Firefly-GAN codec, the text front end,
/// decoding as the release's API runs it, and training. Each model's options set its release's values.
/// </summary>
public abstract class FishSpeechFamilyOptions : TtsModelOptions
{
    /// <summary>Sets the values both releases share: Firefly-GAN at 44.1 kHz and a 24-layer, 1024-wide Dual-AR.</summary>
    protected FishSpeechFamilyOptions()
    {
        SampleRate = 44100;
        MelChannels = 160;
        HopSize = 512;
        FftSize = 2048;
        HiddenDim = 1024;
        NumHeads = 16;
        NumDecoderLayers = 24;
    }

    // ---------------------------------------------------------------- Dual-AR transformer

    /// <summary>The slow transformer's key-value heads (<c>n_local_heads</c>; grouped-query attention).</summary>
    public int KeyValueHeads { get; set; } = 2;

    /// <summary>The SwiGLU feed-forward's width (<c>intermediate_size</c>).</summary>
    public int IntermediateSize { get; set; } = 4096;

    /// <summary>The fast transformer's layers (<c>n_fast_layer</c>).</summary>
    public int FastLayers { get; set; } = 4;

    /// <summary>The codec's codebooks, one per quantizer group (<c>num_codebooks</c>).</summary>
    public int NumCodebooks { get; set; } = 8;

    /// <summary>Each codebook's codes (<c>codebook_size</c>).</summary>
    public int CodebookSize { get; set; } = 1024;

    /// <summary>The transformer's context (<c>max_seq_len</c>).</summary>
    public int MaxSequenceLength { get; set; } = 8192;

    /// <summary>The rotary positions' base (<c>rope_base</c>).</summary>
    public double RopeBase { get; set; } = 1e6;

    /// <summary>RMSNorm's epsilon (<c>norm_eps</c>).</summary>
    public double NormEpsilon { get; set; } = 1e-6;

    // ---------------------------------------------------------------- Firefly-GAN

    /// <summary>The ConvNeXt encoder's blocks per stage.</summary>
    public int[] CodecEncoderDepths { get; set; } = { 3, 3, 9, 3 };

    /// <summary>The ConvNeXt encoder's widths per stage.</summary>
    public int[] CodecEncoderDims { get; set; } = { 128, 256, 384, 512 };

    /// <summary>The quantizer's input width (<c>input_dim</c>).</summary>
    public int CodecQuantizerDim { get; set; } = 512;

    /// <summary>The finite scalar quantizer's levels per dimension (8·5·5·5 = 1,000 codes).</summary>
    public int[] CodecLevels { get; set; } = { 8, 5, 5, 5 };

    /// <summary>The quantizer's strided downsampling before quantization (and upsampling after).</summary>
    public int[] CodecDownsampleFactors { get; set; } = { 2, 2 };

    /// <summary>The HiFi-GAN generator's upsampling strides.</summary>
    public int[] CodecUpsampleRates { get; set; } = { 8, 8, 2, 2, 2 };

    /// <summary>The HiFi-GAN generator's upsampling kernels.</summary>
    public int[] CodecUpsampleKernels { get; set; } = { 16, 16, 4, 4, 4 };

    /// <summary>The kernels of the residual blocks each ParallelBlock averages.</summary>
    public int[] CodecResBlockKernels { get; set; } = { 3, 7, 11 };

    /// <summary>The dilations of each residual block.</summary>
    public int[][] CodecResBlockDilations { get; set; } = { new[] { 1, 3, 5 }, new[] { 1, 3, 5 }, new[] { 1, 3, 5 } };

    /// <summary>The generator's initial channels (<c>upsample_initial_channel</c>).</summary>
    public int CodecChannels { get; set; } = 512;

    /// <summary>The generator's first convolution's kernel.</summary>
    public int CodecPreKernel { get; set; } = 13;

    /// <summary>The generator's last convolution's kernel.</summary>
    public int CodecPostKernel { get; set; } = 13;

    // ---------------------------------------------------------------- text

    /// <summary>The release's tokenizer file (1.5: <c>tokenizer.tiktoken</c>; 1.4: <c>tokenizer.json</c>). When null,
    /// <see cref="FishSpeechBase{T}.LoadReferenceWeights"/> takes it from the checkpoint directory.</summary>
    public string? TokenizerPath { get; set; }

    /// <summary>Whether text is read through the Chinese number normalizer first (the 1.4 API's
    /// <c>ChnNormedText</c>, its <c>normalize</c> default; the 1.5 engine ignores the flag).</summary>
    public bool NormalizeChineseText { get; set; }

    /// <summary>The text is split into segments of at most this many UTF-8 bytes, each generated in turn
    /// (<c>chunk_length</c>; 0 speaks the whole text as one segment).</summary>
    public int ChunkLength { get; set; } = 200;

    // ---------------------------------------------------------------- decoding

    /// <summary>The most frames generated per segment (<c>max_new_tokens</c>; 0 fills the context).</summary>
    public int MaxNewTokens { get; set; } = 1024;

    /// <summary>The sampling temperature, in (0, 2).</summary>
    public double Temperature { get; set; } = 0.7;

    /// <summary>Nucleus sampling's probability mass, in (0, 1].</summary>
    public double TopP { get; set; } = 0.7;

    /// <summary>The repetition penalty over the last 16 frames, in (0, 2).</summary>
    public double RepetitionPenalty { get; set; } = 1.2;

    /// <summary>The token budget <c>generate_long</c> keeps earlier segments within (<c>max_length</c>: what remains after
    /// 1,024 tokens for the new segment and the references).</summary>
    public int ContextLength { get; set; } = 4096;

    /// <summary>Whether every generated frame is speech: the token is sampled among the semantic tokens only (1.4: and each
    /// code among those a frame can store), so decoding runs to <see cref="MaxNewTokens"/>. The release samples freely
    /// and stops at <c>&lt;|im_end|&gt;</c>; this is for a model not yet trained to end its speech.</summary>
    public bool ConstrainToSemanticTokens { get; set; }

    /// <summary>The seed of decoding's draws.</summary>
    public int SamplingSeed { get; set; } = 42;

    // ---------------------------------------------------------------- training

    /// <summary>The seed of training's draws (dropout).</summary>
    public int TrainingSeed { get; set; } = 1234;

    /// <summary>The recipe <see cref="ApplyRecipe"/> last set.</summary>
    public FishSpeechRecipe Recipe { get; private set; }

    /// <summary>AdamW's first-moment decay.</summary>
    public double Beta1 { get; set; }

    /// <summary>AdamW's second-moment decay.</summary>
    public double Beta2 { get; set; }

    /// <summary>AdamW's epsilon.</summary>
    public double Epsilon { get; set; }

    /// <summary>Linear warm-up steps.</summary>
    public int WarmupSteps { get; set; }

    /// <summary>The schedule's length; with <see cref="CosineDecay"/>, the step at which the rate reaches its floor.</summary>
    public int TrainingSteps { get; set; }

    /// <summary>Whether the rate decays along a cosine after warm-up (otherwise it stays constant).</summary>
    public bool CosineDecay { get; set; }

    /// <summary>The cosine's floor as a fraction of the peak rate (<c>final_lr_ratio</c>).</summary>
    public double FinalLearningRateRatio { get; set; }

    /// <summary>The gradient norm is clipped to this (the release's <c>gradient_clip_val</c>).</summary>
    public double MaxGradientNorm { get; set; } = 1.0;

    /// <summary>Standard DPO's β (<see cref="FishSpeechBase{T}.TrainPreference"/>).</summary>
    public double DpoBeta { get; set; } = 0.1;

    /// <summary>The release's reference-free DPO's β (<c>dpo_beta</c>).</summary>
    public double ReferenceFreeDpoBeta { get; set; } = 0.2;

    /// <summary>Sets the optimizer's values to <paramref name="recipe"/>'s.</summary>
    public void ApplyRecipe(FishSpeechRecipe recipe)
    {
        Recipe = recipe;
        if (recipe == FishSpeechRecipe.Paper)
        {
            LearningRate = 5e-4;
            Beta1 = 0.9;
            Beta2 = 0.98;
            Epsilon = 1e-8;
            WeightDecay = 0.01;
            WarmupSteps = 2000;
            TrainingSteps = 500_000;
            CosineDecay = true;
            FinalLearningRateRatio = 0.1;
        }
        else
        {
            LearningRate = 1e-4;
            Beta1 = 0.9;
            Beta2 = 0.95;
            Epsilon = 1e-5;
            WeightDecay = 0.0;
            WarmupSteps = 10;
            TrainingSteps = 0;
            CosineDecay = false;
            FinalLearningRateRatio = 0.0;
        }
    }
}
