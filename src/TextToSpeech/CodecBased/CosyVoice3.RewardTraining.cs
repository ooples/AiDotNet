using AiDotNet.Tensors.Engines.Autodiff;

namespace AiDotNet.TextToSpeech.CodecBased;

public partial class CosyVoice3<T>
{
    private Random? _diffroRandom;

    /// <summary>Runs one DiffRO policy step, equations 3-7 of the CosyVoice 3 paper.</summary>
    /// <remarks>The sample supplies transcript Tokens and ground-truth S3 CodecTokens. The reference policy and
    /// differentiable token reward must already be trained. Only this model's language-model parameters are updated;
    /// gradients flow through the frozen reward into soft speech codes. Flow, HiFT, reference and reward weights stay fixed.</remarks>
    public T TrainDifferentiableReward(TtsTrainingSample<T> sample, CosyVoice3<T> referencePolicy,
        ICosyVoiceTokenReward<T> reward, CosyVoiceDiffROOptions? options = null, Random? random = null)
    {
        var configuration = options ?? new CosyVoiceDiffROOptions();
        var (text, speech) = ValidateRewardInputs(sample, referencePolicy, reward, configuration);
        return WithFrozenReward(referencePolicy, reward, () => TrainLanguageModelObjective(sample.Tokens,
            () => RewardObjective(text, speech, referencePolicy, reward, configuration,
                random ?? (_diffroRandom ??= AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.SamplingSeed))).Loss));
    }

    /// <summary>Evaluates DiffRO with reproducible noise without consuming the training noise stream.</summary>
    public T EvaluateDifferentiableReward(TtsTrainingSample<T> sample, CosyVoice3<T> referencePolicy,
        ICosyVoiceTokenReward<T> reward, CosyVoiceDiffROOptions? options = null)
    {
        var configuration = options ?? new CosyVoiceDiffROOptions();
        var (text, speech) = ValidateRewardInputs(sample, referencePolicy, reward, configuration);
        bool previous = IsTrainingMode;
        SetTrainingMode(false);
        try
        {
            return WithFrozenReward(referencePolicy, reward, () =>
            {
                using var noGrad = new NoGradScope<T>();
                return RewardObjective(text, speech, referencePolicy, reward, configuration,
                    AiDotNet.Tensors.Helpers.RandomHelper.CreateSeededRandom(_options.SamplingSeed)).Loss.GetFlat(0);
            });
        }
        finally { SetTrainingMode(previous); }
    }

    private (int[] Text, int[] Speech) ValidateRewardInputs(TtsTrainingSample<T> sample,
        CosyVoice3<T> referencePolicy, ICosyVoiceTokenReward<T> reward, CosyVoiceDiffROOptions options)
    {
        ThrowIfDisposed();
        if (sample is null) throw new ArgumentNullException(nameof(sample));
        if (referencePolicy is null) throw new ArgumentNullException(nameof(referencePolicy));
        if (reward is null) throw new ArgumentNullException(nameof(reward));
        referencePolicy.ThrowIfDisposed();
        if (ReferenceEquals(this, referencePolicy)) throw new ArgumentException("The reference policy must be a distinct frozen model.", nameof(referencePolicy));
        if (IsOnnxMode || referencePolicy.IsOnnxMode || !HasPaperLayers || !referencePolicy.HasPaperLayers)
            throw new NotSupportedException("DiffRO needs native policy and reference language models.");
        options.Validate();
        if (referencePolicy._options.CodebookSize != _options.CodebookSize || reward.SpeechVocabularySize != _options.CodebookSize)
            throw new ArgumentException("The policy, reference and reward must share the acoustic speech-code vocabulary.");
        var owned = new HashSet<Tensor<T>>(Training.TapeTrainingStep<T>.CollectParameters(LanguageModelLayers.Cast<ILayer<T>>().ToList(), -1),
            Helpers.TensorReferenceComparer<Tensor<T>>.Instance);
        var referenceParameters = Training.TapeTrainingStep<T>.CollectParameters(referencePolicy.LanguageModelLayers.Cast<ILayer<T>>().ToList(), -1);
        if (referenceParameters.Any(owned.Contains)) throw new ArgumentException("The reference must not alias policy weights.", nameof(referencePolicy));
        owned.UnionWith(referenceParameters);
        if (Training.TapeTrainingStep<T>.CollectParameters(reward.Layers.Cast<ILayer<T>>().ToList(), -1).Any(owned.Contains))
            throw new ArgumentException("The frozen reward must not share policy or reference weights.", nameof(reward));
        var text = FrontEnd.TtsTokenIds.Read(sample.Tokens, Math.Min(_options.VocabSize, referencePolicy._options.VocabSize),
            int.MaxValue, "CosyVoice DiffRO transcript");
        var speech = FrontEnd.TtsTokenIds.Read(sample.CodecTokens ?? throw new ArgumentException("DiffRO requires ground-truth S3 CodecTokens.", nameof(sample)),
            _options.CodebookSize, int.MaxValue, "CosyVoice DiffRO speech");
        return (text, speech);
    }

    internal (Tensor<T> Loss, Tensor<T> SoftTokens, Tensor<T> Kl, Tensor<T> Reward) RewardObjective(
        IReadOnlyList<int> text, IReadOnlyList<int> speech, CosyVoice3<T> referencePolicy,
        ICosyVoiceTokenReward<T> reward, CosyVoiceDiffROOptions options, Random random)
    {
        var prepared = EnsureSystemPrompt(text);
        var policy = Lm.SpeechLogits(prepared, speech);
        Tensor<T> reference;
        using (new NoGradScope<T>()) reference = referencePolicy.Lm.SpeechLogits(prepared, speech);
        return CosyVoiceDiffRO<T>.Objective(Engine, policy, reference,
            soft => reward.LogReward(soft, text), options, random);
    }

    private static TResult WithFrozenReward<TResult>(CosyVoice3<T> reference,
        ICosyVoiceTokenReward<T> reward, Func<TResult> action)
    {
        bool referenceMode = reference.IsTrainingMode;
        var modes = reward.Layers.Select(layer => layer.InTrainingMode).ToArray();
        reference.SetTrainingMode(false);
        foreach (var layer in reward.Layers) layer.SetTrainingMode(false);
        try { return action(); }
        finally
        {
            reference.SetTrainingMode(referenceMode);
            for (int i = 0; i < modes.Length; i++) reward.Layers[i].SetTrainingMode(modes[i]);
        }
    }
}
