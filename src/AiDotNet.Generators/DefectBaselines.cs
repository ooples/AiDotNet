namespace AiDotNet.Generators;

/// <summary>
/// Checked-in baselines for the defect-class diagnostics: every existing violation, listed so that a NEW
/// violation is a compile error while the known ones are worked down.
/// </summary>
/// <remarks>
/// <para>
/// A baseline may only shrink. An entry that no longer violates (the model gained tests, or was deleted)
/// is itself an error (ADNGEN002), so a fix cannot land without removing its line. That keeps the list an
/// exact inventory of the remaining debt rather than a growing allowlist.
/// </para>
/// <para>
/// Parking a known defect class in prose is how the same defects recurred; this is the ratchet in code.
/// </para>
/// </remarks>
internal static class DefectBaselines
{
    /// <summary>
    /// Models with no generated or hand-written test class when AIDN040 became an error (2026-09-28): 328
    /// of 1807. Keyed by fully qualified name because simple names collide.
    /// </summary>
    internal static readonly System.Collections.Generic.HashSet<string> UntestedModels =
        new System.Collections.Generic.HashSet<string>(System.StringComparer.Ordinal)
        {
            "global::AiDotNet.AutoML.AutoMLEnsembleModel<T>",
            "global::AiDotNet.AutoML.DiffusionAutoML<T>",
            "global::AiDotNet.ComputerVision.Segmentation.Referring.GLaMM<T>",
            "global::AiDotNet.ContinualLearning.Memory.ExperienceReplayBuffer<T, TInput, TOutput>",
            "global::AiDotNet.Diffusion.NoisePredictors.AsymmDiTPredictor<T>",
            "global::AiDotNet.Diffusion.NoisePredictors.DiTNoisePredictor<T>",
            "global::AiDotNet.Diffusion.NoisePredictors.EMMDiTPredictor<T>",
            "global::AiDotNet.Diffusion.NoisePredictors.FlagDiTPredictor<T>",
            "global::AiDotNet.Diffusion.NoisePredictors.FluxDoubleStreamPredictor<T>",
            "global::AiDotNet.Diffusion.NoisePredictors.MMDiTNoisePredictor<T>",
            "global::AiDotNet.Diffusion.NoisePredictors.MMDiTXNoisePredictor<T>",
            "global::AiDotNet.Diffusion.NoisePredictors.SiTPredictor<T>",
            "global::AiDotNet.Diffusion.NoisePredictors.UNetNoisePredictor<T>",
            "global::AiDotNet.Diffusion.NoisePredictors.UViTNoisePredictor<T>",
            "global::AiDotNet.Diffusion.VAE.AudioVAE<T>",
            "global::AiDotNet.Diffusion.VAE.AutoencoderKL<T>",
            "global::AiDotNet.Diffusion.VAE.Causal3DVAE<T>",
            "global::AiDotNet.Diffusion.VAE.DeepCompressionVAE<T>",
            "global::AiDotNet.Diffusion.VAE.EQVAEModel<T>",
            "global::AiDotNet.Diffusion.VAE.ImprovedVideoVAE<T>",
            "global::AiDotNet.Diffusion.VAE.LiteVAEModel<T>",
            "global::AiDotNet.Diffusion.VAE.SDXLVAEModel<T>",
            "global::AiDotNet.Diffusion.VAE.StandardVAE<T>",
            "global::AiDotNet.Diffusion.VAE.TemporalInterpolationVAE<T>",
            "global::AiDotNet.Diffusion.VAE.TemporalVAE<T>",
            "global::AiDotNet.DistributedTraining.DDPModel<T, TInput, TOutput>",
            "global::AiDotNet.DistributedTraining.FSDPModel<T, TInput, TOutput>",
            "global::AiDotNet.DistributedTraining.HybridShardedModel<T, TInput, TOutput>",
            "global::AiDotNet.DistributedTraining.PipelineParallelModel<T, TInput, TOutput>",
            "global::AiDotNet.DistributedTraining.TensorParallelModel<T, TInput, TOutput>",
            "global::AiDotNet.DistributedTraining.TensorParallelPagedModel<T>",
            "global::AiDotNet.DistributedTraining.ZeRO1Model<T, TInput, TOutput>",
            "global::AiDotNet.DistributedTraining.ZeRO2Model<T, TInput, TOutput>",
            "global::AiDotNet.DistributedTraining.ZeRO3Model<T, TInput, TOutput>",
            "global::AiDotNet.Finance.AutoML.FinancialAutoML<T>",
            "global::AiDotNet.Finance.Probabilistic.CSDI<T>",
            "global::AiDotNet.Finance.Probabilistic.TimeGrad<T>",
            "global::AiDotNet.Finance.Probabilistic.TSDiff<T>",
            "global::AiDotNet.KnowledgeDistillation.FeatureDistillationStrategy<T>",
            "global::AiDotNet.MetaLearning.Algorithms.AdaptedMetaModel<T, TInput, TOutput>",
            "global::AiDotNet.MetaLearning.Algorithms.MbPAAdaptedModel<T, TInput, TOutput>",
            "global::AiDotNet.MetaLearning.Algorithms.NeuralProcessModel<T, TInput, TOutput>",
            "global::AiDotNet.MetaLearning.Models.LinearVectorModel",
            "global::AiDotNet.ModelLoading.Pretrained.OnnxFullModelAdapter<T>",
            "global::AiDotNet.NeuralNetworks.SuperNet<T>",
            "global::AiDotNet.ReinforcementLearning.Policies.BetaPolicy<T>",
            "global::AiDotNet.ReinforcementLearning.Policies.ContinuousPolicy<T>",
            "global::AiDotNet.ReinforcementLearning.Policies.DeterministicPolicy<T>",
            "global::AiDotNet.ReinforcementLearning.Policies.DiscretePolicy<T>",
            "global::AiDotNet.ReinforcementLearning.Policies.MixedPolicy<T>",
            "global::AiDotNet.ReinforcementLearning.Policies.MultiModalPolicy<T>",
            "global::AiDotNet.Safety.Audio.AcousticToxicityDetector<T>",
            "global::AiDotNet.Safety.Audio.MaskingVoiceProtector<T>",
            "global::AiDotNet.Safety.Audio.PerturbationVoiceProtector<T>",
            "global::AiDotNet.Safety.Audio.SpectralDeepfakeDetector<T>",
            "global::AiDotNet.Safety.Audio.TranscriptionToxicityDetector<T>",
            "global::AiDotNet.Safety.Audio.VoiceprintDeepfakeDetector<T>",
            "global::AiDotNet.Safety.Audio.WatermarkDeepfakeDetector<T>",
            "global::AiDotNet.Safety.Audio.WatermarkVoiceProtector<T>",
            "global::AiDotNet.Safety.Multimodal.TextImageAlignmentChecker<T>",
            "global::AiDotNet.Safety.Text.ClassifierToxicityDetector<T>",
            "global::AiDotNet.Safety.Text.CompositePIIDetector<T>",
            "global::AiDotNet.Safety.Text.ContextAwarePIIDetector<T>",
            "global::AiDotNet.Safety.Text.EmbeddingCopyrightDetector<T>",
            "global::AiDotNet.Safety.Text.EmbeddingToxicityDetector<T>",
            "global::AiDotNet.Safety.Text.EnsembleJailbreakDetector<T>",
            "global::AiDotNet.Safety.Text.EnsembleToxicityDetector<T>",
            "global::AiDotNet.Safety.Text.EntailmentHallucinationDetector<T>",
            "global::AiDotNet.Safety.Text.GradientJailbreakDetector<T>",
            "global::AiDotNet.Safety.Text.KnowledgeTripletHallucinationDetector<T>",
            "global::AiDotNet.Safety.Text.NERPIIDetector<T>",
            "global::AiDotNet.Safety.Text.NgramCopyrightDetector<T>",
            "global::AiDotNet.Safety.Text.PatternJailbreakDetector<T>",
            "global::AiDotNet.Safety.Text.PerplexityMemorizationDetector<T>",
            "global::AiDotNet.Safety.Text.ReferenceBasedHallucinationDetector<T>",
            "global::AiDotNet.Safety.Text.RegexPIIDetector<T>",
            "global::AiDotNet.Safety.Text.RuleBasedToxicityDetector<T>",
            "global::AiDotNet.Safety.Text.SelfConsistencyHallucinationDetector<T>",
            "global::AiDotNet.Safety.Text.SemanticJailbreakDetector<T>",
            "global::AiDotNet.Safety.Watermarking.AudioSealWatermarker<T>",
            "global::AiDotNet.Safety.Watermarking.AudioWatermarker<T>",
            "global::AiDotNet.Safety.Watermarking.FrequencyImageWatermarker<T>",
            "global::AiDotNet.Safety.Watermarking.InvisibleImageWatermarker<T>",
            "global::AiDotNet.Safety.Watermarking.LexicalWatermarker<T>",
            "global::AiDotNet.Safety.Watermarking.NeuralImageWatermarker<T>",
            "global::AiDotNet.Safety.Watermarking.SamplingWatermarker<T>",
            "global::AiDotNet.Safety.Watermarking.SpectralAudioWatermarker<T>",
            "global::AiDotNet.Safety.Watermarking.SyntacticWatermarker<T>",
            "global::AiDotNet.Safety.Watermarking.WatermarkDetector<T>",
            "global::AiDotNet.SelfSupervisedLearning.BarlowTwins<T>",
            "global::AiDotNet.SelfSupervisedLearning.BYOL<T>",
            "global::AiDotNet.SelfSupervisedLearning.DINO<T>",
            "global::AiDotNet.SelfSupervisedLearning.Evaluation.KNNEvaluator<T>",
            "global::AiDotNet.SelfSupervisedLearning.iBOT<T>",
            "global::AiDotNet.SelfSupervisedLearning.MAE<T>",
            "global::AiDotNet.SelfSupervisedLearning.MoCo<T>",
            "global::AiDotNet.SelfSupervisedLearning.MoCoV2<T>",
            "global::AiDotNet.SelfSupervisedLearning.MoCoV3<T>",
            "global::AiDotNet.SelfSupervisedLearning.SimCLR<T>",
            "global::AiDotNet.SelfSupervisedLearning.SimSiam<T>",
            "global::AiDotNet.SpeechRecognition.ConformerFamily.Branchformer<T>",
            "global::AiDotNet.SpeechRecognition.ConformerFamily.EBranchformer<T>",
            "global::AiDotNet.SpeechRecognition.LLMIntegrated.AudioPaLM<T>",
            "global::AiDotNet.TextToSpeech.CodecBased.AudioLM<T>",
            "global::AiDotNet.TextToSpeech.CodecBased.CosyVoice2<T>",
            "global::AiDotNet.TextToSpeech.CodecBased.FishSpeech<T>",
            "global::AiDotNet.TextToSpeech.CodecBased.VALLE<T>",
            "global::AiDotNet.TextToSpeech.CodecBased.VoiceCraft<T>",
            "global::AiDotNet.TextToSpeech.FlowDiffusion.MatchaTTS<T>",
            "global::AiDotNet.TextToSpeech.StyleEmotion.StyleTTS2<T>",
            "global::AiDotNet.TextToSpeech.Vocoders.WaveNet<T>",
            "global::AiDotNet.VisionLanguage.Grounding.GroundedSAM2<T>",
        };
}