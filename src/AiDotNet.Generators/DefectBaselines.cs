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
            "global::AiDotNet.ContinualLearning.Memory.ExperienceReplayBuffer<T, TInput, TOutput>",
            "global::AiDotNet.Finance.Probabilistic.CSDI<T>",
            "global::AiDotNet.Finance.Probabilistic.TSDiff<T>",
            "global::AiDotNet.Diffusion.VAE.AudioVAE<T>",
            "global::AiDotNet.Diffusion.VAE.Causal3DVAE<T>",
            "global::AiDotNet.Diffusion.VAE.TemporalInterpolationVAE<T>",
            "global::AiDotNet.DistributedTraining.TensorParallelPagedModel<T>",
            "global::AiDotNet.Finance.AutoML.FinancialAutoML<T>",
            "global::AiDotNet.KnowledgeDistillation.FeatureDistillationStrategy<T>",
            "global::AiDotNet.MetaLearning.Algorithms.AdaptedMetaModel<T, TInput, TOutput>",
            "global::AiDotNet.MetaLearning.Algorithms.MbPAAdaptedModel<T, TInput, TOutput>",
            "global::AiDotNet.MetaLearning.Algorithms.NeuralProcessModel<T, TInput, TOutput>",
            "global::AiDotNet.MetaLearning.Models.LinearVectorModel",
            "global::AiDotNet.ModelLoading.Pretrained.OnnxFullModelAdapter<T>",
            "global::AiDotNet.NeuralNetworks.SuperNet<T>",
            "global::AiDotNet.Safety.Watermarking.FrequencyImageWatermarker<T>",
            "global::AiDotNet.Safety.Watermarking.InvisibleImageWatermarker<T>",
            "global::AiDotNet.Safety.Watermarking.NeuralImageWatermarker<T>",
            "global::AiDotNet.SelfSupervisedLearning.Evaluation.KNNEvaluator<T>",
            "global::AiDotNet.TextToSpeech.ProprietaryAPI.AmazonPolly<T>",
            "global::AiDotNet.TextToSpeech.ProprietaryAPI.AzureNeuralTTS<T>",
            "global::AiDotNet.TextToSpeech.ProprietaryAPI.ElevenLabsTTS<T>",
            "global::AiDotNet.TextToSpeech.ProprietaryAPI.GoogleCloudTTS<T>",
            "global::AiDotNet.TextToSpeech.ProprietaryAPI.Murf<T>",
            "global::AiDotNet.TextToSpeech.ProprietaryAPI.NVIDIARivaTTS<T>",
        };
}