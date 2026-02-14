namespace AiDotNet.Interfaces;

using AiDotNet.Models.Options;

/// <summary>
/// Defines the contract for fine-tuning methods that adapt pre-trained models to specific tasks or preferences.
/// </summary>
/// <remarks>
/// <para>
/// Fine-tuning encompasses a wide range of techniques for adapting models, from supervised fine-tuning (SFT)
/// to advanced preference optimization methods like DPO, RLHF, and their variants.
/// </para>
/// <para><b>For Beginners:</b> Fine-tuning is like specialized training for an AI that already knows the basics.
/// Just like a doctor goes through general education before specializing, AI models first learn general knowledge
/// (pre-training) and then learn specific skills or behaviors (fine-tuning).
/// </para>
/// <para><b>Categories of Fine-Tuning Methods:</b></para>
/// <list type="bullet">
/// <item><term>Supervised Fine-Tuning (SFT)</term><description>Train on labeled input-output pairs</description></item>
/// <item><term>Reinforcement Learning</term><description>RLHF, PPO, GRPO - learn from reward signals</description></item>
/// <item><term>Direct Preference Optimization</term><description>DPO, IPO, KTO, SimPO - learn from preference pairs</description></item>
/// <item><term>Constitutional Methods</term><description>CAI, RLAIF - learn from AI-generated feedback with principles</description></item>
/// <item><term>Self-Play Methods</term><description>SPIN - model learns from itself</description></item>
/// </list>
/// </remarks>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The input data type for the model.</typeparam>
/// <typeparam name="TOutput">The output data type for the model.</typeparam>
[AiDotNet.Configuration.YamlConfigurable("FineTuning")]
public interface IFineTuning<T, TInput, TOutput> : IModelSerializer
{
    /// <summary>
    /// Gets the name of this fine-tuning method.
    /// </summary>
    /// <remarks>
    /// Examples: "DPO", "RLHF", "SimPO", "ORPO", "SFT", "Constitutional AI"
    /// </remarks>
    string MethodName { get; }

    /// <summary>
    /// Gets the category of this fine-tuning method.
    /// </summary>
    FineTuningCategory Category { get; }

    /// <summary>
    /// Fine-tunes a model using the configured method and provided training data.
    /// </summary>
    /// <remarks>
    /// <para>
    /// This method applies the fine-tuning algorithm to adapt the base model. The specific
    /// behavior depends on the method category:
    /// </para>
    /// <list type="bullet">
    /// <item><term>SFT</term><description>Uses labeled examples from trainingData</description></item>
    /// <item><term>Preference-based (DPO, etc.)</term><description>Uses preference pairs from trainingData</description></item>
    /// <item><term>RL-based (RLHF, PPO)</term><description>Uses reward model and training data</description></item>
    /// </list>
    /// <para><b>For Beginners:</b> This is where the actual training happens. You give it a model
    /// and training data, and it returns an improved model that's better at the specific task.</para>
    /// </remarks>
    /// <param name="baseModel">The pre-trained model to fine-tune.</param>
    /// <param name="trainingData">The training data appropriate for this fine-tuning method.</param>
    /// <param name="cancellationToken">Token for cancellation.</param>
    /// <returns>The fine-tuned model.</returns>
    Task<IFullModel<T, TInput, TOutput>> FineTuneAsync(
        IFullModel<T, TInput, TOutput> baseModel,
        FineTuningData<T, TInput, TOutput> trainingData,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Evaluates the fine-tuning quality of a model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Different fine-tuning methods have different evaluation metrics:
    /// </para>
    /// <list type="bullet">
    /// <item><term>Preference methods</term><description>Win rate against reference, preference accuracy</description></item>
    /// <item><term>RL methods</term><description>Reward scores, KL divergence from base model</description></item>
    /// <item><term>Safety methods</term><description>Harmlessness scores, refusal rates</description></item>
    /// </list>
    /// </remarks>
    /// <param name="model">The fine-tuned model to evaluate.</param>
    /// <param name="evaluationData">Evaluation dataset.</param>
    /// <param name="cancellationToken">Token for cancellation.</param>
    /// <returns>Metrics describing the fine-tuning quality.</returns>
    Task<FineTuningMetrics<T>> EvaluateAsync(
        IFullModel<T, TInput, TOutput> model,
        FineTuningData<T, TInput, TOutput> evaluationData,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Gets the configuration options for this fine-tuning method.
    /// </summary>
    FineTuningOptions<T> GetOptions();

    /// <summary>
    /// Gets whether this method requires a reward model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// RL-based methods (RLHF, PPO, GRPO) require a reward model.
    /// Direct preference methods (DPO, IPO, KTO) do not require one.
    /// </para>
    /// </remarks>
    bool RequiresRewardModel { get; }

    /// <summary>
    /// Gets whether this method requires a reference model.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Most preference methods (DPO, IPO) require a reference model for KL regularization.
    /// Reference-free methods (SimPO, ORPO) do not require one, making them more memory efficient.
    /// </para>
    /// </remarks>
    bool RequiresReferenceModel { get; }

    /// <summary>
    /// Gets whether this method supports parameter-efficient fine-tuning (PEFT).
    /// </summary>
    /// <remarks>
    /// <para>
    /// When true, this method can be combined with LoRA, QLoRA, or other PEFT techniques
    /// to reduce memory requirements during fine-tuning.
    /// </para>
    /// </remarks>
    bool SupportsPEFT { get; }

    /// <summary>
    /// Resets the fine-tuning method state.
    /// </summary>
    void Reset();
}

/// <summary>
/// Categories of fine-tuning methods.
/// </summary>
/// <remarks>
/// <para><b>For Beginners:</b> These categories group fine-tuning methods by how they learn.
/// Some learn from labeled data, others from preferences, and some from reward signals.</para>
/// </remarks>
public enum FineTuningCategory
{
    /// <summary>
    /// Supervised Fine-Tuning - learns from labeled input-output pairs.
    /// </summary>
    /// <remarks>
    /// Examples: Standard SFT, instruction tuning
    /// </remarks>
    SupervisedFineTuning,

    /// <summary>
    /// Reinforcement Learning - learns from reward signals.
    /// </summary>
    /// <remarks>
    /// Examples: RLHF, PPO, GRPO, REINFORCE
    /// </remarks>
    ReinforcementLearning,

    /// <summary>
    /// Direct Preference Optimization - learns directly from preference pairs.
    /// </summary>
    /// <remarks>
    /// Examples: DPO, IPO, KTO, SimPO, CPO, R-DPO
    /// </remarks>
    DirectPreference,

    /// <summary>
    /// Odds/Ratio-based methods that combine SFT and preference learning.
    /// </summary>
    /// <remarks>
    /// Examples: ORPO
    /// </remarks>
    OddsRatioPreference,

    /// <summary>
    /// Ranking-based methods that learn from response rankings.
    /// </summary>
    /// <remarks>
    /// Examples: RSO, RRHF, SLiC-HF, PRO
    /// </remarks>
    RankingBased,

    /// <summary>
    /// Constitutional AI methods that use principles for self-improvement.
    /// </summary>
    /// <remarks>
    /// Examples: Constitutional AI, RLAIF
    /// </remarks>
    Constitutional,

    /// <summary>
    /// Self-play methods where the model learns from itself.
    /// </summary>
    /// <remarks>
    /// Examples: SPIN
    /// </remarks>
    SelfPlay,

    /// <summary>
    /// Knowledge distillation - transfer knowledge from teacher to student.
    /// </summary>
    /// <remarks>
    /// Examples: Standard distillation, response distillation
    /// </remarks>
    KnowledgeDistillation,

    /// <summary>
    /// Contrastive methods that learn from positive/negative examples.
    /// </summary>
    /// <remarks>
    /// Examples: NCA, Safe-NCA
    /// </remarks>
    Contrastive,

    /// <summary>
    /// Adversarial methods that use game-theoretic approaches.
    /// </summary>
    /// <remarks>
    /// Examples: APO, GAPO
    /// </remarks>
    Adversarial
}

/// <summary>
/// Specific fine-tuning method types.
/// </summary>
/// <remarks>
/// <para>
/// This enum provides a complete list of supported fine-tuning methods based on
/// the latest research (2023-2025).
/// </para>
/// </remarks>
public enum FineTuningMethodType
{
    // Supervised Fine-Tuning
    /// <summary>Supervised Fine-Tuning - standard labeled data training.</summary>
    SFT,

    // Reinforcement Learning Based
    /// <summary>Reinforcement Learning from Human Feedback with PPO.</summary>
    RLHF,
    /// <summary>Proximal Policy Optimization.</summary>
    PPO,
    /// <summary>Group Relative Policy Optimization (DeepSeek).</summary>
    GRPO,
    /// <summary>Reinforcement Learning from AI Feedback.</summary>
    RLAIF,
    /// <summary>Reinforcement Learning with Verifiable Rewards.</summary>
    RLVR,
    /// <summary>REINFORCE algorithm.</summary>
    REINFORCE,

    // Direct Preference Optimization Family
    /// <summary>Direct Preference Optimization.</summary>
    DPO,
    /// <summary>Identity Preference Optimization - addresses DPO overfitting.</summary>
    IPO,
    /// <summary>Kahneman-Tversky Optimization - uses prospect theory.</summary>
    KTO,
    /// <summary>Simple Preference Optimization - reference-free.</summary>
    SimPO,
    /// <summary>Contrastive Preference Optimization.</summary>
    CPO,
    /// <summary>Robust DPO.</summary>
    RDPO,
    /// <summary>Calibrated DPO.</summary>
    CalDPO,
    /// <summary>Triple Preference Optimization.</summary>
    TPO,
    /// <summary>Adversarial/Anchored Preference Optimization.</summary>
    APO,
    /// <summary>Latent Collective Preference Optimization.</summary>
    LCPO,

    // Odds Ratio Based
    /// <summary>Odds Ratio Preference Optimization - combines SFT and alignment.</summary>
    ORPO,

    // Ranking/Contrastive Methods
    /// <summary>Rejection Sampling Optimization.</summary>
    RSO,
    /// <summary>Rank Responses to align Human Feedback.</summary>
    RRHF,
    /// <summary>Sequence Likelihood Calibration with Human Feedback.</summary>
    SLiCHF,
    /// <summary>Noise Contrastive Alignment.</summary>
    NCA,
    /// <summary>Safe Noise Contrastive Alignment.</summary>
    SafeNCA,
    /// <summary>Preference Ranking Optimization.</summary>
    PRO,

    // Constitutional/Safety Methods
    /// <summary>Constitutional AI - principle-based alignment.</summary>
    ConstitutionalAI,
    /// <summary>Safe Externalized Optimization.</summary>
    SafeEXO,

    // Self-Play Methods
    /// <summary>Self-Play Fine-Tuning.</summary>
    SPIN,

    // Knowledge Transfer
    /// <summary>Knowledge Distillation from teacher model.</summary>
    KnowledgeDistillation,

    // Adversarial Methods
    /// <summary>Generative Adversarial Policy Optimization.</summary>
    GAPO
}
