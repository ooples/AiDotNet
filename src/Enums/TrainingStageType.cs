namespace AiDotNet.Enums;

/// <summary>
/// Types of training stages in a multi-stage training pipeline.
/// </summary>
/// <remarks>
/// <para>
/// Training pipelines consist of multiple stages, each with a specific purpose.
/// Modern LLM training typically follows patterns like:
/// </para>
/// <list type="bullet">
/// <item><description><b>InstructGPT/ChatGPT:</b> Pre-training → SFT → Reward Model → PPO</description></item>
/// <item><description><b>Llama 2:</b> Pre-training → SFT → Rejection Sampling → DPO</description></item>
/// <item><description><b>Anthropic:</b> Pre-training → SFT → Constitutional AI → RLHF</description></item>
/// <item><description><b>DeepSeek:</b> Pre-training → SFT → GRPO</description></item>
/// </list>
/// </remarks>
public enum TrainingStageType
{
    // ========================================================================
    // Foundation Training Stages
    // ========================================================================

    /// <summary>
    /// Pre-training on large unlabeled text corpora (next token prediction).
    /// </summary>
    /// <remarks>
    /// The foundation stage where models learn general language understanding.
    /// Uses self-supervised learning on massive datasets (trillions of tokens).
    /// </remarks>
    PreTraining,

    /// <summary>
    /// Continued pre-training on domain-specific data.
    /// </summary>
    /// <remarks>
    /// Extends pre-training with domain-focused data (e.g., code, medical, legal).
    /// Used to create domain-specialized foundation models.
    /// </remarks>
    ContinuedPreTraining,

    /// <summary>
    /// Supervised fine-tuning on input-output pairs.
    /// </summary>
    /// <remarks>
    /// The most common fine-tuning approach where models learn from labeled examples.
    /// Used for instruction following, question answering, and task-specific training.
    /// </remarks>
    SupervisedFineTuning,

    /// <summary>
    /// Instruction tuning - SFT specifically for following instructions.
    /// </summary>
    /// <remarks>
    /// A specialized form of SFT focused on instruction-following datasets.
    /// Examples: FLAN, T0, Alpaca-style instruction datasets.
    /// </remarks>
    InstructionTuning,

    // ========================================================================
    // Preference Optimization Stages
    // ========================================================================

    /// <summary>
    /// Direct Preference Optimization family (DPO, IPO, KTO, SimPO, CPO, R-DPO, etc.).
    /// </summary>
    /// <remarks>
    /// <para>
    /// Learns directly from preference pairs without a reward model.
    /// More efficient than RLHF with comparable results.
    /// </para>
    /// <para>Includes: DPO, IPO, KTO, SimPO, CPO, R-DPO, CalDPO, TPO, APO, LCPO</para>
    /// </remarks>
    PreferenceOptimization,

    /// <summary>
    /// Odds Ratio Preference Optimization (ORPO).
    /// </summary>
    /// <remarks>
    /// Combines SFT and preference learning in a single stage.
    /// Uses odds ratios instead of log probability differences.
    /// </remarks>
    OddsRatioPreference,

    /// <summary>
    /// Contrastive Preference Optimization (CPO, NCA, Safe-NCA).
    /// </summary>
    /// <remarks>
    /// Uses contrastive learning objectives for preference optimization.
    /// Often more stable than standard DPO on noisy preference data.
    /// </remarks>
    ContrastivePreference,

    // ========================================================================
    // Reinforcement Learning Stages
    // ========================================================================

    /// <summary>
    /// Reinforcement Learning from Human Feedback (RLHF).
    /// </summary>
    /// <remarks>
    /// The classic OpenAI approach: train a reward model, then optimize with PPO.
    /// Most computationally expensive but historically most effective.
    /// </remarks>
    ReinforcementLearning,

    /// <summary>
    /// Proximal Policy Optimization (PPO) stage.
    /// </summary>
    /// <remarks>
    /// The policy optimization step of RLHF.
    /// Requires a trained reward model.
    /// </remarks>
    ProximalPolicyOptimization,

    /// <summary>
    /// Group Relative Policy Optimization (GRPO) - DeepSeek's approach.
    /// </summary>
    /// <remarks>
    /// Groups responses and uses relative rankings instead of absolute rewards.
    /// More stable than PPO and doesn't require a separate reward model.
    /// </remarks>
    GroupRelativePolicyOptimization,

    /// <summary>
    /// Reinforcement Learning from AI Feedback (RLAIF).
    /// </summary>
    /// <remarks>
    /// Uses AI-generated feedback instead of human feedback.
    /// Often combined with Constitutional AI principles.
    /// </remarks>
    ReinforcementLearningAIFeedback,

    /// <summary>
    /// Reinforcement Learning with Verifiable Rewards (RLVR).
    /// </summary>
    /// <remarks>
    /// Uses programmatically verifiable rewards (e.g., code execution, math checking).
    /// Popular for reasoning and code models.
    /// </remarks>
    ReinforcementLearningVerifiable,

    // ========================================================================
    // Reward Model Training Stages
    // ========================================================================

    /// <summary>
    /// Reward model training stage.
    /// </summary>
    /// <remarks>
    /// Trains a model to predict human preferences from comparison data.
    /// Required for RLHF pipelines.
    /// </remarks>
    RewardModelTraining,

    /// <summary>
    /// Process reward model training (step-level rewards).
    /// </summary>
    /// <remarks>
    /// Trains reward models that evaluate reasoning steps, not just final outputs.
    /// Used for math reasoning and complex problem solving.
    /// </remarks>
    ProcessRewardModelTraining,

    /// <summary>
    /// Outcome reward model training (final answer rewards).
    /// </summary>
    /// <remarks>
    /// Traditional reward model that only evaluates final outputs.
    /// </remarks>
    OutcomeRewardModelTraining,

    // ========================================================================
    // Constitutional and Safety Stages
    // ========================================================================

    /// <summary>
    /// Constitutional AI training stage.
    /// </summary>
    /// <remarks>
    /// Uses a set of principles to generate critiques and revisions.
    /// The model learns to self-correct based on constitutional principles.
    /// </remarks>
    Constitutional,

    /// <summary>
    /// Safety alignment training.
    /// </summary>
    /// <remarks>
    /// Specifically focuses on reducing harmful outputs.
    /// May use red-teaming data or safety-focused preferences.
    /// </remarks>
    SafetyAlignment,

    /// <summary>
    /// Harmlessness training stage.
    /// </summary>
    /// <remarks>
    /// Trains the model to refuse harmful requests appropriately.
    /// Often uses adversarial examples and refusal training.
    /// </remarks>
    HarmlessnessTraining,

    /// <summary>
    /// Helpfulness training stage.
    /// </summary>
    /// <remarks>
    /// Trains the model to be maximally helpful within safety constraints.
    /// Balances helpfulness with harmlessness.
    /// </remarks>
    HelpfulnessTraining,

    /// <summary>
    /// Honesty/truthfulness training stage.
    /// </summary>
    /// <remarks>
    /// Trains the model to be truthful and acknowledge uncertainty.
    /// Uses factuality datasets and calibration training.
    /// </remarks>
    HonestyTraining,

    // ========================================================================
    // Ranking and Sampling Stages
    // ========================================================================

    /// <summary>
    /// Rejection sampling optimization (RSO).
    /// </summary>
    /// <remarks>
    /// Generates multiple responses and filters using a reward model.
    /// Fine-tunes on the best responses. Used in Llama 2.
    /// </remarks>
    RejectionSampling,

    /// <summary>
    /// Rank Responses to align Human Feedback (RRHF).
    /// </summary>
    /// <remarks>
    /// Uses response rankings instead of pairwise preferences.
    /// More efficient data collection than pairwise comparisons.
    /// </remarks>
    RankResponses,

    /// <summary>
    /// Sequence Likelihood Calibration (SLiC-HF).
    /// </summary>
    /// <remarks>
    /// Calibrates sequence likelihoods using human feedback.
    /// Alternative to DPO with different optimization properties.
    /// </remarks>
    SequenceLikelihoodCalibration,

    /// <summary>
    /// Preference Ranking Optimization (PRO).
    /// </summary>
    /// <remarks>
    /// Optimizes based on preference rankings over multiple responses.
    /// </remarks>
    PreferenceRanking,

    /// <summary>
    /// Best-of-N sampling and fine-tuning.
    /// </summary>
    /// <remarks>
    /// Generates N responses, selects best using reward model, fine-tunes on selection.
    /// Simple but effective baseline approach.
    /// </remarks>
    BestOfNSampling,

    // ========================================================================
    // Self-Play and Self-Improvement Stages
    // ========================================================================

    /// <summary>
    /// Self-Play Fine-Tuning (SPIN).
    /// </summary>
    /// <remarks>
    /// Model generates responses, then learns to prefer human responses over its own.
    /// Iteratively improves without new human data.
    /// </remarks>
    SelfPlay,

    /// <summary>
    /// Self-training / self-improvement stage.
    /// </summary>
    /// <remarks>
    /// Model generates training data for itself using consistency filtering.
    /// </remarks>
    SelfTraining,

    /// <summary>
    /// Self-rewarding language models stage.
    /// </summary>
    /// <remarks>
    /// Model acts as both generator and reward model.
    /// Meta's approach to scalable alignment.
    /// </remarks>
    SelfRewarding,

    // ========================================================================
    // Knowledge Transfer Stages
    // ========================================================================

    /// <summary>
    /// Knowledge distillation from teacher model.
    /// </summary>
    /// <remarks>
    /// Transfers knowledge from a larger teacher model to a smaller student.
    /// Used for model compression and capability transfer.
    /// </remarks>
    KnowledgeDistillation,

    /// <summary>
    /// Response distillation stage.
    /// </summary>
    /// <remarks>
    /// Distills only the response behavior, not intermediate representations.
    /// </remarks>
    ResponseDistillation,

    /// <summary>
    /// Feature distillation stage.
    /// </summary>
    /// <remarks>
    /// Distills intermediate layer representations from teacher to student.
    /// </remarks>
    FeatureDistillation,

    /// <summary>
    /// Synthetic data generation and training.
    /// </summary>
    /// <remarks>
    /// Uses a teacher model to generate training data for the student.
    /// Common approach for instruction tuning without human data.
    /// </remarks>
    SyntheticDataTraining,

    // ========================================================================
    // Efficiency and Adaptation Stages
    // ========================================================================

    /// <summary>
    /// LoRA/QLoRA adapter training stage.
    /// </summary>
    /// <remarks>
    /// Parameter-efficient fine-tuning using low-rank adapters.
    /// Significantly reduces memory and compute requirements.
    /// </remarks>
    LoRAAdapterTraining,

    /// <summary>
    /// Adapter merging stage.
    /// </summary>
    /// <remarks>
    /// Merges multiple trained LoRA adapters into the base model.
    /// </remarks>
    AdapterMerging,

    /// <summary>
    /// Prefix tuning stage.
    /// </summary>
    /// <remarks>
    /// Learns soft prompts prepended to the input.
    /// Very parameter-efficient but less flexible than LoRA.
    /// </remarks>
    PrefixTuning,

    /// <summary>
    /// Prompt tuning stage.
    /// </summary>
    /// <remarks>
    /// Learns soft prompt tokens. Even more efficient than prefix tuning.
    /// </remarks>
    PromptTuning,

    // ========================================================================
    // Multi-Modal Stages
    // ========================================================================

    /// <summary>
    /// Vision-language pre-training stage.
    /// </summary>
    /// <remarks>
    /// Aligns vision and language representations.
    /// Used for models like LLaVA, GPT-4V.
    /// </remarks>
    VisionLanguagePreTraining,

    /// <summary>
    /// Visual instruction tuning stage.
    /// </summary>
    /// <remarks>
    /// Instruction tuning with image-text pairs.
    /// </remarks>
    VisualInstructionTuning,

    /// <summary>
    /// Audio-language alignment stage.
    /// </summary>
    /// <remarks>
    /// Aligns audio and language representations.
    /// </remarks>
    AudioLanguageAlignment,

    // ========================================================================
    // Specialized Training Stages
    // ========================================================================

    /// <summary>
    /// Code-specific fine-tuning stage.
    /// </summary>
    /// <remarks>
    /// Fine-tuning on code with execution feedback.
    /// May include unit test feedback.
    /// </remarks>
    CodeFineTuning,

    /// <summary>
    /// Math/reasoning fine-tuning stage.
    /// </summary>
    /// <remarks>
    /// Fine-tuning on mathematical reasoning with solution verification.
    /// </remarks>
    MathReasoningFineTuning,

    /// <summary>
    /// Chain-of-thought training stage.
    /// </summary>
    /// <remarks>
    /// Trains the model to produce step-by-step reasoning.
    /// </remarks>
    ChainOfThoughtTraining,

    /// <summary>
    /// Tool use training stage.
    /// </summary>
    /// <remarks>
    /// Trains the model to use external tools (calculators, search, code execution).
    /// </remarks>
    ToolUseTraining,

    /// <summary>
    /// Agentic behavior training stage.
    /// </summary>
    /// <remarks>
    /// Trains the model for multi-step autonomous task completion.
    /// </remarks>
    AgenticTraining,

    /// <summary>
    /// Long context training stage.
    /// </summary>
    /// <remarks>
    /// Extends the model's context window through position interpolation or similar techniques.
    /// </remarks>
    LongContextTraining,

    /// <summary>
    /// Multi-turn conversation training stage.
    /// </summary>
    /// <remarks>
    /// Trains on multi-turn dialogues to improve conversation coherence.
    /// </remarks>
    MultiTurnConversation,

    // ========================================================================
    // Curriculum and Progressive Stages
    // ========================================================================

    /// <summary>
    /// Curriculum learning stage with progressively harder examples.
    /// </summary>
    /// <remarks>
    /// Trains on easy examples first, then progressively harder ones.
    /// </remarks>
    CurriculumLearning,

    /// <summary>
    /// Progressive training with increasing model capacity.
    /// </summary>
    /// <remarks>
    /// Gradually increases model size or unfreezes more layers.
    /// </remarks>
    ProgressiveTraining,

    /// <summary>
    /// Warmup stage with lower learning rate.
    /// </summary>
    /// <remarks>
    /// Initial training with reduced learning rate before main training.
    /// </remarks>
    Warmup,

    /// <summary>
    /// Cooldown stage with learning rate decay.
    /// </summary>
    /// <remarks>
    /// Final training stage with decreasing learning rate.
    /// </remarks>
    Cooldown,

    // ========================================================================
    // Adversarial and Robustness Stages
    // ========================================================================

    /// <summary>
    /// Adversarial training for robustness.
    /// </summary>
    /// <remarks>
    /// Trains on adversarial examples to improve robustness.
    /// </remarks>
    AdversarialTraining,

    /// <summary>
    /// Red-teaming data training.
    /// </summary>
    /// <remarks>
    /// Trains on data from red-teaming efforts to patch vulnerabilities.
    /// </remarks>
    RedTeamingTraining,

    /// <summary>
    /// Jailbreak resistance training.
    /// </summary>
    /// <remarks>
    /// Specifically trains to resist jailbreak attempts.
    /// </remarks>
    JailbreakResistance,

    // ========================================================================
    // Evaluation and Utility Stages
    // ========================================================================

    /// <summary>
    /// Evaluation-only stage (no training).
    /// </summary>
    /// <remarks>
    /// Runs evaluation metrics without modifying the model.
    /// Used for checkpointing and monitoring.
    /// </remarks>
    Evaluation,

    /// <summary>
    /// Checkpoint and validation stage.
    /// </summary>
    /// <remarks>
    /// Saves a checkpoint and runs validation metrics.
    /// </remarks>
    Checkpoint,

    /// <summary>
    /// Model merging stage.
    /// </summary>
    /// <remarks>
    /// Merges multiple models using techniques like SLERP, TIES, or DARE.
    /// </remarks>
    ModelMerging,

    /// <summary>
    /// Custom user-defined training logic.
    /// </summary>
    /// <remarks>
    /// Allows users to implement arbitrary training logic.
    /// </remarks>
    Custom
}
