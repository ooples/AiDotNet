namespace AiDotNet.Enums;

/// <summary>
/// Types of training stages in a multi-stage training pipeline.
/// </summary>
public enum TrainingStageType
{
    /// <summary>
    /// Supervised fine-tuning on input-output pairs.
    /// </summary>
    SupervisedFineTuning,

    /// <summary>
    /// Preference optimization (DPO, IPO, KTO, etc.).
    /// </summary>
    PreferenceOptimization,

    /// <summary>
    /// Reinforcement learning (RLHF, PPO, GRPO, etc.).
    /// </summary>
    ReinforcementLearning,

    /// <summary>
    /// Constitutional AI training.
    /// </summary>
    Constitutional,

    /// <summary>
    /// Evaluation-only stage (no training).
    /// </summary>
    Evaluation,

    /// <summary>
    /// Custom user-defined training logic.
    /// </summary>
    Custom
}
