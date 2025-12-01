
using AiDotNet.Interfaces;

namespace AiDotNet.MetaLearning.Config;

/// <summary>
/// Configuration for the SEAL (Self-supervised and Episodic Active Learning) meta-learning algorithm.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// SEAL extends traditional meta-learning by leveraging unlabeled data in the query set through
/// a three-phase training process: self-supervised pre-training, active learning selection,
/// and supervised fine-tuning. This configuration controls all three phases plus the meta-update.
/// </para>
/// <para><b>For Beginners:</b> SEAL uses unlabeled data to improve few-shot learning performance.
///
/// <b>Three-Phase Process:</b>
/// 1. <b>Self-Supervised Pre-Training:</b> Learn patterns from unlabeled query set
///    - Controlled by: SelfSupervisedSteps
///    - Typical value: 10-20 steps
///    - What it does: Teaches the model useful features without labels
///
/// 2. <b>Active Learning:</b> Select most confident predictions to pseudo-label
///    - Controlled by: ActiveLearningK
///    - Typical value: 10-30 examples
///    - What it does: Picks examples the model is confident about
///
/// 3. <b>Supervised Fine-Tuning:</b> Train on real labels + pseudo-labels
///    - Controlled by: SupervisedSteps (replaces traditional InnerSteps)
///    - Typical value: 5-10 steps
///    - What it does: Adapts to the task using more training data
///
/// <b>Meta-Update:</b> Like Reptile, SEAL moves toward adapted parameters
/// - Controlled by: MetaLearningRate
/// - Typical value: 0.001
///
/// <b>Key Insight:</b> Traditional meta-learning only uses 25 examples (5-way 5-shot support set).
/// SEAL also uses the 75 unlabeled query examples, giving ~3x more data to learn from!
/// </para>
/// <para><b>Recommended Settings:</b>
///
/// For image classification (e.g., mini-ImageNet):
/// - SelfSupervisedSteps: 15 (learn features from rotations)
/// - SupervisedSteps: 5 (adapt to task)
/// - ActiveLearningK: 20 (select 20 confident examples)
/// - MetaLearningRate: 0.001 (stable meta-updates)
///
/// For quick experimentation:
/// - SelfSupervisedSteps: 5 (faster but less feature learning)
/// - SupervisedSteps: 3 (faster adaptation)
/// - ActiveLearningK: 10 (fewer pseudo-labels)
/// </para>
/// </remarks>
public class SEALTrainerConfig<T> : IMetaLearnerConfig<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    /// <summary>
    /// Gets or sets the number of self-supervised pre-training steps on the query set.
    /// </summary>
    /// <value>
    /// How many gradient steps to perform on self-supervised tasks (e.g., rotation prediction).
    /// Typical values: 10-20
    /// </value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This controls how much the model learns from unlabeled data.
    ///
    /// - <b>Higher values (15-20):</b> Model learns better features, but takes longer
    /// - <b>Lower values (5-10):</b> Faster training, but less feature learning
    ///
    /// Think of it like: "How many practice problems should I solve before the actual exam?"
    /// More practice = better prepared, but takes more time.
    /// </para>
    /// </remarks>
    public int SelfSupervisedSteps { get; set; } = 15;

    /// <summary>
    /// Gets or sets the number of supervised fine-tuning steps on support + pseudo-labeled data.
    /// </summary>
    /// <value>
    /// How many gradient steps to perform on the combined labeled dataset.
    /// Typical values: 5-10
    /// </value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is like InnerSteps in MAML/Reptile, but uses MORE data.
    ///
    /// Traditional meta-learning: Train on 25 labeled examples
    /// SEAL: Train on 25 labeled + 20 pseudo-labeled = 45 total examples
    ///
    /// More data means faster adaptation with fewer steps needed!
    /// </para>
    /// </remarks>
    public int SupervisedSteps { get; set; } = 5;

    /// <summary>
    /// Gets or sets the number of confident examples to select for pseudo-labeling (K in active learning).
    /// </summary>
    /// <value>
    /// How many high-confidence predictions to use as additional training data.
    /// Typical values: 10-30
    /// </value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This controls how many "confident guesses" to add to training.
    ///
    /// The model looks at unlabeled query set and picks examples where it's confident:
    /// - "I'm 95% sure this is a cat" → Include it
    /// - "I'm 55% sure this is a dog" → Skip it
    ///
    /// <b>Trade-offs:</b>
    /// - <b>Higher K (20-30):</b> More training data, but some pseudo-labels may be wrong
    /// - <b>Lower K (5-15):</b> Safer (only very confident examples), but less data
    ///
    /// Typical: K = 20 works well for 5-way 15-shot query sets (75 total query examples)
    /// </para>
    /// </remarks>
    public int ActiveLearningK { get; set; } = 20;

    /// <inheritdoc/>
    public T InnerLearningRate { get; set; } = NumOps.FromDouble(0.01);

    /// <inheritdoc/>
    public T MetaLearningRate { get; set; } = NumOps.FromDouble(0.001);

    /// <inheritdoc/>
    /// <remarks>
    /// <para><b>Note for SEAL:</b> This property is kept for interface compatibility with IMetaLearnerConfig
    /// and maps directly to SupervisedSteps. Getting InnerSteps returns SupervisedSteps, and setting InnerSteps
    /// modifies SupervisedSteps. SEAL has multiple adaptation phases:
    /// - SelfSupervisedSteps (phase 1: self-supervised pre-training)
    /// - SupervisedSteps (phase 3: supervised fine-tuning, mapped from InnerSteps)
    /// - ActiveLearningK (phase 2: active learning selection)
    ///
    /// For full control over SEAL training, use the specific phase properties instead of InnerSteps.
    /// </para>
    /// </remarks>
    public int InnerSteps
    {
        get => SupervisedSteps;
        set => SupervisedSteps = value;
    }

    /// <inheritdoc/>
    public int MetaBatchSize { get; set; } = 4;

    /// <inheritdoc/>
    public int NumMetaIterations { get; set; } = 1000;

    /// <summary>
    /// Creates a default SEAL configuration with standard values.
    /// </summary>
    /// <remarks>
    /// Default values based on the SEAL paper (Hao et al., 2019) and empirical best practices:
    /// - Self-supervised steps: 15 (balance between feature learning and speed)
    /// - Supervised steps: 5 (fewer steps needed thanks to more data from pseudo-labeling)
    /// - Active learning K: 20 (works well for typical 5-way 15-shot query sets)
    /// - Inner learning rate: 0.01 (conservative for stability)
    /// - Meta learning rate: 0.001 (10x smaller than inner rate, Reptile-style)
    /// - Meta batch size: 4 (balance between stability and speed)
    /// - Num meta iterations: 1000 (standard training duration)
    /// </remarks>
    public SEALTrainerConfig()
    {
    }

    /// <summary>
    /// Creates a SEAL configuration with custom values.
    /// </summary>
    /// <param name="selfSupervisedSteps">Number of self-supervised pre-training steps.</param>
    /// <param name="supervisedSteps">Number of supervised fine-tuning steps.</param>
    /// <param name="activeLearningK">Number of confident examples to pseudo-label.</param>
    /// <param name="innerLearningRate">Learning rate for task adaptation.</param>
    /// <param name="metaLearningRate">Meta-learning rate (epsilon in Reptile-style update).</param>
    /// <param name="metaBatchSize">Number of tasks per meta-update.</param>
    /// <param name="numMetaIterations">Total number of meta-training iterations.</param>
    public SEALTrainerConfig(
        int selfSupervisedSteps,
        int supervisedSteps,
        int activeLearningK,
        double innerLearningRate = 0.01,
        double metaLearningRate = 0.001,
        int metaBatchSize = 4,
        int numMetaIterations = 1000)
    {
        SelfSupervisedSteps = selfSupervisedSteps;
        SupervisedSteps = supervisedSteps;
        ActiveLearningK = activeLearningK;
        InnerLearningRate = NumOps.FromDouble(innerLearningRate);
        MetaLearningRate = NumOps.FromDouble(metaLearningRate);
        MetaBatchSize = metaBatchSize;
        NumMetaIterations = numMetaIterations;
    }

    /// <inheritdoc/>
    /// <remarks>
    /// <para><b>Validation rules:</b>
    /// - InnerLearningRate: must be in range (0, 1.0]
    /// - MetaLearningRate: must be in range (0, 1.0]
    /// - SelfSupervisedSteps: must be in range [1, 100]
    /// - SupervisedSteps: must be in range [1, 100]
    /// - ActiveLearningK: must be in range [1, 1000]
    /// - MetaBatchSize: must be in range [1, 128]
    /// - NumMetaIterations: must be in range [1, 1000000]
    /// </para>
    /// </remarks>
    public bool IsValid()
    {
        var innerLr = Convert.ToDouble(InnerLearningRate);
        var metaLr = Convert.ToDouble(MetaLearningRate);

        return innerLr > 0 && innerLr <= 1.0 &&
               metaLr > 0 && metaLr <= 1.0 &&
               SelfSupervisedSteps > 0 && SelfSupervisedSteps <= 100 &&
               SupervisedSteps > 0 && SupervisedSteps <= 100 &&
               ActiveLearningK > 0 && ActiveLearningK <= 1000 &&
               MetaBatchSize > 0 && MetaBatchSize <= 128 &&
               NumMetaIterations > 0 && NumMetaIterations <= 1000000;
    }
}
