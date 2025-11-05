namespace AiDotNet.MetaLearning.Config;

/// <summary>
/// Base configuration for meta-learning algorithms.
/// </summary>
/// <typeparam name="T">The numeric type (float, double)</typeparam>
public class MetaLearnerConfig<T> : IMetaLearnerConfig<T> where T : struct
{
    /// <inheritdoc/>
    public T InnerLearningRate { get; set; }

    /// <inheritdoc/>
    public T OuterLearningRate { get; set; }

    /// <inheritdoc/>
    public int InnerSteps { get; set; }

    /// <inheritdoc/>
    public int MetaBatchSize { get; set; }

    /// <inheritdoc/>
    public bool FirstOrder { get; set; }

    /// <summary>
    /// Creates a default meta-learner configuration.
    /// </summary>
    public MetaLearnerConfig()
    {
        InnerLearningRate = ConvertToT(0.01);
        OuterLearningRate = ConvertToT(0.001);
        InnerSteps = 5;
        MetaBatchSize = 4;
        FirstOrder = false;
    }

    /// <inheritdoc/>
    public virtual bool IsValid()
    {
        var innerLr = Convert.ToDouble(InnerLearningRate);
        var outerLr = Convert.ToDouble(OuterLearningRate);

        return innerLr > 0 && innerLr <= 1.0 &&
               outerLr > 0 && outerLr <= 1.0 &&
               InnerSteps > 0 && InnerSteps <= 100 &&
               MetaBatchSize > 0 && MetaBatchSize <= 128;
    }

    protected T ConvertToT(double value)
    {
        return (T)Convert.ChangeType(value, typeof(T));
    }
}

/// <summary>
/// Configuration specific to MAML (Model-Agnostic Meta-Learning).
/// </summary>
/// <typeparam name="T">The numeric type (float, double)</typeparam>
/// <remarks>
/// MAML learns an initialization that allows rapid adaptation through gradient descent.
/// The algorithm computes second-order gradients (unless FirstOrder is true).
/// </remarks>
public class MAMLConfig<T> : MetaLearnerConfig<T> where T : struct
{
    /// <summary>
    /// Whether to allow batch normalization updates during adaptation.
    /// </summary>
    public bool AllowBatchNormUpdates { get; set; } = true;

    /// <summary>
    /// Creates a default MAML configuration.
    /// </summary>
    public MAMLConfig() : base()
    {
        InnerLearningRate = ConvertToT(0.01);
        OuterLearningRate = ConvertToT(0.001);
        InnerSteps = 5;
        MetaBatchSize = 4;
        FirstOrder = false;
    }
}

/// <summary>
/// Configuration specific to Reptile meta-learning algorithm.
/// </summary>
/// <typeparam name="T">The numeric type (float, double)</typeparam>
/// <remarks>
/// Reptile is a simpler alternative to MAML that performs multiple inner steps
/// and then moves the meta-parameters toward the adapted parameters.
/// It's first-order only and often performs similarly to MAML.
/// </remarks>
public class ReptileConfig<T> : MetaLearnerConfig<T> where T : struct
{
    /// <summary>
    /// Interpolation coefficient for combining meta-params with adapted params.
    /// Typical values: 0.1 to 1.0 (higher = more aggressive updates).
    /// </summary>
    public T Epsilon { get; set; }

    /// <summary>
    /// Creates a default Reptile configuration.
    /// </summary>
    public ReptileConfig() : base()
    {
        InnerLearningRate = ConvertToT(0.01);
        OuterLearningRate = ConvertToT(0.001);
        InnerSteps = 10; // Reptile typically uses more inner steps
        MetaBatchSize = 1; // Reptile often works with single tasks
        FirstOrder = true; // Reptile is first-order by design
        Epsilon = ConvertToT(1.0); // Full update by default
    }

    public override bool IsValid()
    {
        var epsilon = Convert.ToDouble(Epsilon);
        return base.IsValid() && epsilon > 0 && epsilon <= 1.0;
    }
}

/// <summary>
/// Configuration specific to SEAL (Self-Adapting Meta-Learning).
/// </summary>
/// <typeparam name="T">The numeric type (float, double)</typeparam>
/// <remarks>
/// SEAL extends meta-learning with self-adaptation mechanisms that allow
/// the model to adjust its learning strategy based on task characteristics.
/// It incorporates adaptive learning rates and task-specific adjustments.
/// </remarks>
public class SEALConfig<T> : MetaLearnerConfig<T> where T : struct
{
    /// <summary>
    /// Temperature parameter for adaptation weighting.
    /// Lower values make the model more selective in adaptation.
    /// </summary>
    public T Temperature { get; set; }

    /// <summary>
    /// Whether to use adaptive inner learning rates based on task difficulty.
    /// </summary>
    public bool AdaptiveInnerLR { get; set; }

    /// <summary>
    /// Whether to use self-improvement mechanism that refines adaptation strategy.
    /// </summary>
    public bool UseSelfImprovement { get; set; }

    /// <summary>
    /// Regularization coefficient for preventing overfitting during adaptation.
    /// </summary>
    public T AdaptationRegularization { get; set; }

    /// <summary>
    /// Number of self-improvement iterations.
    /// </summary>
    public int SelfImprovementSteps { get; set; }

    /// <summary>
    /// Whether to use task embeddings for conditioning adaptation.
    /// </summary>
    public bool UseTaskEmbeddings { get; set; }

    /// <summary>
    /// Dimension of task embedding vectors.
    /// </summary>
    public int TaskEmbeddingDim { get; set; }

    /// <summary>
    /// Creates a default SEAL configuration.
    /// </summary>
    public SEALConfig() : base()
    {
        InnerLearningRate = ConvertToT(0.01);
        OuterLearningRate = ConvertToT(0.001);
        InnerSteps = 5;
        MetaBatchSize = 4;
        FirstOrder = false;
        Temperature = ConvertToT(1.0);
        AdaptiveInnerLR = true;
        UseSelfImprovement = true;
        AdaptationRegularization = ConvertToT(0.01);
        SelfImprovementSteps = 3;
        UseTaskEmbeddings = true;
        TaskEmbeddingDim = 64;
    }

    public override bool IsValid()
    {
        var temp = Convert.ToDouble(Temperature);
        var reg = Convert.ToDouble(AdaptationRegularization);

        return base.IsValid() &&
               temp > 0 && temp <= 10.0 &&
               reg >= 0 && reg <= 1.0 &&
               SelfImprovementSteps >= 0 && SelfImprovementSteps <= 10 &&
               (!UseTaskEmbeddings || TaskEmbeddingDim > 0);
    }
}
