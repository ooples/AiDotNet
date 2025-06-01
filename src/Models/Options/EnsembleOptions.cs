using AiDotNet.Enums;
using AiDotNet.Models.Options;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for ensemble models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> These options control how your ensemble model works. Think of them as 
/// settings you can adjust to get better performance for your specific problem.
/// </para>
/// </remarks>
public class EnsembleOptions<T> : ModelOptions
{
    // Basic Configuration
    /// <summary>
    /// Gets or sets the maximum number of models allowed in the ensemble.
    /// </summary>
    /// <value>Default is 10.</value>
    /// <remarks>
    /// <b>For Beginners:</b> More models can improve accuracy but take longer to train and predict. 
    /// Start with 3-5 models and increase if needed.
    /// </remarks>
    public int MaxModels { get; set; } = 10;
    
    /// <summary>
    /// Gets or sets the minimum number of models required for the ensemble.
    /// </summary>
    /// <value>Default is 2.</value>
    /// <remarks>
    /// <b>For Beginners:</b> You need at least 2 models to have an ensemble. Some strategies may 
    /// require more models to work effectively.
    /// </remarks>
    public int MinModels { get; set; } = 2;
    
    /// <summary>
    /// Gets or sets whether to allow multiple instances of the same model type.
    /// </summary>
    /// <value>Default is false.</value>
    /// <remarks>
    /// <b>For Beginners:</b> When false, you can't add two RandomForest models, for example. 
    /// This encourages diversity in your ensemble.
    /// </remarks>
    public bool AllowDuplicateModelTypes { get; set; } = false;
    
    /// <summary>
    /// Gets or sets the strategy for combining predictions.
    /// </summary>
    /// <value>Default is WeightedAverage.</value>
    /// <remarks>
    /// <b>For Beginners:</b> This determines how predictions from different models are merged 
    /// into a final prediction. WeightedAverage is a good general-purpose choice.
    /// </remarks>
    public EnsembleStrategy Strategy { get; set; } = EnsembleStrategy.WeightedAverage;
    
    // Training Configuration
    /// <summary>
    /// Gets or sets how models are trained within the ensemble.
    /// </summary>
    /// <value>Default is Parallel.</value>
    /// <remarks>
    /// <b>For Beginners:</b> Parallel training is faster if you have multiple CPU cores. 
    /// Sequential is needed for boosting methods where each model learns from previous ones.
    /// </remarks>
    public EnsembleTrainingStrategy TrainingStrategy { get; set; } = EnsembleTrainingStrategy.Parallel;
    
    /// <summary>
    /// Gets or sets whether to update model weights after training.
    /// </summary>
    /// <value>Default is true.</value>
    /// <remarks>
    /// <b>For Beginners:</b> When true, the ensemble will automatically adjust weights based on 
    /// how well each model performs on validation data.
    /// </remarks>
    public bool UpdateWeightsAfterTraining { get; set; } = true;
    
    /// <summary>
    /// Gets or sets the method for updating model weights.
    /// </summary>
    /// <value>Default is Performance.</value>
    /// <remarks>
    /// <b>For Beginners:</b> Performance-based weighting gives more importance to models that 
    /// make better predictions. Diversity-based encourages different perspectives.
    /// </remarks>
    public WeightUpdateMethod WeightUpdateMethod { get; set; } = WeightUpdateMethod.Performance;
    
    // Performance Configuration
    /// <summary>
    /// Gets or sets whether to make predictions in parallel.
    /// </summary>
    /// <value>Default is true.</value>
    /// <remarks>
    /// <b>For Beginners:</b> Parallel prediction is faster but uses more memory. Turn off if 
    /// you're running out of memory or have few CPU cores.
    /// </remarks>
    public bool PredictInParallel { get; set; } = true;
    
    /// <summary>
    /// Gets or sets the maximum degree of parallelism.
    /// </summary>
    /// <value>Default is the number of processor cores.</value>
    /// <remarks>
    /// <b>For Beginners:</b> This limits how many models can train or predict at the same time. 
    /// Lower values use less CPU but take longer.
    /// </remarks>
    public int MaxParallelism { get; set; } = Environment.ProcessorCount;
    
    // Advanced Configuration
    /// <summary>
    /// Gets or sets the minimum weight threshold for keeping a model.
    /// </summary>
    /// <value>Models with weights below this are removed.</value>
    /// <remarks>
    /// <b>For Beginners:</b> If a model's weight falls below this threshold, it's considered 
    /// not useful and removed from the ensemble.
    /// </remarks>
    public T MinimumModelWeight { get; set; } = default!;
    
    /// <summary>
    /// Gets or sets the weight regularization parameter.
    /// </summary>
    /// <value>Prevents any single model from dominating.</value>
    /// <remarks>
    /// <b>For Beginners:</b> This prevents one model from getting too much weight, ensuring 
    /// the ensemble stays diverse.
    /// </remarks>
    public T WeightRegularization { get; set; } = default!;
    
    /// <summary>
    /// Gets or sets whether to use out-of-bag scoring for weight calculation.
    /// </summary>
    /// <value>Default is true.</value>
    /// <remarks>
    /// <b>For Beginners:</b> Out-of-bag scoring tests each model on data it wasn't trained on, 
    /// giving a more honest assessment of performance.
    /// </remarks>
    public bool UseOutOfBagScoring { get; set; } = true;
    
    /// <summary>
    /// Gets or sets the train/test split ratio for weight calculation.
    /// </summary>
    /// <value>Default is 0.2 (20% for testing).</value>
    /// <remarks>
    /// <b>For Beginners:</b> When calculating weights, this portion of data is held back to 
    /// test model performance. 0.2 means 20% for testing, 80% for training.
    /// </remarks>
    public double TrainTestSplitRatio { get; set; } = 0.2;
    
    // Dynamic Selection Configuration
    /// <summary>
    /// Gets or sets the competence threshold for dynamic selection.
    /// </summary>
    /// <value>Default is 0.7.</value>
    /// <remarks>
    /// <b>For Beginners:</b> For dynamic selection, models must achieve at least this accuracy 
    /// on similar data to be considered competent.
    /// </remarks>
    public double CompetenceThreshold { get; set; } = 0.7;
    
    /// <summary>
    /// Gets or sets the size of the local region for competence calculation.
    /// </summary>
    /// <value>Default is 50.</value>
    /// <remarks>
    /// <b>For Beginners:</b> When determining if a model is good for a specific input, look at 
    /// its performance on this many similar examples.
    /// </remarks>
    public int LocalRegionSize { get; set; } = 50;
    
    // Stacking Configuration
    /// <summary>
    /// Gets or sets the number of folds for cross-validation in stacking.
    /// </summary>
    /// <value>Default is 5.</value>
    /// <remarks>
    /// <b>For Beginners:</b> For stacking, data is split into this many parts to avoid overfitting. 
    /// More folds are better but take longer.
    /// </remarks>
    public int NumberOfFolds { get; set; } = 5;
    
    /// <summary>
    /// Gets or sets whether to use blending instead of full stacking.
    /// </summary>
    /// <value>Default is false.</value>
    /// <remarks>
    /// <b>For Beginners:</b> Blending is simpler than stacking - it uses a fixed validation set 
    /// instead of cross-validation.
    /// </remarks>
    public bool UseBlending { get; set; } = false;
    
    // Boosting Configuration
    /// <summary>
    /// Gets or sets the learning rate for boosting methods.
    /// </summary>
    /// <value>Default is 1.0.</value>
    /// <remarks>
    /// <b>For Beginners:</b> Lower values make boosting more conservative but may need more 
    /// iterations. Try 0.1 for more stable learning.
    /// </remarks>
    public double LearningRate { get; set; } = 1.0;
    
    /// <summary>
    /// Gets or sets the maximum number of boosting rounds.
    /// </summary>
    /// <value>Default is 100.</value>
    /// <remarks>
    /// <b>For Beginners:</b> More rounds can improve accuracy but may overfit. Monitor validation 
    /// performance to find the right number.
    /// </remarks>
    public int MaxBoostingRounds { get; set; } = 100;
    
    // Validation Configuration
    /// <summary>
    /// Gets or sets whether to validate models before adding to ensemble.
    /// </summary>
    /// <value>Default is true.</value>
    /// <remarks>
    /// <b>For Beginners:</b> When true, models are tested before being added to ensure they 
    /// meet minimum performance standards.
    /// </remarks>
    public bool ValidateModelsBeforeAdding { get; set; } = true;
    
    /// <summary>
    /// Gets or sets the minimum performance threshold for model validation.
    /// </summary>
    /// <value>Default is 0.5.</value>
    /// <remarks>
    /// <b>For Beginners:</b> Models must achieve at least this accuracy to be added to the 
    /// ensemble. Prevents adding poor models.
    /// </remarks>
    public double MinimumPerformanceThreshold { get; set; } = 0.5;
}

/// <summary>
/// Defines strategies for training models within an ensemble.
/// </summary>
public enum EnsembleTrainingStrategy
{
    /// <summary>
    /// Train all models independently in parallel.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> All models train at the same time on the same data. Fastest option 
    /// when you have multiple CPU cores.
    /// </remarks>
    Parallel,
    
    /// <summary>
    /// Train models one after another.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Models train one at a time. Needed for methods where later models 
    /// depend on earlier ones.
    /// </remarks>
    Sequential,
    
    /// <summary>
    /// Train each model on random subsets (bootstrap aggregating).
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Each model trains on a different random sample of the data, creating 
    /// diversity. This is how Random Forests work.
    /// </remarks>
    Bagging,
    
    /// <summary>
    /// Train models sequentially, focusing on errors of previous models.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Each new model tries to fix the mistakes of previous models. This 
    /// is how AdaBoost and Gradient Boosting work.
    /// </remarks>
    Boosting
}

/// <summary>
/// Defines methods for updating model weights in an ensemble.
/// </summary>
public enum WeightUpdateMethod
{
    /// <summary>
    /// Update weights based on validation performance.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Models that make better predictions get higher weights.
    /// </remarks>
    Performance,
    
    /// <summary>
    /// Update weights to favor diverse models.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Models that make different kinds of errors get higher weights, 
    /// encouraging different perspectives.
    /// </remarks>
    Diversity,
    
    /// <summary>
    /// Update weights using Bayesian inference.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Uses probability theory to determine the most likely correct weights.
    /// </remarks>
    Bayesian,
    
    /// <summary>
    /// Update weights using gradient descent.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Adjusts weights in small steps to minimize prediction errors.
    /// </remarks>
    Gradient,
    
    /// <summary>
    /// Update weights using evolutionary algorithms.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Evolves weights over generations, keeping the best performers.
    /// </remarks>
    Evolutionary,
    
    /// <summary>
    /// Don't update weights after initialization.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> Keeps the initial weights you set without any automatic adjustment.
    /// </remarks>
    Fixed
}