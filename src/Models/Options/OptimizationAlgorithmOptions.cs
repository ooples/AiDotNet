using AiDotNet.Enums;

namespace AiDotNet.Models.Options;

/// <summary>
/// Configuration options for optimization algorithms used in machine learning models.
/// </summary>
/// <remarks>
/// <para>
/// Optimization algorithms are methods used to find the best parameters for a machine learning model
/// by minimizing or maximizing an objective function.
/// </para>
/// <para><b>For Beginners:</b> Think of optimization as the process of "learning" in machine learning.
/// It's like adjusting the knobs on a radio until you get the clearest signal. These settings control
/// how quickly and accurately the algorithm learns from your data.
/// </para>
/// </remarks>
public class OptimizationAlgorithmOptions<T, TInput, TOutput> : ModelOptions
{
    /// <summary>
    /// Gets or sets the maximum number of iterations (epochs) for the optimization algorithm.
    /// </summary>
    /// <value>The maximum number of iterations, defaulting to 100.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is how many times the algorithm will go through your entire dataset.
    /// Each complete pass is called an "epoch". More iterations give the algorithm more chances to learn,
    /// but too many can lead to overfitting (memorizing the data instead of learning patterns).</para>
    /// </remarks>
    public int MaxIterations { get; set; } = 100;

    /// <summary>
    /// Gets or sets whether to use early stopping to prevent overfitting.
    /// </summary>
    /// <value>True to use early stopping (default), false otherwise.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Early stopping is like knowing when to stop studying - if your performance
    /// isn't improving anymore, it's time to stop. This prevents the model from memorizing the training data
    /// (overfitting) instead of learning general patterns.</para>
    /// </remarks>
    public bool UseEarlyStopping { get; set; } = true;

    /// <summary>
    /// Gets or sets the number of iterations to wait before stopping if no improvement is observed.
    /// </summary>
    /// <value>The number of iterations to wait, defaulting to 10.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is how patient the algorithm is. If it hasn't seen any improvement
    /// for this many iterations, it will stop training. Higher values make the algorithm more patient,
    /// giving it more chances to find improvements.</para>
    /// </remarks>
    public int EarlyStoppingPatience { get; set; } = 10;

    /// <summary>
    /// Gets or sets the number of iterations to wait before adjusting parameters when the model is performing poorly.
    /// </summary>
    /// <value>The number of iterations to wait, defaulting to 5.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> If your model is performing poorly, this setting determines how long to wait
    /// before making adjustments. It's like giving the model a few more chances before changing your approach.</para>
    /// </remarks>
    public int BadFitPatience { get; set; } = 5;

    /// <summary>
    /// Gets or sets the minimum number of features to consider in the model.
    /// </summary>
    /// <value>The minimum number of features.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Features are the characteristics or attributes in your data (like height, weight, color).
    /// This setting defines the minimum number of these characteristics the model should use.</para>
    /// </remarks>
    public int MinimumFeatures { get; set; }

    /// <summary>
    /// Gets or sets the maximum number of features to consider in the model.
    /// </summary>
    /// <value>The maximum number of features.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This limits how many characteristics (features) from your data the model can use.
    /// Setting a maximum can help prevent the model from becoming too complex.</para>
    /// </remarks>
    public int MaximumFeatures { get; set; }

    /// <summary>
    /// Gets or sets whether to use expression trees for optimization.
    /// </summary>
    /// <value>True to use expression trees, false otherwise (default).</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Expression trees are a way to represent code as data, which can make
    /// certain operations faster. This is an advanced feature that most beginners won't need to change.</para>
    /// </remarks>
    public bool UseExpressionTrees { get; set; } = false;

    /// <summary>
    /// Gets or sets the initial learning rate for the optimization algorithm.
    /// </summary>
    /// <value>The initial learning rate, defaulting to 0.01.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The learning rate controls how big of steps the algorithm takes when learning.
    /// A higher rate means bigger steps (faster learning but might overshoot), while a lower rate means smaller steps
    /// (more precise but slower learning). Think of it like adjusting the speed of learning.</para>
    /// </remarks>
    public virtual double InitialLearningRate { get; set; } = 0.01;

    /// <summary>
    /// Gets or sets whether to automatically adjust the learning rate during training.
    /// </summary>
    /// <value>True to use adaptive learning rate (default), false otherwise.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> When enabled, the algorithm will automatically adjust how fast it learns
    /// based on its progress. It's like slowing down when you're getting close to your destination.</para>
    /// </remarks>
    public bool UseAdaptiveLearningRate { get; set; } = true;

    /// <summary>
    /// Gets or sets the rate at which the learning rate decreases over time.
    /// </summary>
    /// <value>The learning rate decay factor, defaulting to 0.99.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This controls how quickly the learning rate decreases. A value of 0.99 means
    /// the learning rate becomes 99% of its previous value after each iteration, gradually slowing down the learning process.</para>
    /// </remarks>
    public double LearningRateDecay { get; set; } = 0.99;

    /// <summary>
    /// Gets or sets the minimum allowed learning rate.
    /// </summary>
    /// <value>The minimum learning rate, defaulting to 0.000001.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This prevents the learning rate from becoming too small.
    /// Even if the rate keeps decreasing, it won't go below this value.</para>
    /// </remarks>
    public double MinLearningRate { get; set; } = 1e-6;

    /// <summary>
    /// Gets or sets the maximum allowed learning rate.
    /// </summary>
    /// <value>The maximum learning rate, defaulting to 1.0.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This prevents the learning rate from becoming too large,
    /// which could cause the algorithm to make wild guesses instead of learning properly.</para>
    /// </remarks>
    public double MaxLearningRate { get; set; } = 1.0;

    /// <summary>
    /// Gets or sets whether to automatically adjust the momentum during training.
    /// </summary>
    /// <value>True to use adaptive momentum (default), false otherwise.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Momentum helps the algorithm push through flat areas and avoid getting stuck in local minima.
    /// It's like a ball rolling downhill that can roll through small bumps. When adaptive momentum is enabled,
    /// the algorithm will adjust this effect based on its progress.</para>
    /// </remarks>
    public bool UseAdaptiveMomentum { get; set; } = true;

    /// <summary>
    /// Gets or sets the initial momentum value.
    /// </summary>
    /// <value>The initial momentum, defaulting to 0.9.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> Momentum determines how much the previous update influences the current one.
    /// A value of 0.9 means 90% of the previous update is carried forward. Higher values make the algorithm
    /// less likely to get stuck but might make it harder to settle on the best solution.</para>
    /// </remarks>
    public double InitialMomentum { get; set; } = 0.9;

    /// <summary>
    /// Gets or sets the factor by which momentum increases when performance improves.
    /// </summary>
    /// <value>The momentum increase factor, defaulting to 1.05.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> When the model is improving, momentum will be increased by this factor.
    /// A value of 1.05 means momentum becomes 105% of its previous value, helping the model move faster
    /// in promising directions.</para>
    /// </remarks>
    public double MomentumIncreaseFactor { get; set; } = 1.05;

    /// <summary>
    /// Gets or sets the factor by which momentum decreases when performance worsens.
    /// </summary>
    /// <value>The momentum decrease factor, defaulting to 0.95.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> When the model is getting worse, momentum will be decreased by this factor.
    /// A value of 0.95 means momentum becomes 95% of its previous value, helping the model slow down
    /// when it might be heading in the wrong direction.</para>
    /// </remarks>
    public double MomentumDecreaseFactor { get; set; } = 0.95;

    /// <summary>
    /// Gets or sets the minimum allowed momentum value.
    /// </summary>
    /// <value>The minimum momentum, defaulting to 0.5.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This prevents the momentum from becoming too small.
    /// Even if the momentum keeps decreasing, it won't go below this value.</para>
    /// </remarks>
    public double MinMomentum { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets the maximum allowed momentum value.
    /// </summary>
    /// <value>The maximum momentum, defaulting to 0.99.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This prevents the momentum from becoming too large,
    /// which could cause the algorithm to overshoot good solutions.</para>
    /// </remarks>
    public double MaxMomentum { get; set; } = 0.99;

    /// <summary>
    /// Gets or sets the exploration rate for reinforcement learning and some optimization algorithms.
    /// </summary>
    /// <value>The exploration rate, defaulting to 0.5.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This balances between trying new things (exploration) and using what's already known to work (exploitation).
    /// A value of 0.5 means the algorithm spends equal time exploring new possibilities and exploiting known good solutions.</para>
    /// </remarks>
    public double ExplorationRate { get; set; } = 0.5;

    /// <summary>
    /// Gets or sets the minimum allowed exploration rate.
    /// </summary>
    /// <value>The minimum exploration rate, defaulting to 0.1.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This ensures the algorithm always spends at least some time exploring new
    /// possibilities (at least 10% with the default value), even if it's found a good solution. This helps
    /// prevent the algorithm from getting stuck in a suboptimal solution.</para>
    /// </remarks>
    public double MinExplorationRate { get; set; } = 0.1;

    /// <summary>
    /// Gets or sets the maximum allowed exploration rate.
    /// </summary>
    /// <value>The maximum exploration rate, defaulting to 0.9.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> This limits how much time the algorithm spends exploring new possibilities
    /// (at most 90% with the default value). This ensures the algorithm always spends some time using what
    /// it already knows works well.</para>
    /// </remarks>
    public double MaxExplorationRate { get; set; } = 0.9;

    /// <summary>
    /// Gets or sets the convergence tolerance for the optimization algorithm.
    /// </summary>
    /// <value>The convergence tolerance, defaulting to 0.000001 (1e-6).</value>
    /// <remarks>
    /// <para>
    /// Tolerance defines how close the algorithm needs to get to the optimal solution before considering
    /// the optimization process complete. When the change in the objective function between iterations
    /// becomes smaller than this value, the algorithm will stop, assuming it has converged to a solution.
    /// </para>
    /// <para><b>For Beginners:</b> Think of tolerance as the precision level of your answer. A smaller value
    /// (like 0.000001) means the algorithm will try to find a more precise answer, but might take longer.
    /// It's like deciding whether you need to measure something to the nearest millimeter or just the nearest
    /// centimeter. If the improvement between steps becomes smaller than this value, the algorithm decides
    /// it's close enough to the best answer and stops searching.</para>
    /// </remarks>
    public double Tolerance { get; set; } = 1e-6;

    /// <summary>
    /// Gets or sets the optimization mode (feature selection, parameter tuning, or both).
    /// </summary>
    /// <value>The optimization mode, defaulting to Both.</value>
    /// <remarks>
    /// <para>
    /// OptimizationMode determines what aspects of the model the optimizer will modify.
    /// FeatureSelectionOnly only selects which features to use, ParametersOnly only adjusts model parameters,
    /// and Both allows the optimizer to do both.
    /// </para>
    /// <para><b>For Beginners:</b> This controls what the optimizer is allowed to change. It can choose
    /// which features (input variables) to use, adjust the model's internal settings, or do both. The default
    /// is 'Both', which gives the optimizer maximum flexibility to improve your model.</para>
    /// </remarks>
    public OptimizationMode OptimizationMode { get; set; } = OptimizationMode.Both;

    private double _parameterAdjustmentScale = 0.1;

    /// <summary>
    /// Gets or sets the scale factor for parameter adjustments during optimization.
    /// </summary>
    /// <value>The parameter adjustment scale, defaulting to 0.1. Values are automatically clamped to [0.0, 1.0] and invalid values (NaN/Infinity) are rejected.</value>
    /// <remarks>
    /// <para>
    /// ParameterAdjustmentScale controls how much model parameters are changed during each perturbation.
    /// Larger values result in bigger parameter changes, which can speed up exploration but may overshoot
    /// optimal values. Smaller values make smaller changes, leading to more precise but slower optimization.
    /// </para>
    /// <para><b>For Beginners:</b> This controls how big the changes are when the optimizer adjusts your model's
    /// parameters. A value of 0.1 means parameters change by about 10%. Increase this if optimization is too slow,
    /// decrease it if the optimizer seems to be overshooting good solutions.</para>
    /// <para><b>Validation:</b> The value is automatically clamped between 0.0 and 1.0. Invalid values (NaN, Infinity)
    /// will throw an ArgumentException.</para>
    /// </remarks>
    public double ParameterAdjustmentScale
    {
        get => _parameterAdjustmentScale;
        set
        {
            if (double.IsNaN(value) || double.IsInfinity(value))
            {
                throw new ArgumentException("ParameterAdjustmentScale must be a finite number.", nameof(value));
            }
            _parameterAdjustmentScale = Math.Max(0.0, Math.Min(1.0, value));
        }
    }

    private double _signFlipProbability = 0.1;

    /// <summary>
    /// Gets or sets the probability of flipping the sign of a parameter during perturbation.
    /// </summary>
    /// <value>The sign flip probability, defaulting to 0.1 (10% chance). Values are automatically clamped to [0.0, 1.0] and invalid values (NaN/Infinity) are rejected.</value>
    /// <remarks>
    /// <para>
    /// SignFlipProbability determines how often parameter signs are randomly flipped during optimization.
    /// This helps the optimizer explore different regions of the solution space by allowing parameters to
    /// change direction. Value must be between 0 and 1.
    /// </para>
    /// <para><b>For Beginners:</b> Sometimes the optimizer tries flipping a parameter from positive to negative
    /// (or vice versa) to see if that improves the model. This setting controls how often that happens.
    /// A value of 0.1 means there's a 10% chance of flipping each time.</para>
    /// <para><b>Validation:</b> The value is automatically clamped between 0.0 and 1.0. Invalid values (NaN, Infinity)
    /// will throw an ArgumentException.</para>
    /// </remarks>
    public double SignFlipProbability
    {
        get => _signFlipProbability;
        set
        {
            if (double.IsNaN(value) || double.IsInfinity(value))
            {
                throw new ArgumentException("SignFlipProbability must be a finite number.", nameof(value));
            }
            _signFlipProbability = Math.Max(0.0, Math.Min(1.0, value));
        }
    }

    private double _featureSelectionProbability = 0.5;

    /// <summary>
    /// Gets or sets the probability of selecting a feature during feature selection mode.
    /// </summary>
    /// <value>The feature selection probability, defaulting to 0.5 (50% chance). Values are automatically clamped to [0.0, 1.0] and invalid values (NaN/Infinity) are rejected.</value>
    /// <remarks>
    /// <para>
    /// FeatureSelectionProbability controls how likely each feature is to be included when the optimizer
    /// is performing feature selection. Higher values mean more features will typically be selected, while
    /// lower values result in sparser feature sets. Value must be between 0 and 1.
    /// </para>
    /// <para><b>For Beginners:</b> When the optimizer is choosing which features (input variables) to use,
    /// this setting controls how likely each one is to be included. A value of 0.5 means each feature has
    /// a 50/50 chance of being selected. Increase this to use more features, decrease it to use fewer.</para>
    /// <para><b>Validation:</b> The value is automatically clamped between 0.0 and 1.0. Invalid values (NaN, Infinity)
    /// will throw an ArgumentException.</para>
    /// </remarks>
    public double FeatureSelectionProbability
    {
        get => _featureSelectionProbability;
        set
        {
            if (double.IsNaN(value) || double.IsInfinity(value))
            {
                throw new ArgumentException("FeatureSelectionProbability must be a finite number.", nameof(value));
            }
            _featureSelectionProbability = Math.Max(0.0, Math.Min(1.0, value));
        }
    }

    private double _parameterAdjustmentProbability = 0.3;

    /// <summary>
    /// Gets or sets the probability of adjusting a parameter during parameter tuning mode.
    /// </summary>
    /// <value>The parameter adjustment probability, defaulting to 0.3 (30% chance). Values are automatically clamped to [0.0, 1.0] and invalid values (NaN/Infinity) are rejected.</value>
    /// <remarks>
    /// <para>
    /// ParameterAdjustmentProbability determines how likely each parameter is to be modified during
    /// parameter tuning. Lower values result in more conservative updates (fewer parameters changed),
    /// while higher values make more aggressive updates. Value must be between 0 and 1.
    /// </para>
    /// <para><b>For Beginners:</b> When the optimizer is adjusting model parameters, this controls how many
    /// of them get changed at once. A value of 0.3 means each parameter has a 30% chance of being adjusted.
    /// Lower values make smaller, more careful changes; higher values make bigger, bolder changes.</para>
    /// <para><b>Validation:</b> The value is automatically clamped between 0.0 and 1.0. Invalid values (NaN, Infinity)
    /// will throw an ArgumentException.</para>
    /// </remarks>
    public double ParameterAdjustmentProbability
    {
        get => _parameterAdjustmentProbability;
        set
        {
            if (double.IsNaN(value) || double.IsInfinity(value))
            {
                throw new ArgumentException("ParameterAdjustmentProbability must be a finite number.", nameof(value));
            }
            _parameterAdjustmentProbability = Math.Max(0.0, Math.Min(1.0, value));
        }
    }

    /// <summary>
    /// Gets or sets the options for prediction statistics calculation.
    /// </summary>
    /// <value>The prediction statistics options.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> These settings control how predictions are evaluated and what additional
    /// information (like confidence intervals) is included with predictions. This is useful for understanding
    /// how reliable your model's predictions are.</para>
    /// </remarks>
    public PredictionStatsOptions PredictionOptions { get; set; } = new();

    /// <summary>
    /// Gets or sets the options for model statistics calculation.
    /// </summary>
    /// <value>The model statistics options.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> These settings control what statistics are calculated about your model,
    /// such as complexity metrics or feature importance. This helps you understand how your model works
    /// and what factors are most important in its predictions.</para>
    /// </remarks>
    public ModelStatsOptions ModelStatsOptions { get; set; } = new();

    /// <summary>
    /// Gets or sets the fit detector to determine when a model has converged or is overfitting.
    /// </summary>
    /// <value>The fit detector implementation.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The fit detector helps determine when training should stop. It can detect
    /// when your model has learned as much as it can from the data, or when it's starting to memorize the
    /// training data instead of learning general patterns (overfitting).</para>
    /// </remarks>
    public IFitDetector<T, TInput, TOutput> FitDetector { get; set; } = new CalibratedProbabilityFitDetector<T, TInput, TOutput>();

    /// <summary>
    /// Gets or sets the fitness calculator to evaluate model quality during optimization.
    /// </summary>
    /// <value>The fitness calculator implementation.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The fitness calculator assigns a score to your model based on how well it performs.
    /// This score is used by the optimization algorithm to determine which model parameters are better than others,
    /// guiding the learning process toward better solutions.</para>
    /// </remarks>
    public IFitnessCalculator<T, TInput, TOutput> FitnessCalculator { get; set; } = new RSquaredFitnessCalculator<T, TInput, TOutput>();

    /// <summary>
    /// Gets or sets the model cache to store and retrieve previously evaluated models.
    /// </summary>
    /// <value>The model cache implementation.</value>
    /// <remarks>
    /// <para><b>For Beginners:</b> The model cache stores models that have already been evaluated, which can
    /// save time if the same model parameters are encountered again during optimization. This is especially
    /// useful for complex models that take a long time to evaluate.</para>
    /// </remarks>
    public IModelCache<T, TInput, TOutput> ModelCache { get; set; } = new DefaultModelCache<T, TInput, TOutput>();

    /// <summary>
    /// Creates default implementations for the nullable interface objects based on optimizer type.
    /// </summary>
    /// <param name="optimizerType">The type of optimizer to create defaults for.</param>
    /// <returns>An OptimizationAlgorithmOptions<T, TInput, TOutput> instance with appropriate default implementations.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a new instance of OptimizationAlgorithmOptions<T, TInput, TOutput> with default implementations
    /// for all the interface properties based on the specified optimizer type.
    /// </para>
    /// <para><b>For Beginners:</b> This method sets up all the necessary components for your chosen
    /// optimization method. It's like getting a pre-configured toolkit specifically designed for the
    /// type of problem you're trying to solve. You can use this as a starting point and then customize
    /// individual settings as needed.
    /// </para>
    /// </remarks>
    public static OptimizationAlgorithmOptions<T, TInput, TOutput> CreateDefaults(OptimizerType optimizerType)
    {
        var options = new OptimizationAlgorithmOptions<T, TInput, TOutput>
        {
            // Create default implementations based on optimizer type
            PredictionOptions = new PredictionStatsOptions(),
            ModelStatsOptions = new ModelStatsOptions(),
            // Set up common components that exist in our codebase
            FitDetector = new CalibratedProbabilityFitDetector<T, TInput, TOutput>(),
            FitnessCalculator = new RSquaredFitnessCalculator<T, TInput, TOutput>()
        };

        // Apply optimizer-specific settings
        switch (optimizerType)
        {
            case OptimizerType.GradientDescent:
                // Set gradient descent specific defaults
                options.InitialLearningRate = 0.01;
                options.UseAdaptiveLearningRate = true;
                options.LearningRateDecay = 0.99;
                break;

            case OptimizerType.StochasticGradientDescent:
                // Set SGD specific defaults
                options.InitialLearningRate = 0.05;
                options.UseAdaptiveLearningRate = true;
                options.LearningRateDecay = 0.95;
                options.UseAdaptiveMomentum = true;
                break;

            case OptimizerType.AdaptiveGradient:
                // Set adaptive gradient specific defaults
                options.InitialLearningRate = 0.1;
                options.UseAdaptiveLearningRate = true;
                options.MinLearningRate = 1e-4;
                options.UseAdaptiveMomentum = false;
                break;

            case OptimizerType.EvolutionaryAlgorithm:
                // Set evolutionary algorithm specific defaults
                options.ExplorationRate = 0.7;
                options.MaxIterations = 500;
                options.UseEarlyStopping = true;
                options.EarlyStoppingPatience = 20;
                break;

            case OptimizerType.ParticleSwarm:
                // Set particle swarm specific defaults
                options.ExplorationRate = 0.6;
                options.InitialMomentum = 0.8;
                options.MaxIterations = 300;
                options.UseEarlyStopping = true;
                break;

            case OptimizerType.DifferentialEvolution:
                // Set differential evolution specific defaults
                options.ExplorationRate = 0.8;
                options.MaxIterations = 400;
                options.UseEarlyStopping = true;
                options.EarlyStoppingPatience = 25;
                break;

            case OptimizerType.BayesianOptimization:
                // Set Bayesian optimization specific defaults
                options.ExplorationRate = 0.3;
                options.Tolerance = 1e-5;
                options.MaxIterations = 100;
                break;

            case OptimizerType.NelderMead:
                // Set Nelder-Mead specific defaults
                options.Tolerance = 1e-7;
                options.MaxIterations = 200;
                options.UseEarlyStopping = true;
                break;

            case OptimizerType.LBFGS:
                // Set L-BFGS specific defaults
                options.UseAdaptiveLearningRate = false;
                options.UseAdaptiveMomentum = false;
                options.MaxIterations = 150;
                options.Tolerance = 1e-6;
                break;

            case OptimizerType.CoordinateDescent:
                // Set coordinate descent specific defaults
                options.UseAdaptiveLearningRate = true;
                options.InitialLearningRate = 0.05;
                options.MaxIterations = 250;
                options.Tolerance = 1e-6;
                break;

            case OptimizerType.SimulatedAnnealing:
                // Set simulated annealing specific defaults
                options.ExplorationRate = 0.9;
                options.MaxIterations = 500;
                options.UseEarlyStopping = true;
                options.EarlyStoppingPatience = 30;
                break;

            case OptimizerType.AdamOptimizer:
                // Set Adam optimizer specific defaults
                options.InitialLearningRate = 0.001;
                options.UseAdaptiveLearningRate = true;
                options.UseAdaptiveMomentum = true;
                options.MaxIterations = 200;
                break;

            case OptimizerType.RMSProp:
                // Set RMSProp specific defaults
                options.InitialLearningRate = 0.001;
                options.UseAdaptiveLearningRate = true;
                options.LearningRateDecay = 0.9;
                options.MaxIterations = 200;
                break;

            case OptimizerType.GeneticAlgorithm:
                // Set genetic algorithm specific defaults
                options.ExplorationRate = 0.7;
                options.MaxIterations = 500;
                options.UseEarlyStopping = true;
                options.EarlyStoppingPatience = 25;
                break;

            case OptimizerType.TrustRegion:
                // Set trust region specific defaults
                options.Tolerance = 1e-8;
                options.MaxIterations = 150;
                options.UseAdaptiveLearningRate = false;
                break;

            case OptimizerType.NormalOptimizer:
                // Set normal optimizer specific defaults
                options.InitialLearningRate = 0.01;
                options.UseAdaptiveLearningRate = true;
                options.MaxIterations = 100;
                options.Tolerance = 1e-6;
                break;

            case OptimizerType.QuasiNewton:
                // Set quasi-Newton specific defaults
                options.Tolerance = 1e-7;
                options.MaxIterations = 150;
                options.UseAdaptiveLearningRate = false;
                break;

            case OptimizerType.ConjugateGradient:
                // Set conjugate gradient specific defaults
                options.Tolerance = 1e-7;
                options.MaxIterations = 200;
                options.UseAdaptiveLearningRate = false;
                break;

            case OptimizerType.AdaGrad:
                // Set AdaGrad specific defaults
                options.InitialLearningRate = 0.01;
                options.UseAdaptiveLearningRate = true;
                options.UseAdaptiveMomentum = false;
                options.MaxIterations = 200;
                break;

            case OptimizerType.AdaDelta:
                // Set AdaDelta specific defaults
                options.UseAdaptiveLearningRate = true;
                options.UseAdaptiveMomentum = false;
                options.MaxIterations = 200;
                break;

            case OptimizerType.Momentum:
                // Set momentum specific defaults
                options.InitialLearningRate = 0.01;
                options.InitialMomentum = 0.9;
                options.UseAdaptiveMomentum = true;
                options.MaxIterations = 150;
                break;

            case OptimizerType.Nadam:
                // Set Nadam specific defaults
                options.InitialLearningRate = 0.002;
                options.UseAdaptiveLearningRate = true;
                options.UseAdaptiveMomentum = true;
                options.MaxIterations = 200;
                break;

            case OptimizerType.AMSGrad:
                // Set AMSGrad specific defaults
                options.InitialLearningRate = 0.001;
                options.UseAdaptiveLearningRate = true;
                options.UseAdaptiveMomentum = true;
                options.MaxIterations = 200;
                break;

            case OptimizerType.HillClimbing:
                // Set hill climbing specific defaults
                options.ExplorationRate = 0.5;
                options.MaxIterations = 300;
                options.UseEarlyStopping = true;
                break;

            case OptimizerType.CrossEntropy:
                // Set cross-entropy specific defaults
                options.ExplorationRate = 0.6;
                options.MaxIterations = 250;
                options.UseEarlyStopping = true;
                break;

            case OptimizerType.PowellMethod:
                // Set Powell's method specific defaults
                options.Tolerance = 1e-7;
                options.MaxIterations = 200;
                options.UseEarlyStopping = true;
                break;

            default:
                // Default optimizer with standard settings
                options.InitialLearningRate = 0.01;
                options.UseAdaptiveLearningRate = true;
                options.MaxIterations = 100;
                options.Tolerance = 1e-6;
                break;
        }

        return options;
    }
}
