using AiDotNet.Data.Abstractions;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.MetaLearning.Config;

namespace AiDotNet.MetaLearning.Trainers;

/// <summary>
/// Implements the Reptile meta-learning algorithm for training models to quickly adapt to new tasks.
/// </summary>
/// <typeparam name="T">The numeric data type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// Reptile is a simple and effective meta-learning algorithm published by OpenAI in 2018.
/// It provides a first-order approximation to MAML (Model-Agnostic Meta-Learning) without
/// requiring second-order derivatives, making it simpler to implement and more computationally efficient.
/// </para>
/// <para><b>For Beginners:</b> Reptile teaches a model to be a "quick learner."
///
/// The algorithm is beautifully simple:
/// 1. Start with some initial parameters (your meta-model)
/// 2. For each task:
///    - Make a copy of the current parameters
///    - Train the copy on this task's examples (support set)
///    - Take a small step from the original parameters toward the trained copy
/// 3. Repeat for many tasks
///
/// Why this works:
/// - After seeing many tasks, the parameters naturally settle at a point that's easy to fine-tune
/// - The model learns to find a "good starting point" for quick adaptation
/// - It's like finding a central location that's close to many different destinations
///
/// Key advantages:
/// - <b>Simple:</b> Just parameter averaging, no complex gradients
/// - <b>Efficient:</b> Faster than MAML, no second-order derivatives
/// - <b>Effective:</b> Achieves strong few-shot learning performance
/// - <b>Flexible:</b> Works with any model that has learnable parameters
///
/// When to use Reptile:
/// - Few-shot learning: When you need to adapt to new tasks with limited data
/// - Transfer learning: When you have many related tasks
/// - Quick adaptation: When you need models that can learn fast from few examples
///
/// Example applications:
/// - Character recognition: Learn new alphabets from few examples
/// - Robot control: Adapt to new environments quickly
/// - Drug discovery: Predict properties of new molecules from limited data
/// </para>
/// <para>
/// <b>Thread Safety:</b> This class is not thread-safe. Create separate instances for concurrent training.
/// </para>
/// <para>
/// <b>Performance:</b> O(T × K × P) where T is tasks per iteration, K is inner steps, P is model parameters.
/// Memory usage scales with model size due to parameter cloning.
/// </para>
/// </remarks>
/// <example>
/// <code>
/// // Prepare data for 5-way 3-shot learning
/// var features = new Matrix&lt;double&gt;(1000, 784);  // 1000 examples, 784 features
/// var labels = new Vector&lt;double&gt;(1000);         // 10 classes
///
/// // Create episodic data loader
/// var dataLoader = new UniformEpisodicDataLoader&lt;double&gt;(
///     datasetX: features,
///     datasetY: labels,
///     nWay: 5,        // 5 classes per task
///     kShot: 3,       // 3 training examples per class
///     queryShots: 10  // 10 test examples per class
/// );
///
/// // Create a neural network model
/// var model = new SimpleNeuralNetwork&lt;double&gt;(inputSize: 784, hiddenSize: 128, outputSize: 5);
///
/// // Create Reptile trainer
/// var trainer = new ReptileTrainer&lt;double&gt;(
///     metaModel: model,
///     lossFunction: new MeanSquaredError&lt;double&gt;(),
///     innerSteps: 5,              // 5 gradient steps per task
///     innerLearningRate: 0.01,    // Learning rate for task adaptation
///     metaLearningRate: 0.001     // Meta-learning rate for Reptile updates
/// );
///
/// // Meta-train for 1000 iterations
/// var metadata = trainer.Train(dataLoader, numMetaIterations: 1000);
///
/// // Now the model can quickly adapt to new tasks
/// Console.WriteLine($"Meta-training complete. Final loss: {metadata.FinalLoss}");
///
/// // To adapt to a new task:
/// var newTask = dataLoader.GetNextTask();
/// for (int step = 0; step &lt; 5; step++)
/// {
///     model.Train(newTask.SupportSetX, newTask.SupportSetY);
/// }
/// var predictions = model.Predict(newTask.QuerySetX);
/// </code>
/// </example>
public class ReptileTrainer<T> : ReptileTrainerBase<T>
{
    /// <summary>
    /// Initializes a new instance of the ReptileTrainer class with industry-standard defaults.
    /// </summary>
    /// <param name="metaModel">The model to be meta-trained. Must implement IFullModel for parameter access and cloning.</param>
    /// <param name="lossFunction">The loss function used to evaluate predictions during training.</param>
    /// <param name="innerSteps">The number of gradient steps per task. Default is 5 (common in meta-learning).</param>
    /// <param name="innerLearningRate">The learning rate for task-specific training. Default is 0.01.</param>
    /// <param name="metaLearningRate">The meta-learning rate for Reptile updates. Default is 0.001.</param>
    /// <exception cref="ArgumentNullException">Thrown when metaModel or lossFunction is null.</exception>
    /// <exception cref="ArgumentException">Thrown when configuration parameters are invalid.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This creates a Reptile trainer ready for meta-learning.
    ///
    /// All parameters have sensible defaults based on meta-learning research:
    /// - <b>innerSteps (5):</b> Enough to adapt to a task without overfitting
    /// - <b>innerLearningRate (0.01):</b> Conservative rate for stable task-specific training
    /// - <b>metaLearningRate (0.001):</b> 10x smaller than inner LR, prevents forgetting between tasks
    ///
    /// You can start with these defaults and adjust based on your results:
    /// - Task adaptation too slow? Increase innerSteps or innerLearningRate
    /// - Meta-training unstable? Decrease metaLearningRate
    /// - Meta-training too slow? Increase metaLearningRate (carefully)
    /// </para>
    /// </remarks>
    public ReptileTrainer(
        IFullModel<T, Tensor<T>, Tensor<T>> metaModel,
        ILossFunction<T> lossFunction,
        int innerSteps = 5,
        double innerLearningRate = 0.01,
        double metaLearningRate = 0.001)
        : base(metaModel, lossFunction, innerSteps, innerLearningRate, metaLearningRate)
    {
    }

    /// <summary>
    /// Initializes a new instance of the ReptileTrainer class with a configuration object.
    /// </summary>
    /// <param name="metaModel">The model to be meta-trained. Must implement IFullModel for parameter access and cloning.</param>
    /// <param name="lossFunction">The loss function used to evaluate predictions during training.</param>
    /// <param name="config">Configuration object specifying inner/outer learning rates and training parameters.</param>
    /// <exception cref="ArgumentNullException">Thrown when metaModel, lossFunction, or config is null.</exception>
    /// <exception cref="ArgumentException">Thrown when configuration is invalid.</exception>
    /// <remarks>
    /// <para><b>For Beginners:</b> This constructor accepts a configuration object for cleaner code:
    ///
    /// <code>
    /// // Create configuration
    /// var config = new ReptileTrainerConfig&lt;double&gt;
    /// {
    ///     InnerLearningRate = 0.02,
    ///     MetaLearningRate = 0.001,
    ///     InnerSteps = 10
    /// };
    ///
    /// // Create trainer with config
    /// var trainer = new ReptileTrainer&lt;double&gt;(model, lossFunction, config);
    /// </code>
    ///
    /// Benefits of using configuration objects:
    /// - Easier to manage multiple hyperparameters
    /// - Can be saved/loaded for reproducibility
    /// - Can be shared across multiple trainers
    /// - Validated once at construction
    /// </para>
    /// </remarks>
    public ReptileTrainer(
        IFullModel<T, Tensor<T>, Tensor<T>> metaModel,
        ILossFunction<T> lossFunction,
        IMetaLearnerConfig<T> config)
        : base(metaModel, lossFunction, config)
    {
    }

    /// <summary>
    /// Implements the Reptile meta-learning algorithm.
    /// </summary>
    /// <param name="dataLoader">The episodic data loader that provides meta-learning tasks.</param>
    /// <param name="numMetaIterations">The number of meta-training iterations to perform.</param>
    /// <returns>Metadata about the meta-training process including loss history and performance metrics.</returns>
    /// <remarks>
    /// <para>
    /// This method implements the core Reptile algorithm:
    ///
    /// <code>
    /// Algorithm Reptile:
    ///   Initialize θ (meta-parameters)
    ///   for iteration = 1 to N:
    ///     Sample task T_i from task distribution
    ///     θ_i = θ  (clone parameters)
    ///     for k = 1 to K:  (inner loop)
    ///       θ_i = θ_i - α∇L(θ_i, D_support)  (task-specific training)
    ///     θ = θ + ε(θ_i - θ)  (meta-update toward adapted parameters)
    ///   return θ
    /// </code>
    ///
    /// The key insight is that repeatedly moving toward task-adapted parameters causes
    /// the meta-parameters to converge to a point that's easy to fine-tune for any task.
    /// </para>
    /// <para><b>For Beginners:</b> This is where the actual "learning to learn" happens.
    ///
    /// What happens during meta-training:
    /// 1. <b>Sample a task:</b> Get a new few-shot learning problem from the data loader
    /// 2. <b>Clone the model:</b> Make a copy to train on this specific task
    /// 3. <b>Inner loop (task adaptation):</b> Train the copy on the task's support set for K steps
    /// 4. <b>Meta-update:</b> Move the original model's parameters slightly toward the adapted copy
    /// 5. <b>Repeat:</b> Do this for many tasks
    ///
    /// After many iterations, the model's parameters naturally settle at a location that's:
    /// - Easy to fine-tune for any task in the distribution
    /// - A good "starting point" for quick adaptation
    /// - Robust to the specific task variations
    ///
    /// Progress tracking:
    /// - Loss is computed on query sets to measure adaptation quality
    /// - Every 10% of training, progress is recorded
    /// - Final metadata includes loss history and training time
    /// </para>
    /// <para>
    /// <b>Performance:</b> For N iterations with K inner steps and P parameters:
    /// - Time complexity: O(N × K × P × C) where C is model forward/backward cost
    /// - Space complexity: O(P) for parameter cloning
    /// - Each iteration samples one task and performs K gradient updates
    /// </para>
    /// </remarks>
    protected override ModelMetadata<T> TrainCore(IEpisodicDataLoader<T> dataLoader, int numMetaIterations)
    {
        var lossHistory = new List<T>();
        var startTime = DateTime.Now;

        // Meta-training loop
        for (int iteration = 0; iteration < numMetaIterations; iteration++)
        {
            // Step 1: Sample a task from the episodic data loader
            MetaLearningTask<T> task = dataLoader.GetNextTask();

            // Step 2: Clone the meta-model for task-specific training
            // Save the original parameters before task-specific training
            Vector<T> metaParameters = MetaModel.GetParameters();
            Vector<T> originalParameters = metaParameters.Copy();

            // Step 3: Inner loop - Train on the support set for K steps
            for (int step = 0; step < InnerSteps; step++)
            {
                // Train the model on the support set
                // The model's Train method will update its internal parameters
                MetaModel.Train(task.SupportSetX, task.SupportSetY);
            }

            // Step 4: Get the adapted parameters after inner loop training
            Vector<T> adaptedParameters = MetaModel.GetParameters();

            // Step 5: Reptile meta-update: θ ← θ + ε(θ_adapted - θ)
            // Compute the difference: θ_adapted - θ_original
            Vector<T> parameterDifference = adaptedParameters.Subtract(originalParameters);

            // Scale by meta-learning rate: ε × (θ_adapted - θ_original)
            Vector<T> scaledDifference = parameterDifference.Multiply(MetaLearningRate);

            // Update meta-parameters: θ_new = θ_original + ε × (θ_adapted - θ_original)
            Vector<T> newMetaParameters = originalParameters.Add(scaledDifference);

            // Set the updated parameters back to the model
            MetaModel.SetParameters(newMetaParameters);

            // Step 6: Evaluate on query set and track progress
            T queryLoss = ComputeLoss(MetaModel, task.QuerySetX, task.QuerySetY);
            lossHistory.Add(queryLoss);

            // Log progress every 10% of training
            if ((iteration + 1) % Math.Max(1, numMetaIterations / 10) == 0)
            {
                double percentComplete = 100.0 * (iteration + 1) / numMetaIterations;
                Console.WriteLine($"Reptile: {percentComplete:F1}% complete, Query Loss: {queryLoss}");
            }
        }

        // Create and return metadata
        var endTime = DateTime.Now;
        var trainingTime = endTime - startTime;

        return new ModelMetadata<T>
        {
            TrainingTime = trainingTime,
            Iterations = numMetaIterations,
            FinalLoss = lossHistory.Count > 0 ? lossHistory[lossHistory.Count - 1] : NumOps.Zero,
            LossHistory = lossHistory,
            ConvergenceAchieved = true,
            ModelType = "ReptileTrainer"
        };
    }
}
