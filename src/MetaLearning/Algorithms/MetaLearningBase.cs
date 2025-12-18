using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.MetaLearning.Data;
using AiDotNet.Models;
using AiDotNet.Models.Options;

namespace AiDotNet.MetaLearning.Algorithms;

/// <summary>
/// Base class for meta-learning algorithms.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <typeparam name="TInput">The input data type (e.g., Matrix<T>, Tensor<T>).</typeparam>
/// <typeparam name="TOutput">The output data type (e.g., Vector<T>, Tensor<T>).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This base class provides common functionality for all meta-learning algorithms.
/// Meta-learning algorithms learn to learn - they practice adapting to new tasks quickly by
/// training on many different tasks.
/// </para>
/// </remarks>
public abstract class MetaLearningBase<T, TInput, TOutput> : IMetaLearningAlgorithm<T, TInput, TOutput>
{
    protected readonly INumericOperations<T> NumOps;
    protected IFullModel<T, TInput, TOutput> MetaModel;
    protected ILossFunction<T> LossFunction;
    protected readonly MetaLearningAlgorithmOptions<T, TInput, TOutput> Options;
    protected Random? RandomGenerator;
    protected readonly GradientBasedOptimizerBase<T, TInput, TOutput> InnerOptimizer;
    protected readonly GradientBasedOptimizerBase<T, TInput, TOutput> MetaOptimizer;

    /// <summary>
    /// Initializes a new instance of the MetaLearningBase class.
    /// </summary>
    /// <param name="options">The configuration options for the meta-learning algorithm.</param>
    protected MetaLearningBase(MetaLearningAlgorithmOptions<T, TInput, TOutput> options)
    {
        Options = options ?? throw new ArgumentNullException(nameof(options));
        NumOps = MathHelper.GetNumericOperations<T>();

        if (options.BaseModel == null)
        {
            throw new ArgumentException("BaseModel cannot be null in meta-learning options.", nameof(options));
        }

        MetaModel = options.BaseModel;
        LossFunction = options.LossFunction ?? throw new ArgumentException("LossFunction cannot be null.", nameof(options));

        RandomGenerator = options.RandomSeed.HasValue ? new Random(options.RandomSeed.Value) : new Random();

        // Initialize optimizers with default SGD if not provided
        InnerOptimizer = options.InnerOptimizer ?? new StochasticGradientDescentOptimizer<T, TInput, TOutput>(new StochasticGradientDescentOptimizerOptions<T, TInput, TOutput> { InitialLearningRate = options.InnerLearningRate });
        MetaOptimizer = options.MetaOptimizer ?? new StochasticGradientDescentOptimizer<T, TInput, TOutput>(new StochasticGradientDescentOptimizerOptions<T, TInput, TOutput> { InitialLearningRate = options.OuterLearningRate });
    }

    /// <inheritdoc/>
    public abstract string AlgorithmName { get; }

    /// <inheritdoc/>
    public int AdaptationSteps => Options.AdaptationSteps;

    /// <inheritdoc/>
    public double InnerLearningRate => Options.InnerLearningRate;

    /// <inheritdoc/>
    public double OuterLearningRate => Options.OuterLearningRate;

    /// <inheritdoc/>
    public abstract T MetaTrain(TaskBatch<T, TInput, TOutput> taskBatch);

    /// <inheritdoc/>
    public abstract IModel<TInput, TOutput, ModelMetadata<T>> Adapt(ITask<T, TInput, TOutput> task);

    /// <inheritdoc/>
    public virtual T Evaluate(TaskBatch<T, TInput, TOutput> taskBatch)
    {
        T totalLoss = NumOps.Zero;
        int taskCount = 0;

        foreach (var task in taskBatch.Tasks)
        {
            // Adapt to the task using support set
            var adaptedModel = Adapt(task);

            // Evaluate on query set
            var queryPredictions = adaptedModel.Predict(task.QueryInput);
            var queryLoss = LossFunction.ComputeLoss(queryPredictions, task.QueryOutput);

            totalLoss = NumOps.Add(totalLoss, queryLoss);
            taskCount++;
        }

        // Return average loss
        return NumOps.Divide(totalLoss, NumOps.FromDouble(taskCount));
    }

    /// <inheritdoc/>
    public IFullModel<T, TInput, TOutput> GetMetaModel()
    {
        return MetaModel;
    }

    /// <inheritdoc/>
    public void SetMetaModel(IFullModel<T, TInput, TOutput> model)
    {
        MetaModel = model ?? throw new ArgumentNullException(nameof(model));
    }

    /// <summary>
    /// Computes gradients for a single task.
    /// </summary>
    /// <param name="model">The model to compute gradients for.</param>
    /// <param name="input">The input data.</param>
    /// <param name="expectedOutput">The expected output.</param>
    /// <returns>The gradient vector.</returns>
    protected Vector<T> ComputeGradients(IFullModel<T, TInput, TOutput> model, TInput input, TOutput expectedOutput)
    {
        // Get current parameters
        var parameters = model.GetParameters();
        int paramCount = parameters.Length;
        var gradients = new Vector<T>(paramCount);

        // Numerical gradient computation using finite differences
        T epsilon = NumOps.FromDouble(1e-5);

        for (int i = 0; i < paramCount; i++)
        {
            // Save original value
            T originalValue = parameters[i];

            // Compute loss with parameter + epsilon
            parameters[i] = NumOps.Add(originalValue, epsilon);
            model.UpdateParameters(parameters);
            var predictions1 = model.Predict(input);
            T loss1 = LossFunction.ComputeLoss(predictions1, expectedOutput);

            // Compute loss with parameter - epsilon
            parameters[i] = NumOps.Subtract(originalValue, epsilon);
            model.UpdateParameters(parameters);
            var predictions2 = model.Predict(input);
            T loss2 = LossFunction.ComputeLoss(predictions2, expectedOutput);

            // Compute gradient using central difference
            T gradient = NumOps.Divide(
                NumOps.Subtract(loss1, loss2),
                NumOps.Multiply(NumOps.FromDouble(2.0), epsilon)
            );
            gradients[i] = gradient;

            // Restore original value
            parameters[i] = originalValue;
        }

        // Restore original parameters
        model.UpdateParameters(parameters);

        return gradients;
    }

    /// <summary>
    /// Applies gradients to model parameters with a given learning rate.
    /// </summary>
    /// <param name="parameters">The current parameters.</param>
    /// <param name="gradients">The gradients to apply.</param>
    /// <param name="learningRate">The learning rate.</param>
    /// <returns>The updated parameters.</returns>
    protected Vector<T> ApplyGradients(Vector<T> parameters, Vector<T> gradients, double learningRate)
    {
        T lr = NumOps.FromDouble(learningRate);
        var updatedParameters = new Vector<T>(parameters.Length);

        for (int i = 0; i < parameters.Length; i++)
        {
            updatedParameters[i] = NumOps.Subtract(
                parameters[i],
                NumOps.Multiply(lr, gradients[i])
            );
        }

        return updatedParameters;
    }

    /// <summary>
    /// Clips gradients to prevent exploding gradients.
    /// </summary>
    /// <param name="gradients">The gradients to clip.</param>
    /// <param name="threshold">The clipping threshold.</param>
    /// <returns>The clipped gradients.</returns>
    protected Vector<T> ClipGradients(Vector<T> gradients, double threshold)
    {
        if (threshold <= 0)
        {
            return gradients;
        }

        // Compute gradient norm
        T sumSquares = NumOps.Zero;
        for (int i = 0; i < gradients.Length; i++)
        {
            sumSquares = NumOps.Add(sumSquares, NumOps.Multiply(gradients[i], gradients[i]));
        }
        T norm = NumOps.Sqrt(sumSquares);

        // Clip if norm exceeds threshold
        T thresholdValue = NumOps.FromDouble(threshold);
        if (Convert.ToDouble(norm) > threshold)
        {
            T scale = NumOps.Divide(thresholdValue, norm);
            var clippedGradients = new Vector<T>(gradients.Length);
            for (int i = 0; i < gradients.Length; i++)
            {
                clippedGradients[i] = NumOps.Multiply(gradients[i], scale);
            }
            return clippedGradients;
        }

        return gradients;
    }

    /// <summary>
    /// Creates a deep copy of the meta model.
    /// </summary>
    /// <returns>A cloned instance of the meta model.</returns>
    protected IFullModel<T, TInput, TOutput> CloneModel()
    {
        return MetaModel.Clone();
    }
}
