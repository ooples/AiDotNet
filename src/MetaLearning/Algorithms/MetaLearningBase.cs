using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
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

        // Optimizers must be provided in options since they require the model
        InnerOptimizer = options.InnerOptimizer ?? throw new ArgumentNullException(nameof(options.InnerOptimizer), "InnerOptimizer must be provided in options");
        MetaOptimizer = options.MetaOptimizer ?? throw new ArgumentNullException(nameof(options.MetaOptimizer), "MetaOptimizer must be provided in options");
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
            var queryLoss = LossFunction.CalculateLoss(OutputToVector(queryPredictions), OutputToVector(task.QueryOutput));

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
            model.SetParameters(parameters);
            var predictions1 = model.Predict(input);
            T loss1 = LossFunction.CalculateLoss(OutputToVector(predictions1), OutputToVector(expectedOutput));

            // Compute loss with parameter - epsilon
            parameters[i] = NumOps.Subtract(originalValue, epsilon);
            model.SetParameters(parameters);
            var predictions2 = model.Predict(input);
            T loss2 = LossFunction.CalculateLoss(OutputToVector(predictions2), OutputToVector(expectedOutput));

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
        model.SetParameters(parameters);

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
    /// Converts output to vector for loss computation.
    /// </summary>
    /// <param name="output">The output to convert.</param>
    /// <returns>A vector representation of the output.</returns>
    /// <remarks>
    /// This method handles both Vector&lt;T&gt; and Tensor&lt;T&gt; outputs.
    /// </remarks>
    protected Vector<T> OutputToVector(TOutput output)
    {
        // If it's already a Vector<T>, return it as-is
        if (output is Vector<T> vector)
        {
            return vector;
        }

        // If it's a Tensor<T>, convert it to a vector
        if (output is Tensor<T> tensor)
        {
            return tensor.ToVector();
        }

        // For other types, throw an exception
        throw new NotSupportedException($"Output type {output?.GetType()} is not supported for conversion to Vector<T>");
    }

    /// <summary>
    /// Returns the minimum of two values.
    /// </summary>
    /// <param name="a">The first value.</param>
    /// <param name="b">The second value.</param>
    /// <returns>The smaller of the two values.</returns>
    protected T Min(T a, T b)
    {
        // Compare using GreaterThan method
        if (NumOps.GreaterThan(a, b))
            return b;
        return a;
    }

    /// <summary>
    /// Returns the maximum of two values.
    /// </summary>
    /// <param name="a">The first value.</param>
    /// <param name="b">The second value.</param>
    /// <returns>The larger of the two values.</returns>
    protected T Max(T a, T b)
    {
        // Compare using GreaterThan method
        if (NumOps.GreaterThan(a, b))
            return a;
        return b;
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
