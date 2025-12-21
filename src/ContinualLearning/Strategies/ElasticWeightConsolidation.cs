using AiDotNet.ContinualLearning.Interfaces;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.ContinualLearning.Strategies;

/// <summary>
/// Elastic Weight Consolidation (EWC) strategy for continual learning.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <typeparam name="TInput">The input data type.</typeparam>
/// <typeparam name="TOutput">The output data type.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> EWC protects important parameters from changing when learning new tasks.
/// It does this by:
/// 1. Identifying which parameters were important for previous tasks (using Fisher Information)
/// 2. Adding a penalty when those parameters change too much
/// 3. This penalty is added to the loss function during training
/// </para>
///
/// <para><b>How it works:</b>
/// - After learning each task, EWC computes the Fisher Information Matrix (FIM)
/// - The FIM tells us how sensitive the loss is to changes in each parameter
/// - High Fisher Information = parameter is very important for the task
/// - When learning new tasks, EWC adds a regularization term that penalizes changes to important parameters
/// </para>
///
/// <para><b>Reference:</b> Kirkpatrick et al. "Overcoming catastrophic forgetting in neural networks" (2017)</para>
/// </remarks>
public class ElasticWeightConsolidation<T, TInput, TOutput> : IContinualLearningStrategy<T, TInput, TOutput>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();

    private readonly ILossFunction<T> _lossFunction;
    private readonly T _lambda; // Regularization strength
    private readonly int _numFisherSamples;

    // Stored parameters and Fisher Information from previous tasks
    private Vector<T>? _optimalParameters;
    private Vector<T>? _fisherInformation;

    /// <summary>
    /// Initializes a new EWC strategy.
    /// </summary>
    /// <param name="lossFunction">The loss function to use.</param>
    /// <param name="lambda">Regularization strength (higher = more protection of old tasks).</param>
    /// <param name="numFisherSamples">Number of samples to use for computing Fisher Information.</param>
    public ElasticWeightConsolidation(
        ILossFunction<T> lossFunction,
        T lambda,
        int numFisherSamples = 200)
    {
        _lossFunction = lossFunction ?? throw new ArgumentNullException(nameof(lossFunction));
        _lambda = lambda;
        _numFisherSamples = numFisherSamples;
    }

    /// <inheritdoc/>
    public void PrepareForTask(IFullModel<T, TInput, TOutput> model, IDataset<T, TInput, TOutput> taskData)
    {
        // No preparation needed before training a new task
        // EWC uses information stored from previous tasks
    }

    /// <inheritdoc/>
    public T ComputeRegularizationLoss(IFullModel<T, TInput, TOutput> model)
    {
        // If no previous task, no regularization
        if (_optimalParameters == null || _fisherInformation == null)
            return NumOps.Zero;

        var currentParams = model.GetParameters();

        if (currentParams.Length != _optimalParameters.Length)
            throw new InvalidOperationException("Parameter dimensions do not match");

        // EWC loss: (lambda/2) * sum_i F_i * (theta_i - theta*_i)^2
        // where F_i is Fisher Information, theta_i is current parameter, theta*_i is optimal parameter from previous task

        T loss = NumOps.Zero;

        for (int i = 0; i < currentParams.Length; i++)
        {
            var paramDiff = NumOps.Subtract(currentParams[i], _optimalParameters[i]);
            var squaredDiff = NumOps.Multiply(paramDiff, paramDiff);
            var weightedDiff = NumOps.Multiply(_fisherInformation[i], squaredDiff);
            loss = NumOps.Add(loss, weightedDiff);
        }

        // Multiply by lambda/2
        var halfLambda = NumOps.Divide(_lambda, NumOps.FromDouble(2.0));
        loss = NumOps.Multiply(halfLambda, loss);

        return loss;
    }

    /// <inheritdoc/>
    public Vector<T> AdjustGradients(Vector<T> gradients)
    {
        // EWC doesn't adjust gradients directly - it adds regularization to the loss
        // The gradient adjustment happens automatically through backpropagation
        return gradients;
    }

    /// <inheritdoc/>
    public void FinalizeTask(IFullModel<T, TInput, TOutput> model)
    {
        // Store the optimal parameters from this task
        _optimalParameters = model.GetParameters().Clone();

        // Compute and store Fisher Information
        // Note: This is a simplified diagonal Fisher Information Matrix
        // For a full implementation, you would need access to gradients during training

        // For now, we'll initialize with uniform importance
        // In a complete implementation, this would compute:
        // F_i = E[(d log P(y|x,theta) / d theta_i)^2]

        if (_fisherInformation == null)
        {
            _fisherInformation = new Vector<T>(_optimalParameters!.Length);
            for (int i = 0; i < _fisherInformation.Length; i++)
            {
                _fisherInformation[i] = NumOps.FromDouble(1.0);
            }
        }
        else
        {
            // Accumulate Fisher Information across tasks
            // F_total = F_old + F_new
            var newFisher = ComputeFisherInformation(model);
            for (int i = 0; i < _fisherInformation.Length; i++)
            {
                _fisherInformation[i] = NumOps.Add(_fisherInformation[i], newFisher[i]);
            }
        }
    }

    /// <summary>
    /// Computes the diagonal Fisher Information Matrix.
    /// </summary>
    /// <remarks>
    /// <para><b>For Implementers:</b> The Fisher Information Matrix measures the sensitivity
    /// of the model's predictions to parameter changes. For each parameter theta_i:
    /// F_i = E[(d log P(y|x,theta) / d theta_i)^2]
    /// </para>
    ///
    /// <para>In practice, this is computed as:
    /// 1. Sample data from the task
    /// 2. Compute gradients of log probability with respect to parameters
    /// 3. Square the gradients and average over samples
    /// </para>
    ///
    /// <para><b>Note:</b> This is a simplified implementation. A full implementation would require
    /// access to gradient computation during training.</para>
    /// </remarks>
    private Vector<T> ComputeFisherInformation(IFullModel<T, TInput, TOutput> model)
    {
        int numParams = model.ParameterCount;
        var fisher = new Vector<T>(numParams);

        // Initialize with small positive values to avoid division by zero
        for (int i = 0; i < numParams; i++)
        {
            fisher[i] = NumOps.FromDouble(0.01);
        }

        // In a full implementation:
        // 1. Sample _numFisherSamples examples from the task data
        // 2. For each sample, compute gradient of log likelihood
        // 3. Square gradients and accumulate
        // 4. Average over samples

        // Placeholder: return uniform importance
        // This should be replaced with actual Fisher computation when gradient access is available

        return fisher;
    }

    /// <summary>
    /// Gets the stored optimal parameters from previous tasks.
    /// </summary>
    public Vector<T>? OptimalParameters => _optimalParameters;

    /// <summary>
    /// Gets the stored Fisher Information.
    /// </summary>
    public Vector<T>? FisherInformation => _fisherInformation;
}
