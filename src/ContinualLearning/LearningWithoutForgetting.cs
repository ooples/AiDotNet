using AiDotNet.Helpers;
using AiDotNet.Interfaces;

namespace AiDotNet.ContinualLearning;

/// <summary>
/// Implements Learning without Forgetting (LwF) for continual learning.
/// </summary>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
/// <remarks>
/// <para><b>For Beginners:</b> Learning without Forgetting is like teaching a student to solve
/// new problems while making sure they remember how to solve old ones. It does this by asking
/// the student to match their old answers (before learning new material) even as they learn
/// new things.</para>
///
/// <para><b>How it works:</b></para>
/// <list type="number">
/// <item><description>Before learning a new task, LwF records the network's predictions (soft targets)
/// on the new task's inputs using the old model.</description></item>
/// <item><description>During training on the new task, the loss function includes both:
/// the regular task loss AND a distillation loss that encourages matching the old predictions.</description></item>
/// <item><description>The distillation loss uses temperature-scaled softmax to capture the
/// relationships between classes, not just the predicted class.</description></item>
/// </list>
///
/// <para><b>Reference:</b> Li and Hoiem, "Learning without Forgetting" (2017). IEEE TPAMI.</para>
/// </remarks>
public class LearningWithoutForgetting<T> : IContinualLearningStrategy<T>
{
    private readonly INumericOperations<T> _numOps;
    private readonly Dictionary<int, Tensor<T>> _oldPredictions;
    private Tensor<T>? _currentTaskInputs;
    private double _lambda;
    private double _temperature;

    /// <summary>
    /// Initializes a new instance of the LearningWithoutForgetting class.
    /// </summary>
    /// <param name="lambda">The weight of the distillation loss (default: 1.0).</param>
    /// <param name="temperature">Temperature for softmax distillation (default: 2.0).</param>
    /// <remarks>
    /// <para><b>For Beginners:</b></para>
    /// <list type="bullet">
    /// <item><description>Lambda: How much to weight the "don't forget" loss compared to
    /// the "learn new task" loss. Higher values preserve more old knowledge.</description></item>
    /// <item><description>Temperature: Controls how "soft" the predictions are. Higher temperature
    /// makes the probability distribution smoother, which helps transfer more nuanced knowledge
    /// from the old model to the new one.</description></item>
    /// </list>
    /// </remarks>
    public LearningWithoutForgetting(double lambda = 1.0, double temperature = 2.0)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _oldPredictions = [];
        _lambda = lambda;
        _temperature = temperature;
    }

    /// <inheritdoc />
    public double Lambda
    {
        get => _lambda;
        set => _lambda = value;
    }

    /// <summary>
    /// Gets or sets the temperature for knowledge distillation.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> Temperature controls how "soft" the probability distribution becomes:</para>
    /// <list type="bullet">
    /// <item><description>T = 1: Normal softmax (sharp, peaked distribution)</description></item>
    /// <item><description>T = 2-5: Softer distribution that reveals class relationships</description></item>
    /// <item><description>T > 5: Very soft, almost uniform distribution</description></item>
    /// </list>
    /// <para>Typical values for LwF are 2-4.</para>
    /// </remarks>
    public double Temperature
    {
        get => _temperature;
        set => _temperature = Math.Max(0.1, value);
    }

    /// <summary>
    /// Gets the number of tasks that have stored predictions.
    /// </summary>
    public int TaskCount => _oldPredictions.Count;

    /// <inheritdoc />
    public void BeforeTask(INeuralNetwork<T> network, int taskId)
    {
        _ = network ?? throw new ArgumentNullException(nameof(network));
        // LwF preparation happens through PrepareDistillation when you have the new task data
    }

    /// <summary>
    /// Prepares distillation by recording the old model's predictions on new task inputs.
    /// </summary>
    /// <param name="network">The neural network before training on the new task.</param>
    /// <param name="newTaskInputs">The inputs for the new task.</param>
    /// <param name="taskId">The identifier for the new task.</param>
    /// <remarks>
    /// <para><b>For Beginners:</b> Call this method before training on a new task. It records
    /// what the network currently predicts for the new task's inputs. These predictions become
    /// the "soft targets" that the network tries to match during training, preventing it from
    /// forgetting its old behavior.</para>
    /// </remarks>
    public void PrepareDistillation(INeuralNetwork<T> network, Tensor<T> newTaskInputs, int taskId)
    {
        _ = network ?? throw new ArgumentNullException(nameof(network));
        _ = newTaskInputs ?? throw new ArgumentNullException(nameof(newTaskInputs));

        _currentTaskInputs = newTaskInputs.Clone();

        // Record the network's current predictions (before training on new task)
        network.SetTrainingMode(false);
        var predictions = network.Predict(newTaskInputs);
        _oldPredictions[taskId] = predictions.Clone();
        network.SetTrainingMode(true);
    }

    /// <inheritdoc />
    public void AfterTask(INeuralNetwork<T> network, (Tensor<T> inputs, Tensor<T> targets) taskData, int taskId)
    {
        // LwF doesn't need to do anything after task completion
        // The distillation targets were already captured in PrepareDistillation
        _currentTaskInputs = null;
    }

    /// <inheritdoc />
    public T ComputeLoss(INeuralNetwork<T> network)
    {
        _ = network ?? throw new ArgumentNullException(nameof(network));

        if (_oldPredictions.Count == 0 || _currentTaskInputs == null)
        {
            return _numOps.Zero;
        }

        // Get current network predictions on the task inputs
        var currentPredictions = network.Predict(_currentTaskInputs);

        // Compute distillation loss for each stored task
        var totalLoss = _numOps.Zero;
        foreach (var (taskId, oldPreds) in _oldPredictions)
        {
            var distillLoss = ComputeDistillationLoss(currentPredictions, oldPreds);
            totalLoss = _numOps.Add(totalLoss, distillLoss);
        }

        // Scale by lambda
        var lambdaT = _numOps.FromDouble(_lambda);
        return _numOps.Multiply(lambdaT, totalLoss);
    }

    /// <summary>
    /// Computes the distillation loss between current and old predictions.
    /// </summary>
    /// <param name="currentPredictions">The network's current predictions.</param>
    /// <param name="oldPredictions">The network's predictions before learning new tasks.</param>
    /// <returns>The distillation loss (KL divergence with temperature scaling).</returns>
    public T ComputeDistillationLoss(Tensor<T> currentPredictions, Tensor<T> oldPredictions)
    {
        _ = currentPredictions ?? throw new ArgumentNullException(nameof(currentPredictions));
        _ = oldPredictions ?? throw new ArgumentNullException(nameof(oldPredictions));

        var batchSize = currentPredictions.Shape[0];
        var numClasses = currentPredictions.Length / batchSize;

        var loss = _numOps.Zero;
        var tempT = _numOps.FromDouble(_temperature);
        var epsilon = _numOps.FromDouble(1e-10);

        for (int b = 0; b < batchSize; b++)
        {
            // Apply temperature-scaled softmax to both predictions
            var currentSoft = TemperatureSoftmax(currentPredictions, b, numClasses);
            var oldSoft = TemperatureSoftmax(oldPredictions, b, numClasses);

            // KL divergence: sum(old * log(old / current))
            for (int c = 0; c < numClasses; c++)
            {
                var oldProb = oldSoft[c];
                var currentProb = _numOps.Add(currentSoft[c], epsilon);

                // old * log(old) - old * log(current)
                var oldPlusEps = _numOps.Add(oldProb, epsilon);
                var logOld = _numOps.FromDouble(Math.Log(_numOps.ToDouble(oldPlusEps)));
                var logCurrent = _numOps.FromDouble(Math.Log(_numOps.ToDouble(currentProb)));

                var term1 = _numOps.Multiply(oldProb, logOld);
                var term2 = _numOps.Multiply(oldProb, logCurrent);
                var klTerm = _numOps.Subtract(term1, term2);

                loss = _numOps.Add(loss, klTerm);
            }
        }

        // Scale by T^2 (as per Hinton et al.) and average over batch
        var tSquared = _numOps.Multiply(tempT, tempT);
        var batchSizeT = _numOps.FromDouble(batchSize);
        loss = _numOps.Multiply(loss, tSquared);
        return _numOps.Divide(loss, batchSizeT);
    }

    /// <inheritdoc />
    public Vector<T> ModifyGradients(INeuralNetwork<T> network, Vector<T> gradients)
    {
        // LwF uses loss-based regularization, not gradient modification
        return gradients;
    }

    /// <inheritdoc />
    public void Reset()
    {
        _oldPredictions.Clear();
        _currentTaskInputs = null;
    }

    /// <summary>
    /// Applies temperature-scaled softmax to logits for a single sample.
    /// </summary>
    private Vector<T> TemperatureSoftmax(Tensor<T> logits, int batchIndex, int numClasses)
    {
        var result = new Vector<T>(numClasses);
        var startIdx = batchIndex * numClasses;
        var tempInv = 1.0 / _temperature;

        // Find max for numerical stability
        var maxLogit = _numOps.MinValue;
        for (int c = 0; c < numClasses; c++)
        {
            var scaledLogit = _numOps.Multiply(logits[startIdx + c], _numOps.FromDouble(tempInv));
            if (_numOps.GreaterThan(scaledLogit, maxLogit))
            {
                maxLogit = scaledLogit;
            }
        }

        // Compute exp(x/T - max) and sum
        var expSum = _numOps.Zero;
        for (int c = 0; c < numClasses; c++)
        {
            var scaledLogit = _numOps.Multiply(logits[startIdx + c], _numOps.FromDouble(tempInv));
            var shifted = _numOps.Subtract(scaledLogit, maxLogit);
            var expVal = _numOps.FromDouble(Math.Exp(_numOps.ToDouble(shifted)));
            result[c] = expVal;
            expSum = _numOps.Add(expSum, expVal);
        }

        // Normalize
        for (int c = 0; c < numClasses; c++)
        {
            result[c] = _numOps.Divide(result[c], expSum);
        }

        return result;
    }
}
