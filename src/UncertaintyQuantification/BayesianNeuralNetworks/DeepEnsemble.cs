using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Results;
using AiDotNet.UncertaintyQuantification.Interfaces;

namespace AiDotNet.UncertaintyQuantification.BayesianNeuralNetworks;

/// <summary>
/// Implements Deep Ensembles for uncertainty estimation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Deep Ensembles is one of the most effective methods for uncertainty estimation.
///
/// The concept is simple: Train multiple independent neural networks (with different random initializations)
/// on the same task, then use them all to make predictions. The diversity in their predictions gives you
/// uncertainty estimates.
///
/// Think of it like a panel of doctors giving diagnoses:
/// - If all doctors agree on the diagnosis, confidence is high
/// - If doctors give different diagnoses, uncertainty is high
///
/// Advantages:
/// - Very reliable uncertainty estimates
/// - No special training procedures needed
/// - Each network can use standard architectures
///
/// Disadvantages:
/// - Requires training and storing multiple networks
/// - Slower inference (must run all networks)
/// - Higher memory usage
///
/// Research shows that ensembles of just 5 networks often outperform more complex Bayesian methods.
/// </para>
/// </remarks>
public class DeepEnsemble<T> : IUncertaintyEstimator<T>
{
    private readonly List<INeuralNetwork<T>> _models;
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Gets the number of models in the ensemble.
    /// </summary>
    public int EnsembleSize => _models.Count;

    /// <summary>
    /// Initializes a new instance of the DeepEnsemble class.
    /// </summary>
    /// <param name="models">The collection of trained models in the ensemble.</param>
    /// <remarks>
    /// <b>For Beginners:</b> Each model should be trained independently with different random
    /// initializations. Typically 5-10 models is a good balance between performance and cost.
    /// </remarks>
    /// <exception cref="ArgumentException">Thrown when the model list is empty.</exception>
    public DeepEnsemble(List<INeuralNetwork<T>> models)
    {
        if (models == null || models.Count == 0)
            throw new ArgumentException("Ensemble must contain at least one model", nameof(models));

        _models = models;
        _numOps = MathHelper.GetNumericOperations<T>();
    }

    /// <summary>
    /// Predicts output with uncertainty estimates from the ensemble.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <returns>A tuple containing the mean prediction and uncertainty.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This runs the input through all models in the ensemble,
    /// then returns the average prediction and how much the models disagreed.
    /// More disagreement = higher uncertainty.
    /// </remarks>
    public UncertaintyPredictionResult<T, Tensor<T>> PredictWithUncertainty(Tensor<T> input)
    {
        var predictions = new List<Tensor<T>>();

        // Get predictions from all models
        foreach (var model in _models)
        {
            predictions.Add(model.Predict(input));
        }

        // Compute mean and variance
        var mean = ComputeMean(predictions);
        var variance = ComputeVariance(predictions, mean);

        var metrics = CreateDefaultMetrics(mean);
        return new UncertaintyPredictionResult<T, Tensor<T>>(
            methodUsed: UncertaintyQuantificationMethod.DeepEnsemble,
            prediction: mean,
            variance: variance,
            metrics: metrics);
    }

    /// <summary>
    /// Estimates aleatoric uncertainty from the ensemble.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <returns>The aleatoric uncertainty estimate.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> If each model in the ensemble outputs both a prediction and
    /// its own uncertainty estimate (e.g., Gaussian likelihood), the average of individual
    /// model uncertainties represents aleatoric uncertainty (data noise).
    ///
    /// <b>Important:</b> This implementation does not currently have per-model variance heads, so a true
    /// aleatoric/epistemic decomposition is not available. As a conservative default, this method returns
    /// zeros; to estimate aleatoric uncertainty, modify ensemble members to output explicit variance (e.g.,
    /// a Gaussian likelihood head).
    /// </remarks>
    public Tensor<T> EstimateAleatoricUncertainty(Tensor<T> input)
    {
        var totalUncertainty = PredictWithUncertainty(input).Variance ?? new Tensor<T>(input.Shape);
        var aleatoric = new Tensor<T>(totalUncertainty.Shape);
        for (int i = 0; i < aleatoric.Length; i++)
        {
            aleatoric[i] = _numOps.Zero;
        }

        return aleatoric;
    }

    /// <summary>
    /// Estimates epistemic uncertainty from the ensemble.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <returns>The epistemic uncertainty estimate.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> The disagreement between ensemble members represents
    /// epistemic uncertainty (model uncertainty). When models disagree, it means they're
    /// uncertain about the correct answer.
    /// </remarks>
    public Tensor<T> EstimateEpistemicUncertainty(Tensor<T> input)
    {
        var totalUncertainty = PredictWithUncertainty(input).Variance ?? new Tensor<T>(input.Shape);
        return totalUncertainty;
    }

    /// <summary>
    /// Gets the predictions from all ensemble members.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <returns>A list of predictions from each ensemble member.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This is useful when you want to analyze individual model
    /// predictions or implement custom uncertainty aggregation methods.
    /// </remarks>
    public List<Tensor<T>> GetAllPredictions(Tensor<T> input)
    {
        var predictions = new List<Tensor<T>>();
        foreach (var model in _models)
        {
            predictions.Add(model.Predict(input));
        }
        return predictions;
    }

    /// <summary>
    /// Computes the mean of multiple predictions.
    /// </summary>
    private Tensor<T> ComputeMean(List<Tensor<T>> predictions)
    {
        if (predictions.Count == 0)
            throw new ArgumentException("Cannot compute mean of empty prediction list");

        var sum = new Tensor<T>(predictions[0].Shape);
        foreach (var pred in predictions)
        {
            for (int i = 0; i < pred.Length; i++)
            {
                sum[i] = _numOps.Add(sum[i], pred[i]);
            }
        }

        var count = _numOps.FromDouble(predictions.Count);
        for (int i = 0; i < sum.Length; i++)
        {
            sum[i] = _numOps.Divide(sum[i], count);
        }

        return sum;
    }

    /// <summary>
    /// Computes the variance of multiple predictions.
    /// </summary>
    private Tensor<T> ComputeVariance(List<Tensor<T>> predictions, Tensor<T> mean)
    {
        var variance = new Tensor<T>(mean.Shape);

        foreach (var pred in predictions)
        {
            for (int i = 0; i < pred.Length; i++)
            {
                var diff = _numOps.Subtract(pred[i], mean[i]);
                variance[i] = _numOps.Add(variance[i], _numOps.Multiply(diff, diff));
            }
        }

        var count = _numOps.FromDouble(predictions.Count);
        for (int i = 0; i < variance.Length; i++)
        {
            variance[i] = _numOps.Divide(variance[i], count);
        }

        return variance;
    }

    private static IReadOnlyDictionary<string, Tensor<T>> CreateDefaultMetrics(Tensor<T> prediction)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var batch = prediction.Rank == 1 ? 1 : Math.Max(1, prediction.Length / Math.Max(1, prediction.Shape[prediction.Shape.Length - 1]));
        var data = new Vector<T>(batch);
        for (int i = 0; i < batch; i++)
        {
            data[i] = numOps.Zero;
        }

        var zeros = new Tensor<T>([batch], data);
        return new Dictionary<string, Tensor<T>>
        {
            ["predictive_entropy"] = zeros,
            ["mutual_information"] = zeros
        };
    }
}
