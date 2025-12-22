using System.Linq;
using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Results;
using AiDotNet.NeuralNetworks;
using AiDotNet.UncertaintyQuantification.Interfaces;
using AiDotNet.UncertaintyQuantification.Layers;

namespace AiDotNet.UncertaintyQuantification.BayesianNeuralNetworks;

/// <summary>
/// Implements Monte Carlo Dropout for uncertainty estimation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> MC Dropout is the simplest way to add uncertainty estimation to existing neural networks.
///
/// The idea is straightforward:
/// 1. Add dropout layers to your network
/// 2. Keep dropout active even during prediction (normally it's turned off)
/// 3. Run multiple predictions with different random dropout patterns
/// 4. The variation in predictions tells you the uncertainty
///
/// This is much easier than full Bayesian neural networks but still provides useful uncertainty estimates.
/// It's like getting a second (and third, and fourth...) opinion from slightly different versions of your model.
/// </para>
/// </remarks>
public class MCDropoutNeuralNetwork<T> : NeuralNetwork<T>, IUncertaintyEstimator<T>
{
    private readonly int _numSamples;

    /// <summary>
    /// Initializes a new instance of the MCDropoutNeuralNetwork class.
    /// </summary>
    /// <param name="architecture">The network architecture (should include MC dropout layers).</param>
    /// <param name="numSamples">Number of forward passes for uncertainty estimation (default: 50).</param>
    /// <remarks>
    /// <b>For Beginners:</b> Make sure your architecture includes MCDropoutLayer instances.
    /// The more samples you use, the better the uncertainty estimate, but prediction becomes slower.
    /// 50 samples is a good default that balances accuracy and speed.
    /// </remarks>
    public MCDropoutNeuralNetwork(NeuralNetworkArchitecture<T> architecture, int numSamples = 50)
        : base(architecture)
    {
        if (numSamples < 1)
            throw new ArgumentException("Number of samples must be at least 1", nameof(numSamples));

        _numSamples = numSamples;
    }

    /// <summary>
    /// Predicts output with uncertainty estimates using MC dropout.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <returns>A prediction result augmented with uncertainty information.</returns>
    public UncertaintyPredictionResult<T, Tensor<T>> PredictWithUncertainty(Tensor<T> input)
    {
        var predictions = new List<Tensor<T>>();

        // Enable Monte Carlo mode for all MC dropout layers
        EnableMCMode(true);

        try
        {
            // Sample multiple predictions with different dropout patterns
            for (int i = 0; i < _numSamples; i++)
            {
                var prediction = Predict(input);
                predictions.Add(prediction);
            }
        }
        finally
        {
            // Restore normal mode
            EnableMCMode(false);
        }

        // Compute mean and variance
        var mean = ComputeMean(predictions);
        var variance = ComputeVariance(predictions, mean);

        var metrics = CreateDefaultMetrics(mean);
        return new UncertaintyPredictionResult<T, Tensor<T>>(
            methodUsed: UncertaintyQuantificationMethod.MonteCarloDropout,
            prediction: mean,
            variance: variance,
            metrics: metrics);
    }

    /// <summary>
    /// Estimates aleatoric uncertainty.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <returns>The aleatoric uncertainty estimate.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> MC Dropout primarily captures epistemic uncertainty.
    /// Aleatoric uncertainty would require the network to explicitly model output variance,
    /// so this is a simplified estimate.
    /// </remarks>
    public Tensor<T> EstimateAleatoricUncertainty(Tensor<T> input)
    {
        // MC Dropout primarily captures epistemic uncertainty
        // Return a small baseline aleatoric estimate
        var totalUncertainty = PredictWithUncertainty(input).Variance ?? new Tensor<T>(input.Shape);

        var aleatoric = new Tensor<T>(totalUncertainty.Shape);
        var aleatoricFactor = NumOps.FromDouble(0.2); // Assume 20% of variance is aleatoric

        for (int i = 0; i < totalUncertainty.Length; i++)
        {
            aleatoric[i] = NumOps.Multiply(totalUncertainty[i], aleatoricFactor);
        }

        return aleatoric;
    }

    /// <summary>
    /// Estimates epistemic uncertainty.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <returns>The epistemic uncertainty estimate.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> MC Dropout excels at capturing epistemic uncertainty,
    /// which represents the model's lack of knowledge about the correct prediction.
    /// </remarks>
    public Tensor<T> EstimateEpistemicUncertainty(Tensor<T> input)
    {
        var totalUncertainty = PredictWithUncertainty(input).Variance ?? new Tensor<T>(input.Shape);

        var epistemic = new Tensor<T>(totalUncertainty.Shape);
        var epistemicFactor = NumOps.FromDouble(0.8); // Assume 80% of variance is epistemic

        for (int i = 0; i < totalUncertainty.Length; i++)
        {
            epistemic[i] = NumOps.Multiply(totalUncertainty[i], epistemicFactor);
        }

        return epistemic;
    }

    /// <summary>
    /// Enables or disables Monte Carlo mode for all MC dropout layers.
    /// </summary>
    private void EnableMCMode(bool enable)
    {
        foreach (var mcDropoutLayer in Layers.OfType<MCDropoutLayer<T>>())
        {
            mcDropoutLayer.MonteCarloMode = enable;
        }
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
                sum[i] = NumOps.Add(sum[i], pred[i]);
            }
        }

        var count = NumOps.FromDouble(predictions.Count);
        for (int i = 0; i < sum.Length; i++)
        {
            sum[i] = NumOps.Divide(sum[i], count);
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
                var diff = NumOps.Subtract(pred[i], mean[i]);
                variance[i] = NumOps.Add(variance[i], NumOps.Multiply(diff, diff));
            }
        }

        var count = NumOps.FromDouble(predictions.Count);
        for (int i = 0; i < variance.Length; i++)
        {
            variance[i] = NumOps.Divide(variance[i], count);
        }

        return variance;
    }

    private IReadOnlyDictionary<string, Tensor<T>> CreateDefaultMetrics(Tensor<T> prediction)
    {
        var batch = prediction.Rank == 1 ? 1 : Math.Max(1, prediction.Length / Math.Max(1, prediction.Shape[prediction.Shape.Length - 1]));
        var data = new Vector<T>(batch);
        for (int i = 0; i < batch; i++)
        {
            data[i] = NumOps.Zero;
        }

        var zeros = new Tensor<T>([batch], data);
        return new Dictionary<string, Tensor<T>>
        {
            ["predictive_entropy"] = zeros,
            ["mutual_information"] = zeros
        };
    }
}
