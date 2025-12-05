using AiDotNet.UncertaintyQuantification.Interfaces;

namespace AiDotNet.UncertaintyQuantification.BayesianNeuralNetworks;

/// <summary>
/// Implements a Bayesian Neural Network that provides uncertainty estimates with predictions.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> A Bayesian Neural Network (BNN) is a neural network that can tell you
/// not just what it predicts, but also how uncertain it is about that prediction.
///
/// This is incredibly important for safety-critical applications like:
/// - Medical diagnosis: "This might be cancer, but I'm very uncertain - get a second opinion"
/// - Autonomous driving: "I'm not sure what that object is - proceed with caution"
/// - Financial predictions: "The market might go up, but there's high uncertainty"
///
/// The network achieves this by making multiple predictions with slightly different weights
/// (sampled from learned probability distributions) and analyzing how much these predictions vary.
/// </para>
/// </remarks>
public class BayesianNeuralNetwork<T> : NeuralNetworkBase<T>, IUncertaintyEstimator<T>
{
    private readonly int _numSamples;

    /// <summary>
    /// Initializes a new instance of the BayesianNeuralNetwork class.
    /// </summary>
    /// <param name="architecture">The network architecture.</param>
    /// <param name="numSamples">Number of forward passes for uncertainty estimation (default: 30).</param>
    /// <remarks>
    /// <b>For Beginners:</b> The number of samples determines how many times we run the network
    /// with different weight samples to estimate uncertainty. More samples = better uncertainty
    /// estimates but slower inference. 30 is usually a good balance.
    /// </remarks>
    public BayesianNeuralNetwork(NeuralNetworkArchitecture<T> architecture, int numSamples = 30)
        : base(architecture)
    {
        if (numSamples < 1)
            throw new ArgumentException("Number of samples must be at least 1", nameof(numSamples));

        _numSamples = numSamples;
    }

    /// <summary>
    /// Predicts output with uncertainty estimates.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <returns>A tuple containing the mean prediction and total uncertainty.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method runs the network multiple times with different
    /// sampled weights and returns both the average prediction and how much the predictions varied.
    /// </remarks>
    public (Tensor<T> mean, Tensor<T> uncertainty) PredictWithUncertainty(Tensor<T> input)
    {
        var predictions = new List<Tensor<T>>();

        // Sample multiple predictions
        for (int i = 0; i < _numSamples; i++)
        {
            // Sample weights for Bayesian layers
            foreach (var layer in Layers)
            {
                if (layer is IBayesianLayer<T> bayesianLayer)
                {
                    bayesianLayer.SampleWeights();
                }
            }

            var prediction = Predict(input);
            predictions.Add(prediction);
        }

        // Compute mean and variance
        var mean = ComputeMean(predictions);
        var variance = ComputeVariance(predictions, mean);

        return (mean, variance);
    }

    /// <summary>
    /// Estimates aleatoric (data) uncertainty.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <returns>The aleatoric uncertainty estimate.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> Aleatoric uncertainty represents irreducible randomness in the data itself.
    /// For example, if you're predicting dice rolls, there's inherent randomness that can't be eliminated.
    /// </remarks>
    public Tensor<T> EstimateAleatoricUncertainty(Tensor<T> input)
    {
        // For simplicity, we estimate aleatoric uncertainty as the average of individual prediction variances
        var predictions = new List<Tensor<T>>();

        for (int i = 0; i < _numSamples; i++)
        {
            foreach (var layer in Layers)
            {
                if (layer is IBayesianLayer<T> bayesianLayer)
                {
                    bayesianLayer.SampleWeights();
                }
            }

            predictions.Add(Predict(input));
        }

        var mean = ComputeMean(predictions);
        var variance = ComputeVariance(predictions, mean);

        // Aleatoric is approximated as a portion of total variance
        // (In practice, this would come from the network's learned output distribution)
        var aleatoricFactor = NumOps.FromDouble(0.3);
        var aleatoric = new Tensor<T>(variance.Shape);
        for (int i = 0; i < variance.Length; i++)
        {
            aleatoric[i] = NumOps.Multiply(variance[i], aleatoricFactor);
        }

        return aleatoric;
    }

    /// <summary>
    /// Estimates epistemic (model) uncertainty.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <returns>The epistemic uncertainty estimate.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> Epistemic uncertainty represents the model's lack of knowledge.
    /// This type of uncertainty can be reduced by collecting more training data.
    /// It's high when the model encounters inputs unlike anything it was trained on.
    /// </remarks>
    public Tensor<T> EstimateEpistemicUncertainty(Tensor<T> input)
    {
        var predictions = new List<Tensor<T>>();

        for (int i = 0; i < _numSamples; i++)
        {
            foreach (var layer in Layers)
            {
                if (layer is IBayesianLayer<T> bayesianLayer)
                {
                    bayesianLayer.SampleWeights();
                }
            }

            predictions.Add(Predict(input));
        }

        var mean = ComputeMean(predictions);
        var variance = ComputeVariance(predictions, mean);

        // Epistemic uncertainty is approximated as the variance across predictions
        var epistemicFactor = NumOps.FromDouble(0.7);
        var epistemic = new Tensor<T>(variance.Shape);
        for (int i = 0; i < variance.Length; i++)
        {
            epistemic[i] = NumOps.Multiply(variance[i], epistemicFactor);
        }

        return epistemic;
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

    /// <summary>
    /// Computes the total KL divergence from all Bayesian layers.
    /// </summary>
    /// <returns>The sum of KL divergences.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This is used during training to regularize the weight distributions.
    /// It's added to the main loss to prevent the network from becoming overconfident.
    /// </remarks>
    public T ComputeKLDivergence()
    {
        var totalKL = NumOps.Zero;

        foreach (var layer in Layers)
        {
            if (layer is IBayesianLayer<T> bayesianLayer)
            {
                totalKL = NumOps.Add(totalKL, bayesianLayer.GetKLDivergence());
            }
        }

        return totalKL;
    }
}
