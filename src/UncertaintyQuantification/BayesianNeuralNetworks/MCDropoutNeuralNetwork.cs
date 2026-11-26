using System.Linq;
using AiDotNet.Attributes;
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
/// <example>
/// <code>
/// var inputTensor = Tensor&lt;float&gt;.CreateRandom(2, 4);
/// // Create an MC Dropout network for uncertainty-aware predictions
/// var architecture = new NeuralNetworkArchitecture&lt;float&gt;(inputFeatures: 10, outputSize: 1);
/// // Add MCDropoutLayer instances to the architecture for stochastic inference
/// var mcDropout = new MCDropoutNeuralNetwork&lt;float&gt;(architecture, options: new MCDropoutNeuralNetworkOptions { NumSamples = 50 });
/// var result = mcDropout.PredictWithUncertainty(inputTensor);
/// // result.Mean = average prediction, result.Variance = epistemic uncertainty
/// </code>
/// </example>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Bayesian)]
[ModelTask(ModelTask.Regression)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
    [ResearchPaper("Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning", "https://arxiv.org/abs/1506.02142")]
public class MCDropoutNeuralNetwork<T> : NeuralNetwork<T>, IUncertaintyEstimator<T>
{
    private readonly int _numSamples;

    /// <summary>The options this network was built with; a clone rebuilds from them.</summary>
    private readonly MCDropoutNeuralNetworkOptions _options;

    /// <summary>
    /// Initializes a new instance of the MCDropoutNeuralNetwork class.
    /// </summary>
    /// <param name="architecture">The network architecture (should include MC dropout layers).</param>
    /// <remarks>
    /// <b>For Beginners:</b> Make sure your architecture includes MCDropoutLayer instances.
    /// The more samples you use, the better the uncertainty estimate, but prediction becomes slower.
    /// 50 samples is a good default that balances accuracy and speed.
    /// </remarks>
    public MCDropoutNeuralNetwork(NeuralNetworkArchitecture<T> architecture,
        MCDropoutNeuralNetworkOptions? options = null)
        : base(architecture)
    {
        options ??= new MCDropoutNeuralNetworkOptions();
        if (options.NumSamples < 1)
            throw new ArgumentException("Number of samples must be at least 1", nameof(options.NumSamples));

        _options = options;
        Options = _options;
        _numSamples = options.NumSamples;
    }

    /// <summary>
    /// Predicts output with uncertainty estimates using MC dropout.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <returns>A prediction result augmented with uncertainty information.</returns>
    public UncertaintyPredictionResult<T, Tensor<T>> PredictWithUncertainty(Tensor<T> input)
    {
        var predictions = SamplePredictions(input);

        // Compute mean and variance
        var mean = ComputeMean(predictions);
        var variance = ComputeVariance(predictions, mean);

        var metrics = PredictiveSampleStatistics<T>.Metrics(predictions, mean);
        return new UncertaintyPredictionResult<T, Tensor<T>>(
            methodUsed: UncertaintyQuantificationMethod.MonteCarloDropout,
            prediction: mean,
            variance: variance,
            metrics: metrics);
    }

    /// <summary>
    /// Draws one prediction per dropout mask with Monte Carlo mode on (Gal and Ghahramani 2016).
    /// </summary>
    private List<Tensor<T>> SamplePredictions(Tensor<T> input)
    {
        var predictions = new List<Tensor<T>>();

        // Enable Monte Carlo mode for all MC dropout layers
        EnableMCMode(true);

        try
        {
            // Sample multiple predictions with different dropout patterns
            for (int i = 0; i < _numSamples; i++)
            {
                predictions.Add(Predict(input));
            }
        }
        finally
        {
            // Restore normal mode
            EnableMCMode(false);
        }

        return predictions;
    }

    /// <summary>
    /// Estimates aleatoric uncertainty.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <returns>The aleatoric uncertainty estimate.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> MC Dropout primarily captures epistemic uncertainty.
    /// For class-probability outputs this returns the expected Bernoulli variance E_s[p_s (1 - p_s)] over the
    /// dropout samples (Kwon et al. 2020). For other outputs it is zero: aleatoric uncertainty would require
    /// the network to explicitly model output variance (Kendall and Gal 2017).
    /// A single sigmoid output ([batch, 1]) has the same shape as a regression output and is treated as one
    /// (zero aleatoric part); output the two-class distribution [1 - p, p] to get the Bernoulli terms.
    /// </remarks>
    public Tensor<T> EstimateAleatoricUncertainty(Tensor<T> input)
    {
        var predictions = SamplePredictions(input);
        var mean = ComputeMean(predictions);
        // Expected Bernoulli variance E_s[p_s (1 - p_s)] for class-probability outputs (Kwon et al. 2020);
        // zero for other outputs, since the network predicts no noise variance (Kendall and Gal 2017).
        return PredictiveSampleStatistics<T>.AleatoricVariance(predictions, mean);
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
        var predictions = SamplePredictions(input);
        var mean = ComputeMean(predictions);
        // Variance of the sampled predictions around their mean (Kendall and Gal 2017; Kwon et al. 2020).
        return ComputeVariance(predictions, mean);
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
    private static Tensor<T> ComputeMean(List<Tensor<T>> predictions)
        => PredictiveSampleStatistics<T>.Mean(predictions);

    /// <summary>
    /// Computes the variance of multiple predictions.
    /// </summary>
    private static Tensor<T> ComputeVariance(List<Tensor<T>> predictions, Tensor<T> mean)
        => PredictiveSampleStatistics<T>.Variance(predictions, mean);
}
