using System.Linq;
using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Results;
using AiDotNet.NeuralNetworks;
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
/// <example>
/// <code>
/// var inputTensor = Tensor&lt;float&gt;.CreateRandom(2, 4);
/// // Create a Bayesian Neural Network with uncertainty estimation
/// var architecture = new NeuralNetworkArchitecture&lt;float&gt;(inputFeatures: 10, outputSize: 1);
/// var bnn = new BayesianNeuralNetwork&lt;float&gt;(architecture, options: new BayesianNeuralNetworkOptions { NumSamples = 30 });
/// // Get prediction with uncertainty via multiple weight-sampled forward passes
/// var result = bnn.PredictWithUncertainty(inputTensor);
/// // result.Mean, result.Variance provide prediction and uncertainty
/// </code>
/// </example>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Bayesian)]
[ModelTask(ModelTask.Regression)]
[ModelTask(ModelTask.Classification)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
    [ResearchPaper("Weight Uncertainty in Neural Networks", "https://arxiv.org/abs/1505.05424")]
public partial class BayesianNeuralNetwork<T> : NeuralNetwork<T>, IUncertaintyEstimator<T>
{
    private readonly int _numSamples;

    /// <summary>The options this network was built with; a clone rebuilds from them.</summary>
    private readonly BayesianNeuralNetworkOptions _options;

    /// <inheritdoc/>
    public override bool SupportsTraining => true;

    /// <summary>
    /// Initializes a new instance of the BayesianNeuralNetwork class.
    /// </summary>
    /// <param name="architecture">The network architecture.</param>
    /// <remarks>
    /// <b>For Beginners:</b> The number of samples determines how many times we run the network
    /// with different weight samples to estimate uncertainty. More samples = better uncertainty
    /// estimates but slower inference. 30 is usually a good balance.
    /// </remarks>
    public BayesianNeuralNetwork(NeuralNetworkArchitecture<T> architecture,
        BayesianNeuralNetworkOptions? options = null)
        : base(architecture)
    {
        options ??= new BayesianNeuralNetworkOptions();
        if (options.NumSamples < 1)
            throw new ArgumentException("Number of samples must be at least 1", nameof(options.NumSamples));

        _options = options;
        Options = _options;
        _numSamples = options.NumSamples;
    }

    /// <inheritdoc/>
    protected override void InitializeLayers()
    {
        if (Architecture.Layers != null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            ValidateCustomLayers(Layers);
            return;
        }

        Layers.AddRange(LayerHelper<T>.CreateDefaultBayesianNeuralNetworkLayers(Architecture));
    }

    /// <summary>
    /// Predicts output with uncertainty estimates.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <returns>A prediction result augmented with uncertainty information.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This method runs the network multiple times with different
    /// sampled weights and returns both the average prediction and how much the predictions varied.
    /// </remarks>
    public UncertaintyPredictionResult<T, Tensor<T>> PredictWithUncertainty(Tensor<T> input)
    {
        var predictions = SamplePredictions(input);

        // Compute mean and variance
        var mean = ComputeMean(predictions);
        var variance = ComputeVariance(predictions, mean);

        var metrics = PredictiveSampleStatistics<T>.Metrics(predictions, mean);
        return new UncertaintyPredictionResult<T, Tensor<T>>(
            methodUsed: UncertaintyQuantificationMethod.BayesianNeuralNetwork,
            prediction: mean,
            variance: variance,
            metrics: metrics);
    }

    /// <summary>
    /// Draws one prediction per weight sample (Blundell et al. 2015, Sec. 3.4).
    /// </summary>
    private List<Tensor<T>> SamplePredictions(Tensor<T> input)
    {
        var predictions = new List<Tensor<T>>();
        for (int i = 0; i < _numSamples; i++)
        {
            // Sample weights for Bayesian layers
            foreach (var bayesianLayer in Layers.OfType<IBayesianLayer<T>>())
            {
                bayesianLayer.SampleWeights();
            }

            predictions.Add(Predict(input));
        }

        return predictions;
    }

    /// <summary>
    /// Estimates aleatoric (data) uncertainty.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <returns>The aleatoric uncertainty estimate.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> Aleatoric uncertainty represents irreducible randomness in the data itself.
    /// For example, if you're predicting dice rolls, there's inherent randomness that can't be eliminated.
    /// For class-probability outputs this returns the expected Bernoulli variance E_s[p_s (1 - p_s)] over the
    /// weight samples (Kwon et al. 2020); for other outputs it is zero, since the network predicts no noise
    /// variance (Kendall and Gal 2017).
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
        var predictions = SamplePredictions(input);
        var mean = ComputeMean(predictions);
        // Variance of the sampled predictions around their mean (Kendall and Gal 2017; Kwon et al. 2020).
        return ComputeVariance(predictions, mean);
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
        var layerKL = Layers.OfType<IBayesianLayer<T>>().Select(layer => layer.GetKLDivergence()).ToArray();
        if (layerKL.Length == 0)
            return NumOps.Zero;

        return Engine.TensorSum(new Tensor<T>([layerKL.Length], new Vector<T>(layerKL)));
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
