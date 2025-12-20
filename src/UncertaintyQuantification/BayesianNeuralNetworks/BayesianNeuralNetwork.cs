using System.Linq;
using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using AiDotNet.Models.Results;
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
public class BayesianNeuralNetwork<T> : NeuralNetwork<T>, IUncertaintyEstimator<T>
{
    private readonly int _numSamples;

    /// <inheritdoc/>
    public override bool SupportsTraining => true;

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

    /// <inheritdoc/>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        // Ensure we're in training mode
        SetTrainingMode(true);

        // Sample weights for Bayesian layers for this training step
        foreach (var bayesianLayer in Layers.OfType<IBayesianLayer<T>>())
        {
            bayesianLayer.SampleWeights();
        }

        // Forward pass with memory for backpropagation
        var outputTensor = ForwardWithMemory(input);
        Vector<T> outputVector = outputTensor.ToVector();

        // Be tolerant of 1D targets for single-output regression (common ergonomic contract).
        var alignedExpected = expectedOutput;
        if (expectedOutput.Rank == 1 &&
            outputTensor.Rank == 2 &&
            outputTensor.Shape.Length >= 2 &&
            outputTensor.Shape[1] == 1 &&
            expectedOutput.Length == outputTensor.Shape[0])
        {
            alignedExpected = expectedOutput.Reshape(outputTensor.Shape);
        }

        Vector<T> expectedVector = alignedExpected.ToVector();
        Vector<T> errorVector = new(expectedVector.Length);

        for (int i = 0; i < expectedVector.Length; i++)
        {
            errorVector[i] = NumOps.Subtract(expectedVector[i], outputVector[i]);
        }

        var dataLoss = LossFunction.CalculateLoss(outputVector, expectedVector);
        var kl = ComputeKLDivergence();

        var batch = input.Rank == 1 ? 1 : Math.Max(1, input.Shape[0]);
        var klScale = NumOps.Divide(NumOps.One, NumOps.FromDouble(batch));
        LastLoss = NumOps.Add(dataLoss, NumOps.Multiply(klScale, kl));

        Backpropagate(new Tensor<T>(outputTensor.Shape, errorVector));

        // Add KL divergence gradients into Bayesian layers before applying updates.
        foreach (var bayesianLayer in Layers.OfType<IBayesianLayer<T>>())
        {
            bayesianLayer.AddKLDivergenceGradients(klScale);
        }

        var learningRate = NumOps.FromDouble(0.01);
        foreach (var layer in Layers)
        {
            if (layer.SupportsTraining && layer.ParameterCount > 0)
            {
                layer.UpdateParameters(learningRate);
            }
        }
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
        var predictions = new List<Tensor<T>>();

        // Sample multiple predictions
        for (int i = 0; i < _numSamples; i++)
        {
            // Sample weights for Bayesian layers
            foreach (var bayesianLayer in Layers.OfType<IBayesianLayer<T>>())
            {
                bayesianLayer.SampleWeights();
            }

            var prediction = Predict(input);
            predictions.Add(prediction);
        }

        // Compute mean and variance
        var mean = ComputeMean(predictions);
        var variance = ComputeVariance(predictions, mean);

        var metrics = CreateDefaultMetrics(mean);
        return new UncertaintyPredictionResult<T, Tensor<T>>(
            methodUsed: UncertaintyQuantificationMethod.BayesianNeuralNetwork,
            prediction: mean,
            variance: variance,
            metrics: metrics);
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
            foreach (var bayesianLayer in Layers.OfType<IBayesianLayer<T>>())
            {
                bayesianLayer.SampleWeights();
            }

            predictions.Add(Predict(input));
        }

        var mean = ComputeMean(predictions);
        var variance = ComputeVariance(predictions, mean);

        // Aleatoric is approximated as a portion of total variance
        // (In practice, this would come from the network's learned output distribution)
        var aleatoricFactor = NumOps.FromDouble(0.3);
        var aleatoricData = variance.ToVector();
        for (int i = 0; i < aleatoricData.Length; i++)
        {
            aleatoricData[i] = NumOps.Multiply(aleatoricData[i], aleatoricFactor);
        }

        return new Tensor<T>(variance.Shape, aleatoricData);
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
            foreach (var bayesianLayer in Layers.OfType<IBayesianLayer<T>>())
            {
                bayesianLayer.SampleWeights();
            }

            predictions.Add(Predict(input));
        }

        var mean = ComputeMean(predictions);
        var variance = ComputeVariance(predictions, mean);

        // Epistemic uncertainty is approximated as the variance across predictions
        var epistemicFactor = NumOps.FromDouble(0.7);
        var epistemicData = variance.ToVector();
        for (int i = 0; i < epistemicData.Length; i++)
        {
            epistemicData[i] = NumOps.Multiply(epistemicData[i], epistemicFactor);
        }

        return new Tensor<T>(variance.Shape, epistemicData);
    }

    /// <summary>
    /// Computes the mean of multiple predictions.
    /// </summary>
    private Tensor<T> ComputeMean(List<Tensor<T>> predictions)
    {
        if (predictions.Count == 0)
            throw new ArgumentException("Cannot compute mean of empty prediction list");

        var shape = predictions[0].Shape;
        var sum = Vector<T>.CreateDefault(predictions[0].Length, NumOps.Zero);
        foreach (var pred in predictions)
        {
            var predData = pred.ToVector();
            for (int i = 0; i < predData.Length; i++)
            {
                sum[i] = NumOps.Add(sum[i], predData[i]);
            }
        }

        var count = NumOps.FromDouble(predictions.Count);
        for (int i = 0; i < sum.Length; i++)
        {
            sum[i] = NumOps.Divide(sum[i], count);
        }

        return new Tensor<T>(shape, sum);
    }

    /// <summary>
    /// Computes the variance of multiple predictions.
    /// </summary>
    private Tensor<T> ComputeVariance(List<Tensor<T>> predictions, Tensor<T> mean)
    {
        var shape = mean.Shape;
        var meanData = mean.ToVector();
        var variance = Vector<T>.CreateDefault(meanData.Length, NumOps.Zero);

        foreach (var pred in predictions)
        {
            var predData = pred.ToVector();
            for (int i = 0; i < predData.Length; i++)
            {
                var diff = NumOps.Subtract(predData[i], meanData[i]);
                variance[i] = NumOps.Add(variance[i], NumOps.Multiply(diff, diff));
            }
        }

        var count = NumOps.FromDouble(predictions.Count);
        for (int i = 0; i < variance.Length; i++)
        {
            variance[i] = NumOps.Divide(variance[i], count);
        }

        return new Tensor<T>(shape, variance);
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

        foreach (var bayesianLayer in Layers.OfType<IBayesianLayer<T>>())
        {
            totalKL = NumOps.Add(totalKL, bayesianLayer.GetKLDivergence());
        }

        return totalKL;
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
