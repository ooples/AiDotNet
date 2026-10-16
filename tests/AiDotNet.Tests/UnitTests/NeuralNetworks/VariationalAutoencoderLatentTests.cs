using System;
using System.Linq;
using AiDotNet.Enums;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks;

/// <summary>
/// The VAE's latent path (#2326): the encoder's [means | log-variances] go through the reparameterization layer, which
/// samples z = mean + exp(0.5 * logvar) * epsilon with epsilon ~ N(0, 1) in training and returns the mean in inference;
/// the training objective adds beta * KL(q || N(0, I)) on the tape.
/// </summary>
public class VariationalAutoencoderLatentTests
{
    [Fact]
    public void ReparameterizationLayer_SamplesTheEncodedGaussianInTraining_AndReturnsTheMeanInInference()
    {
        const int rows = 20000;
        const double mean = 0.5, std = 2.0;
        var input = new Tensor<double>(new[] { rows, 2 });
        for (int r = 0; r < rows; r++)
        {
            input[r, 0] = mean;
            input[r, 1] = Math.Log(std * std);
        }
        var layer = new RepParameterizationLayer<double>();

        layer.SetTrainingMode(true);
        var z = layer.Forward(input).ToArray();
        double sampleMean = z.Average();
        double sampleStd = Math.Sqrt(z.Select(v => (v - sampleMean) * (v - sampleMean)).Sum() / (rows - 1));
        // epsilon ~ N(0, 1): 20000 draws put the mean within ~0.05 and the spread within ~0.05 of the target. A
        // uniform [0, 1) epsilon (the old sampler) gives mean 1.5 and spread 0.58.
        Assert.InRange(sampleMean, mean - 0.1, mean + 0.1);
        Assert.InRange(sampleStd, std - 0.1, std + 0.1);
        Assert.Contains(z, v => v < mean - 3);   // a Gaussian's tails, not a bounded sampler
        Assert.Contains(z, v => v > mean + 3);

        layer.SetTrainingMode(false);
        Assert.All(layer.Forward(input).ToArray(), v => Assert.Equal(mean, v));
    }

    private sealed class ProbedVae : VariationalAutoencoder<double>
    {
        public ProbedVae(NeuralNetworkArchitecture<double> architecture, int latent) : base(architecture, latent) { }

        public Tensor<double>? TakeAuxiliaryLoss() => ConsumeAuxiliaryTapeLoss();
    }

    [Fact]
    public void TrainingForward_RecordsBetaTimesTheKlDivergence_AndPredictIsDeterministic()
    {
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional, taskType: NeuralNetworkTaskType.Regression,
            inputSize: 12, outputSize: 12);
        var vae = new ProbedVae(architecture, latent: 3);
        var rng = new Random(2326);
        var batch = new Tensor<double>(new[] { 5, 12 });
        for (int i = 0; i < batch.Length; i++) batch.AsWritableSpan()[i] = rng.NextDouble();

        vae.SetTrainingMode(true);
        vae.ForwardForTraining(batch);
        var latentLayer = vae.Layers.OfType<RepParameterizationLayer<double>>().Single();
        var mu = latentLayer.LastMean!.ToArray();
        var logVar = latentLayer.LastLogVariance!.ToArray();
        Assert.Equal(5 * 3, mu.Length);
        double kl = 0;
        for (int i = 0; i < mu.Length; i++) kl += 0.5 * (Math.Exp(logVar[i]) + mu[i] * mu[i] - 1 - logVar[i]);
        double expected = vae.AuxiliaryLossWeight * kl / 5;   // beta * KL summed over latents, averaged over the batch

        var term = vae.TakeAuxiliaryLoss();
        Assert.NotNull(term);
        Assert.Equal(1, term!.Length);
        Assert.Equal(expected, term.ToArray()[0], 10);
        Assert.Null(vae.TakeAuxiliaryLoss());   // consumed once, so a later step never re-adds it
        vae.SetTrainingMode(false);

        var first = vae.Predict(batch).ToArray();
        var second = vae.Predict(batch).ToArray();
        Assert.Equal(first, second);   // inference decodes the latent mean: no sampling
    }

    [Fact]
    public void CustomLayers_NeedExactlyOneReparameterizationLayerBetweenEncoderAndDecoder()
    {
        var withoutLatent = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional, taskType: NeuralNetworkTaskType.Regression,
            inputSize: 8, outputSize: 8,
            layers: new() { new DenseLayer<double>(4), new DenseLayer<double>(8) });
        Assert.Throws<InvalidOperationException>(() => new VariationalAutoencoder<double>(withoutLatent, 2));

        var twoLatents = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional, taskType: NeuralNetworkTaskType.Regression,
            inputSize: 8, outputSize: 8,
            layers: new()
            {
                new DenseLayer<double>(4), new RepParameterizationLayer<double>(), new DenseLayer<double>(4),
                new RepParameterizationLayer<double>(), new DenseLayer<double>(8),
            });
        Assert.Throws<InvalidOperationException>(() => new VariationalAutoencoder<double>(twoLatents, 2));

        var valid = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional, taskType: NeuralNetworkTaskType.Regression,
            inputSize: 8, outputSize: 8,
            layers: new() { new DenseLayer<double>(4), new RepParameterizationLayer<double>(), new DenseLayer<double>(8) });
        var vae = new VariationalAutoencoder<double>(valid, 2);
        var (mean, logVariance) = vae.Encode(new Vector<double>(new double[8]));
        Assert.Equal(2, mean.Length);
        Assert.Equal(2, logVariance.Length);
        Assert.Equal(8, vae.Decode(mean).Length);
    }
}
