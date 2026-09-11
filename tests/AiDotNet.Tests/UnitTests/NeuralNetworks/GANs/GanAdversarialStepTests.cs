using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.NeuralNetworks.GANs;

/// <summary>
/// One adversarial step must train every network it claims to train (#2155).
/// </summary>
/// <remarks>
/// <para>
/// Several GANs computed their losses as detached scalars and then stepped their optimizers on gradients
/// no backward had produced -- the Backward calls had been commented out when manual backprop was
/// removed -- or scored the generator's output with <c>Predict</c>, whose NoGradScope detaches the
/// adversarial gradient. The losses still came back and looked like training, and a model-level
/// "parameters changed" check passes when any one network moves. So these tests hold each sub-network
/// separately to the claim that it learned.
/// </para>
/// <para>
/// Some checks are sharper than "it moved". Pix2Pix trains its generator with the L1 weight set to zero,
/// so only the adversarial gradient -- which has to cross the conditional discriminator's input
/// concatenation -- can move it. StyleGAN's mapping network sits behind the synthesis network, so it
/// moves only if the gradient crosses the style reshape and mixing between them. InfoGAN's Q network is
/// trained only through the generator's loss.
/// </para>
/// </remarks>
public class GanAdversarialStepTests
{
    private static NeuralNetworkArchitecture<double> Dense(
        NeuralNetworkTaskType task, int input, int hidden, int output, IActivationFunction<double> head)
        => new(
            inputType: InputType.OneDimensional,
            taskType: task,
            inputSize: input,
            outputSize: output,
            layers: new List<ILayer<double>>
            {
                new DenseLayer<double>(hidden, (IActivationFunction<double>)new LeakyReLUActivation<double>()),
                new DenseLayer<double>(output, head),
            });

    private static Tensor<double> Random(int rows, int columns, int seed)
    {
        var rng = RandomHelper.CreateSeededRandom(seed);
        var tensor = new Tensor<double>(new[] { rows, columns });
        for (int i = 0; i < tensor.Length; i++) tensor[i] = rng.NextDouble() * 2.0 - 1.0;
        return tensor;
    }

    private static Tensor<double> OneHot(int rows, int classes, int hot)
    {
        var tensor = new Tensor<double>(new[] { rows, classes });
        for (int r = 0; r < rows; r++) tensor[r, hot % classes] = 1.0;
        return tensor;
    }

    private static Vector<double> Snapshot(INeuralNetworkModel<double> network)
    {
        if (network is NeuralNetworkBase<double> concrete) concrete.MaterializeParameters();
        var parameters = network.GetParameters();
        var copy = new Vector<double>(parameters.Length);
        for (int i = 0; i < parameters.Length; i++) copy[i] = parameters[i];
        return copy;
    }

    private static void AssertTrained(string name, Vector<double> before, INeuralNetworkModel<double> network)
    {
        var after = network.GetParameters();
        Assert.Equal(before.Length, after.Length);

        double change = 0;
        for (int i = 0; i < after.Length; i++)
        {
            Assert.False(double.IsNaN(after[i]) || double.IsInfinity(after[i]),
                $"{name} parameter {i} is {after[i]} after one step.");
            change += Math.Abs(after[i] - before[i]);
        }

        Assert.True(change > 0,
            $"{name} did not change in an adversarial step that claims to train it: its optimizer stepped " +
            "on no gradient.");
    }

    [Fact(Timeout = 120000)]
    public async Task ACGAN_TrainStep_TrainsGeneratorAndDiscriminator()
    {
        await Task.Yield();
        var acgan = new ACGAN<double>(
            Dense(NeuralNetworkTaskType.Generative, 16, 32, 4, new TanhActivation<double>()),
            Dense(NeuralNetworkTaskType.Regression, 4, 32, 4, new IdentityActivation<double>()),
            numClasses: 3,
            inputType: InputType.OneDimensional);
        var generatorBefore = Snapshot(acgan.Generator);
        var discriminatorBefore = Snapshot(acgan.Discriminator);

        acgan.TrainStep(Random(4, 4, 11), acgan.CreateOneHotLabels(4, 1), Random(4, 13, 12), acgan.CreateOneHotLabels(4, 2));

        AssertTrained("ACGAN generator", generatorBefore, acgan.Generator);
        AssertTrained("ACGAN discriminator", discriminatorBefore, acgan.Discriminator);
    }

    [Fact(Timeout = 120000)]
    public async Task ACGAN_Train_AcceptsThePredictInput()
    {
        await Task.Yield();
        var acgan = new ACGAN<double>(
            Dense(NeuralNetworkTaskType.Generative, 16, 32, 4, new TanhActivation<double>()),
            Dense(NeuralNetworkTaskType.Regression, 4, 32, 4, new IdentityActivation<double>()),
            numClasses: 3,
            inputType: InputType.OneDimensional);
        var generatorBefore = Snapshot(acgan.Generator);

        // Train takes the same input Predict does: 13 noise values followed by the class conditioning.
        var input = Random(1, 16, 21);
        var sample = acgan.Predict(input);
        acgan.Train(input, Random(1, 4, 22));

        Assert.Equal(4, sample.Length);
        AssertTrained("ACGAN generator", generatorBefore, acgan.Generator);
    }

    [Fact(Timeout = 120000)]
    public async Task StyleGAN_TrainStep_TrainsMappingSynthesisAndDiscriminator()
    {
        await Task.Yield();
        var stylegan = new StyleGAN<double>(
            Dense(NeuralNetworkTaskType.Regression, 16, 16, 8, new IdentityActivation<double>()),
            Dense(NeuralNetworkTaskType.Generative, 8, 16, 4, new TanhActivation<double>()),
            Dense(NeuralNetworkTaskType.Regression, 4, 16, 1, new IdentityActivation<double>()),
            latentSize: 16,
            intermediateLatentSize: 8,
            inputType: InputType.OneDimensional);
        var mappingBefore = Snapshot(stylegan.MappingNetwork);
        var synthesisBefore = Snapshot(stylegan.SynthesisNetwork);
        var discriminatorBefore = Snapshot(stylegan.Discriminator);

        stylegan.TrainStep(Random(4, 4, 31), Random(4, 16, 32));

        AssertTrained("StyleGAN mapping network", mappingBefore, stylegan.MappingNetwork);
        AssertTrained("StyleGAN synthesis network", synthesisBefore, stylegan.SynthesisNetwork);
        AssertTrained("StyleGAN discriminator", discriminatorBefore, stylegan.Discriminator);
    }

    [Fact(Timeout = 120000)]
    public async Task Pix2Pix_TrainStep_TrainsTheGeneratorThroughTheDiscriminatorAlone()
    {
        await Task.Yield();
        var pix2pix = new Pix2Pix<double>(
            Dense(NeuralNetworkTaskType.Generative, 16, 32, 4, new TanhActivation<double>()),
            Dense(NeuralNetworkTaskType.Regression, 20, 32, 1, new IdentityActivation<double>()),
            InputType.OneDimensional,
            l1Lambda: 0.0);
        var generatorBefore = Snapshot(pix2pix.Generator);
        var discriminatorBefore = Snapshot(pix2pix.Discriminator);

        pix2pix.TrainStep(Random(4, 16, 41), Random(4, 4, 42));

        AssertTrained("Pix2Pix generator (adversarial term only)", generatorBefore, pix2pix.Generator);
        AssertTrained("Pix2Pix discriminator", discriminatorBefore, pix2pix.Discriminator);
    }

    [Fact(Timeout = 120000)]
    public async Task WGAN_TrainStep_TrainsGeneratorThroughTheCritic()
    {
        await Task.Yield();
        var wgan = new WGAN<double>(
            generatorArchitecture: new NeuralNetworkArchitecture<double>(
                inputType: InputType.OneDimensional, taskType: NeuralNetworkTaskType.Regression,
                inputSize: 16, outputSize: 4),
            criticArchitecture: new NeuralNetworkArchitecture<double>(
                inputType: InputType.OneDimensional, taskType: NeuralNetworkTaskType.Regression,
                inputSize: 4, outputSize: 1),
            inputType: InputType.OneDimensional);
        var generatorBefore = Snapshot(wgan.Generator);
        var criticBefore = Snapshot(wgan.Critic);

        wgan.TrainStep(Random(4, 4, 51), Random(4, 16, 52));

        AssertTrained("WGAN generator", generatorBefore, wgan.Generator);
        AssertTrained("WGAN critic", criticBefore, wgan.Critic);
    }

    [Fact(Timeout = 120000)]
    public async Task InfoGAN_TrainStep_TrainsGeneratorDiscriminatorAndQNetwork()
    {
        await Task.Yield();
        var infogan = new InfoGAN<double>(
            new NeuralNetworkArchitecture<double>(
                inputType: InputType.OneDimensional, taskType: NeuralNetworkTaskType.Regression,
                inputSize: 6, outputSize: 4),
            new NeuralNetworkArchitecture<double>(
                inputType: InputType.OneDimensional, taskType: NeuralNetworkTaskType.BinaryClassification,
                inputSize: 4, outputSize: 1),
            new NeuralNetworkArchitecture<double>(
                inputType: InputType.OneDimensional, taskType: NeuralNetworkTaskType.Regression,
                inputSize: 4, outputSize: 2),
            latentCodeSize: 2,
            inputType: InputType.OneDimensional);
        var generatorBefore = Snapshot(infogan.Generator);
        var discriminatorBefore = Snapshot(infogan.Discriminator);
        var qBefore = Snapshot(infogan.QNetwork);

        infogan.TrainStep(Random(4, 4, 61), Random(4, 4, 62), Random(4, 2, 63));

        AssertTrained("InfoGAN generator", generatorBefore, infogan.Generator);
        AssertTrained("InfoGAN discriminator", discriminatorBefore, infogan.Discriminator);
        AssertTrained("InfoGAN Q network", qBefore, infogan.QNetwork);
    }

    [Fact(Timeout = 120000)]
    public async Task ConditionalGAN_TrainStep_TrainsGeneratorThroughTheDiscriminator()
    {
        await Task.Yield();
        // Generator: 22 noise values followed by 10 one-hot classes, 32 -> 64. The discriminator is declared
        // on the 64-wide sample; ConditionalGAN widens its input by the 10 condition values itself.
        using var cgan = new ConditionalGAN<double>(
            new NeuralNetworkArchitecture<double>(
                InputType.OneDimensional, NeuralNetworkTaskType.Generative, NetworkComplexity.Simple,
                inputSize: 32, outputSize: 64),
            new NeuralNetworkArchitecture<double>(
                InputType.OneDimensional, NeuralNetworkTaskType.BinaryClassification, NetworkComplexity.Simple,
                inputSize: 64, outputSize: 1),
            numConditionClasses: 10,
            InputType.OneDimensional);
        var generatorBefore = Snapshot(cgan.Generator);
        var discriminatorBefore = Snapshot(cgan.Discriminator);

        cgan.TrainStep(Random(4, 64, 71), OneHot(4, 10, 3), Random(4, 22, 72));

        AssertTrained("ConditionalGAN generator", generatorBefore, cgan.Generator);
        AssertTrained("ConditionalGAN discriminator", discriminatorBefore, cgan.Discriminator);
    }

    [Fact(Timeout = 120000)]
    public async Task CycleGAN_TrainStep_TrainsBothGeneratorsAndBothDiscriminators()
    {
        await Task.Yield();
        var cyclegan = new CycleGAN<double>();
        var generatorAtoBBefore = Snapshot(cyclegan.GeneratorAtoB);
        var generatorBtoABefore = Snapshot(cyclegan.GeneratorBtoA);
        var discriminatorABefore = Snapshot(cyclegan.DiscriminatorA);
        var discriminatorBBefore = Snapshot(cyclegan.DiscriminatorB);

        cyclegan.TrainStep(Random(2, 784, 81), Random(2, 784, 82));

        AssertTrained("CycleGAN generator A->B", generatorAtoBBefore, cyclegan.GeneratorAtoB);
        AssertTrained("CycleGAN generator B->A", generatorBtoABefore, cyclegan.GeneratorBtoA);
        AssertTrained("CycleGAN discriminator A", discriminatorABefore, cyclegan.DiscriminatorA);
        AssertTrained("CycleGAN discriminator B", discriminatorBBefore, cyclegan.DiscriminatorB);
    }
}
