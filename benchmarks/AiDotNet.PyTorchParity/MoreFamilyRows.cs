using AiDotNet.ActivationFunctions;
using AiDotNet.Configuration;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.NeuralNetworks.Options;
using AiDotNet.Tensors.LinearAlgebra;

/// <summary>
/// The "efficientnet-b0" row: the shipped <see cref="EfficientNetNetwork{T}"/> at B0 scaling (width 1.0, depth 1.0:
/// stem 32, MBConv stages 16/24/40/80/112/192/320 with squeeze-excitation, head 1280) on CIFAR [3, 32, 32] images,
/// 10 classes. The network is built through the Custom variant only because the named B0 variant pins its input to
/// 224x224; Custom with multipliers 1.0 is the B0 network. Custom's classifier dropout is 0.1 (B0's is 0.2) and the
/// drop-connect rate is the paper's 0.2. benchmark.py's <c>EfficientNetB0</c> is torchvision's efficientnet_b0 with the
/// same BatchNorm settings (eps 1e-3, TensorFlow momentum 0.99) and the TensorFlow "same" padding AiDotNet's trunk
/// applies before each strided convolution; the trainable parameter counts are equal.
/// </summary>
internal sealed class AiDotNetEfficientNetB0Model(int seed) : AiDotNetBenchmarkModel(seed)
{
    private const int ImageSize = 32;
    private const int Classes = 10;

    protected override int[] InputShapePerSample => new[] { 3, ImageSize, ImageSize };
    protected override int OutputClasses => Classes;
    protected override IReadOnlyList<string> PyTorchModulePrefixes => [];
    public override bool HasStep1Twin => false;

    protected override NeuralNetworkBase<float> BuildNetwork()
    {
        var arch = new NeuralNetworkArchitecture<float>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputHeight: ImageSize, inputWidth: ImageSize, inputDepth: 3,
            outputSize: Classes);
        // The model's own loss: categorical cross-entropy on its softmax head, the same function as PyTorch's
        // CrossEntropyLoss on logits.
        return new EfficientNetNetwork<float>(arch, new EfficientNetConfiguration(EfficientNetVariant.Custom, Classes,
            inputChannels: 3, customInputHeight: ImageSize, customWidthMultiplier: 1.0, customDepthMultiplier: 1.0));
    }
}

/// <summary>
/// The "vae" row: the shipped <see cref="VariationalAutoencoder{T}"/> as Kingma &amp; Welling's MNIST VAE (the
/// pytorch/examples layout): 784 -> 400 (ReLU) -> 2 x 20 (means, log-variances), reparameterized sample,
/// 20 -> 400 (ReLU) -> 784 (sigmoid). The loss is the model's own: mean squared reconstruction error plus its
/// beta = 0.01 KL term, 0.5 * sum(exp(logvar) + mu^2 - 1 - logvar) averaged over the batch. Inference decodes the
/// latent mean. benchmark.py's <c>MnistVAE</c> computes the same function with the same 652,824 parameters (its
/// fc21/fc22 split of the 400 -> 40 layer is the same weights).
/// </summary>
internal sealed class AiDotNetVaeModel(int seed) : AiDotNetBenchmarkModel(seed)
{
    private const int Pixels = 784;
    private const int Hidden = 400;
    private const int Latent = 20;

    protected override int[] InputShapePerSample => new[] { Pixels };
    protected override int OutputClasses => Pixels;
    protected override IReadOnlyList<string> PyTorchModulePrefixes => [];
    public override bool HasStep1Twin => false;

    protected override NeuralNetworkBase<float> BuildNetwork()
    {
        var layers = new List<ILayer<float>>
        {
            new DenseLayer<float>(Hidden, new ReLUActivation<float>() as IActivationFunction<float>),
            new DenseLayer<float>(2 * Latent, NoActivation),
            new RepParameterizationLayer<float>(),
            new DenseLayer<float>(Hidden, new ReLUActivation<float>() as IActivationFunction<float>),
            new DenseLayer<float>(Pixels, new SigmoidActivation<float>() as IActivationFunction<float>),
        };
        var arch = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: Pixels,
            outputSize: Pixels,
            layers: layers);
        return new VariationalAutoencoder<float>(arch, Latent);
    }

    /// <summary>Flattened 28x28 images with pixels uniform in [0, 1]; the target is the input (benchmark.py's vae batch).</summary>
    public override void LoadSyntheticBatch(int batchSize)
    {
        var pixels = new float[batchSize * Pixels];
        for (var i = 0; i < pixels.Length; i++) pixels[i] = (float)Random.NextDouble();
        Input = new Tensor<float>(pixels, [batchSize, Pixels]);
        Label = Input;
    }
}

/// <summary>
/// The "gat" row: the shipped <see cref="GraphAttentionNetwork{T}"/> at Velickovic et al.'s Cora configuration -
/// 2,708 nodes, 1,433 input features, 7 classes, a hidden layer of 8 heads x 8 features (ELU, concatenated) and a
/// one-head output layer, attention dropout 0.6 - trained full-batch for node classification. The graph is a fixed,
/// Cora-sized stand-in (benchmark.py's cora_like_graph): a ring plus a chord i -> i+7 per node, 5,416 undirected edges
/// against Cora's 5,429, with self-loops. The model is the authors' masked attention (an additive -1e9 mask off the
/// graph, softmax over each row). AiDotNet's layer evaluates it over the adjacency's edges only (its sparse CSR kernel:
/// the same coefficients, since every masked entry contributes exp(-1e9 - max) = 0); benchmark.py's <c>CoraGAT</c>
/// computes it densely over the N x N matrix, as the reference implementation does, with the same 92,373 parameters. One "sample" is the whole graph, so the row runs at
/// batch size 1 only (--batch-size 1 --inference-batches 1).
/// </summary>
internal sealed class AiDotNetGatModel(int seed) : AiDotNetBenchmarkModel(seed)
{
    internal const int Nodes = 2708;
    internal const int Features = 1433;
    internal const int Classes = 7;

    protected override int[] InputShapePerSample => new[] { Nodes, Features };
    protected override int OutputClasses => Classes;
    protected override IReadOnlyList<string> PyTorchModulePrefixes => [];
    public override bool HasStep1Twin => false;

    protected override NeuralNetworkBase<float> BuildNetwork()
    {
        var arch = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputSize: Features,
            outputSize: Classes);
        // The options' defaults are the paper's Cora settings; they are spelled out so the row states its shape.
        var network = new GraphAttentionNetwork<float>(arch,
            options: new GraphAttentionNetworkOptions { NumHeads = 8, HiddenSize = 64, NumLayers = 2, DropoutRate = 0.6 });
        network.SetAdjacencyMatrix(CoraLikeGraph());
        return network;
    }

    /// <summary>
    /// One graph of standard-normal node features; each node's class is whichever of its first seven features is
    /// largest, so the labels are learnable and the loss falls (benchmark.py's gat batch).
    /// </summary>
    public override void LoadSyntheticBatch(int batchSize)
    {
        if (batchSize != 1)
            throw new ArgumentException(
                "The gat row trains and infers on the whole Cora-sized graph as one sample; run it with --batch-size 1 " +
                "--inference-batches 1.", nameof(batchSize));
        base.LoadSyntheticBatch(1);
        var x = Input.ToArray();
        var labels = new float[Nodes * Classes];
        for (var n = 0; n < Nodes; n++)
        {
            var best = 0;
            for (var c = 1; c < Classes; c++)
                if (x[n * Features + c] > x[n * Features + best]) best = c;
            labels[n * Classes + best] = 1f;
        }
        Label = new Tensor<float>(labels, [1, Nodes, Classes]);
    }

    /// <summary>A ring over the nodes plus a chord i -> i+7 from every node, undirected, with self-loops (0/1 entries).</summary>
    private static Tensor<float> CoraLikeGraph()
    {
        var a = new float[Nodes * Nodes];
        for (var i = 0; i < Nodes; i++)
        {
            a[i * Nodes + i] = 1f;
            foreach (var j in new[] { (i + 1) % Nodes, (i + 7) % Nodes })
            {
                a[i * Nodes + j] = 1f;
                a[j * Nodes + i] = 1f;
            }
        }
        return new Tensor<float>(a, [Nodes, Nodes]);
    }
}

/// <summary>
/// The "dcgan" row: the shipped <see cref="DCGAN{T}"/> at Radford et al.'s 64x64 configuration (latent 100, 64 base
/// feature maps). The generator is Dense(100 -> 512*4*4) + reshape, then transposed convolutions 4x4 stride 2
/// 512 -> 256 -> 128 -> 64 (BatchNorm + ReLU) -> 3 (tanh); the discriminator is 4x4 stride-2 convolutions 3 -> 64
/// (LeakyReLU 0.2) -> 128 -> 256 -> 512 (BatchNorm + LeakyReLU 0.2) and a 4x4 convolution to one logit. One training
/// step is the model's adversarial iteration: the discriminator on the concatenated [real; generated] batch with
/// BCE-with-logits, then the generator through the frozen (eval-mode) discriminator with the non-saturating loss, each
/// under the paper's Adam (lr 2e-4, betas 0.5 / 0.999). The reported loss is the discriminator's. Inference generates
/// a batch of images. benchmark.py's <c>DCGAN64</c> is a layer-for-layer twin of the networks as AiDotNet builds them
/// (with biases, and with no BatchNorm after the projection; see the report), with the same parameter count.
/// </summary>
internal sealed class AiDotNetDcganModel : IBenchmarkModel
{
    internal const int Latent = 100;
    internal const int Channels = 3;
    internal const int ImageSize = 64;

    private readonly DCGAN<float> _model;
    private readonly Random _random;
    private Tensor<float> _noise = Tensor<float>.Empty();
    private Tensor<float> _real = Tensor<float>.Empty();
    private bool _trained;

    public AiDotNetDcganModel(int seed)
    {
        _random = new Random(seed);
        _model = new DCGAN<float>(Latent, Channels, ImageSize, ImageSize);

        // Materialize both networks' lazy layers before counting: generate, then score the real batch.
        LoadSyntheticBatch(2);
        Forward();
        _model.Discriminator.SetTrainingMode(false);
        _ = _model.Discriminator.Predict(_real);
        ParameterCount = _model.Generator.TrainableParameterCount + _model.Discriminator.TrainableParameterCount;
    }

    public long ParameterCount { get; }

    /// <summary>The input is the latent batch, the label the real images the discriminator sees.</summary>
    public (Tensor<float> Input, Tensor<float> Label) CurrentBatch
    {
        get => (_noise, _real);
        set => (_noise, _real) = value;
    }

    /// <summary>Standard-normal latents and real images uniform in [-1, 1], the tanh range (benchmark.py's dcgan batch).</summary>
    public void LoadSyntheticBatch(int batchSize)
    {
        var z = new float[batchSize * Latent];
        for (var i = 0; i < z.Length; i++)
        {
            double u1 = 1.0 - _random.NextDouble(), u2 = _random.NextDouble();
            z[i] = (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
        }
        _noise = new Tensor<float>(z, [batchSize, Latent]);
        var pixels = new float[batchSize * Channels * ImageSize * ImageSize];
        for (var i = 0; i < pixels.Length; i++) pixels[i] = (float)(_random.NextDouble() * 2.0 - 1.0);
        _real = new Tensor<float>(pixels, [batchSize, Channels, ImageSize, ImageSize]);
    }

    public void Forward() => _ = _model.GenerateImages(_noise);

    public void Backward()
    {
        // Train is the whole adversarial iteration: the discriminator update, then the generator update. The loss
        // stays on the device until LastLoss reads it, as the PyTorch twin returns discriminator_loss.detach();
        // TrainStep would read both losses on the host every step.
        _model.Train(_noise, _real);
        _trained = true;
    }

    public void Step()
    {
        // Both optimizer steps ran inside Backward's TrainStep call.
    }

    public double? LastLoss => _trained ? Convert.ToDouble(_model.Discriminator.GetLastLoss()) : null;
}
