using AiDotNet.ActivationFunctions;
using AiDotNet.Audio.Foundations;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.Engines;

namespace AiDotNet.SpeechRecognition.Robust;

/// <summary>
/// Wang et al. 2021's noise-robust wav2vec 2.0 (Figure 1): the wav2vec 2.0 encoder with its CTC head, its contrastive
/// pre-training heads, and a reconstruction module that predicts the clean waveform from the contextual
/// representations of the noisy one.
/// </summary>
internal sealed class NoiseRobustWav2Vec2Network<T>
{
    private static readonly INumericOperations<T> NumOps = MathHelper.GetNumericOperations<T>();
    private readonly IEngine _engine;

    /// <summary>Builds the network.</summary>
    /// <param name="engine">The engine every layer runs on.</param>
    /// <param name="configuration">The wav2vec 2.0 encoder.</param>
    /// <param name="pretraining">The contrastive pre-training heads.</param>
    /// <param name="vocabSize">The CTC vocabulary, blank included.</param>
    /// <param name="reconstructionHidden">LSTM units per direction of the reconstruction bottleneck.</param>
    /// <param name="reconstructionWeight">λ2 of Equation 4.</param>
    /// <param name="seed">The initialization seed.</param>
    public NoiseRobustWav2Vec2Network(IEngine engine, Wav2Vec2Configuration configuration, Wav2Vec2PretrainingConfiguration pretraining,
        int vocabSize, int reconstructionHidden, double reconstructionWeight, int seed)
    {
        _engine = engine;
        ReconstructionWeight = reconstructionWeight;
        Stack = new Wav2Vec2CtcStack<T>(engine, configuration, vocabSize, seed);
        var layers = new List<LayerBase<T>>(Stack.Layers);
        Pretraining = new Wav2Vec2Pretraining<T>(engine, layers, pretraining, configuration.ConvDims[^1], configuration.HiddenSize);
        Pretraining.InitializeLikeReference(Tensors.Helpers.RandomHelper.CreateSeededRandom(seed + 1));

        // Section 2.2: a two-layer BLSTM bottleneck with layer normalization after each layer.
        int width = configuration.HiddenSize;
        for (int i = 0; i < 2; i++)
        {
            var forward = new LSTMLayer<T>(reconstructionHidden);
            var backward = new LSTMLayer<T>(reconstructionHidden);
            var norm = new LayerNormalizationLayer<T>(2 * reconstructionHidden);
            layers.Add(forward); layers.Add(backward); layers.Add(norm);
            Bottleneck.Add((forward, backward, norm));
            width = 2 * reconstructionHidden;
        }

        // A CNN decoder symmetric to the feature encoder: its convolutions transposed in reverse order, GELU between them,
        // ending in one channel of waveform.
        var c = configuration;
        for (int i = c.ConvDims.Length - 1; i >= 0; i--)
        {
            int output = i > 0 ? c.ConvDims[i - 1] : 1;
            var activation = i > 0 ? (IActivationFunction<T>)new GELUActivation<T>() : new IdentityActivation<T>();
            var conv = new Conv1DTransposeLayer<T>(width, output, c.ConvKernels[i], c.ConvStrides[i], padding: 0, activation: activation);
            layers.Add(conv);
            Decoder.Add(conv);
            width = output;
        }
        Layers = layers;
    }

    /// <summary>The wav2vec 2.0 encoder and CTC head.</summary>
    public Wav2Vec2CtcStack<T> Stack { get; }

    /// <summary>The quantizer and projections of the contrastive task.</summary>
    public Wav2Vec2Pretraining<T> Pretraining { get; }

    /// <summary>The BLSTM bottleneck: each layer's forward and backward LSTMs and its layer norm.</summary>
    public List<(LSTMLayer<T> Forward, LSTMLayer<T> Backward, LayerNormalizationLayer<T> Norm)> Bottleneck { get; } = new();

    /// <summary>The transposed convolutions back to the waveform.</summary>
    public List<Conv1DTransposeLayer<T>> Decoder { get; } = new();

    /// <summary>Every layer, the encoder's and the CTC head's first.</summary>
    public IReadOnlyList<LayerBase<T>> Layers { get; }

    /// <summary>λ2 of Equation 4.</summary>
    public double ReconstructionWeight { get; }

    /// <summary>The predicted clean waveforms <c>[B, samples]</c> for contextual representations <c>[B, frames, D]</c>;
    /// the decoder's output is zero-padded or cut to <paramref name="samples"/>, the input's length.</summary>
    public Tensor<T> Reconstruct(Tensor<T> hidden, int samples)
    {
        var x = hidden;
        foreach (var (forward, backward, norm) in Bottleneck)
        {
            var f = forward.Forward(x);
            var b = _engine.TensorFlip(backward.Forward(_engine.TensorFlip(x, new[] { 1 })), new[] { 1 });
            x = norm.Forward(_engine.TensorConcatenate(new[] { f, b }, 2));
        }
        var y = _engine.TensorPermute(x, new[] { 0, 2, 1 });
        foreach (var conv in Decoder) y = conv.Forward(y);
        int batch = y.Shape[0], length = y.Shape[2];
        y = _engine.Reshape(y, new[] { batch, length });
        if (length >= samples) return _engine.TensorSlice(y, new[] { 0, 0 }, new[] { batch, samples });
        return _engine.TensorConcatenate(new[] { y, new Tensor<T>(new[] { batch, samples - length }) }, 1);
    }

    /// <summary>
    /// The continual pre-training loss (Equation 4) of noisy waveforms <c>[B, S]</c> and their clean references:
    /// L = L_c + λ1 L_d + λ2 L_r, where wav2vec 2.0's objective supplies the contrastive and diversity terms (with its
    /// feature penalty) and L_r is the mean absolute error of the reconstructed clean waveform (Equation 3).
    /// </summary>
    public (Tensor<T> Total, Tensor<T> Reconstruction) ContinualPretrainingLoss(Tensor<T> noisy, Tensor<T> clean, bool training, Random random,
        bool[,]? mask = null, int[,,]? negatives = null)
    {
        var network = Stack.Network;
        var c = Pretraining.Configuration;
        var output = mask is null
            ? network.Forward(noisy, training, random, maskProbability: c.MaskProbability, featureGradientScale: c.FeatureGradientScale)
            : network.Forward(noisy, training, random, mask: mask, featureGradientScale: c.FeatureGradientScale);
        var drawn = output.Mask ?? mask ?? throw new InvalidOperationException("Pre-training needs a mask.");
        var losses = Pretraining.Loss(output, negatives ?? Pretraining.SampleNegatives(drawn, random), training, random);
        var predicted = Reconstruct(output.Hidden, noisy.Shape[1]);
        var reconstruction = _engine.ReduceMean(_engine.TensorAbs(_engine.TensorSubtract(predicted, clean)), new[] { 0, 1 }, keepDims: false);
        reconstruction = _engine.Reshape(reconstruction, new[] { 1 });
        var total = _engine.TensorAdd(_engine.Reshape(losses.Total, new[] { 1 }),
            _engine.TensorMultiplyScalar(reconstruction, NumOps.FromDouble(ReconstructionWeight)));
        return (total, reconstruction);
    }
}
