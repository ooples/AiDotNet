using AiDotNet.LearningRateSchedulers;
using AiDotNet.Optimizers;

namespace AiDotNet.ComputerVision.Segmentation.Foundation;

/// <summary>SAM 2's training recipe (Ravi et al. 2024, Table 12b), with the official config's details where the paper is
/// silent.</summary>
internal sealed record Sam2Recipe
{
    public double LearningRate { get; init; } = 3e-4;

    public double ImageEncoderLearningRate { get; init; } = 6e-5;

    /// <summary>The per-block layer-wise decay of the Hiera learning rates.</summary>
    public double LayerDecay { get; init; } = 0.9;

    public double WeightDecay { get; init; } = 0.1;

    public double MaxGradientNorm { get; init; } = 0.1;

    public int Steps { get; init; } = 150_000;

    public int WarmupSteps { get; init; } = 7_500;

    /// <summary>The paper's layer-wise decay for a released size (0 Tiny, 1 Small, 2 Base+, 3 Large).</summary>
    public static double PaperLayerDecay(int size) => size switch { 0 or 1 => 0.8, 2 => 0.9, _ => 0.925 };
}

/// <summary>
/// Builds SAM 2's optimizer: AdamW (β 0.9, 0.999) with one learning rate for the image encoder and another for the rest,
/// the Hiera blocks' rates decayed layer-wise from the top (the official <c>layer_decay_param_modifier</c>: block i of B
/// at decay^(B − i), the patch embedding at decay^(B + 1), the position embeddings undecayed), weight decay on every
/// parameter but biases and LayerNorms, gradient-norm clipping, and a linear warmup into a cosine decay to a tenth.
/// </summary>
internal static class Sam2Optimizer
{
    public static (List<Tensor<T>> Parameters, AdamWOptimizer<T, Tensor<T>, Tensor<T>> Optimizer) Create<T>(Sam2Network<T> network,
        IReadOnlyList<LayerBase<T>> layers, Sam2Recipe recipe)
    {
        var optimizer = new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(null, new AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
        {
            InitialLearningRate = recipe.LearningRate,
            Beta1 = 0.9,
            Beta2 = 0.999,
            Epsilon = 1e-8,
            WeightDecay = recipe.WeightDecay,
            UseAdaptiveLearningRate = false,
            UseAdaptiveBetas = false,
            EnableGradientClipping = recipe.MaxGradientNorm > 0,
            GradientClippingMethod = GradientClippingMethod.ByNorm,
            MaxGradientNorm = recipe.MaxGradientNorm > 0 ? recipe.MaxGradientNorm : 1.0,
            LearningRateScheduler = new LinearWarmupScheduler(recipe.LearningRate, recipe.WarmupSteps, recipe.Steps, 0.0,
                LinearWarmupScheduler.DecayMode.Cosine, recipe.LearningRate / 10),
            SchedulerStepMode = SchedulerStepMode.StepPerBatch,
        });
        return (Refresh(optimizer, network, layers, recipe), optimizer);
    }

    /// <summary>Sets the optimizer's per-parameter rates and decay exemptions from the network's current parameter tensors
    /// (loading weights may replace them) and returns those parameters in mask order.</summary>
    public static List<Tensor<T>> Refresh<T>(AdamWOptimizer<T, Tensor<T>, Tensor<T>> optimizer, Sam2Network<T> network,
        IReadOnlyList<LayerBase<T>> layers, Sam2Recipe recipe)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        double encoder = recipe.ImageEncoderLearningRate / recipe.LearningRate, decay = recipe.LayerDecay;
        int blocks = network.Blocks.Count;
        var rate = new Dictionary<LayerBase<T>, double>();
        rate[network.BackgroundPosition] = encoder;
        rate[network.WindowPosition] = encoder;
        rate[network.PatchProjection] = encoder * Math.Pow(decay, blocks + 1);
        for (int i = 0; i < blocks; i++)
        {
            var b = network.Blocks[i];
            foreach (var layer in new LayerBase<T>?[] { b.Norm1, b.Qkv, b.Projection, b.Shortcut, b.Norm2, b.Mlp1, b.Mlp2 })
                if (layer is not null) rate[layer] = encoder * Math.Pow(decay, blocks - i);
        }
        foreach (var neck in network.NeckConvs) rate[neck] = encoder;
        var biasValues = new HashSet<LayerBase<T>>(network.BiasValueLayers());

        var parameters = new List<Tensor<T>>();
        var rates = new List<double>();
        var decays = new List<double>();
        foreach (var layer in layers)
        {
            double r = rate.TryGetValue(layer, out var v) ? v : 1.0;
            foreach (var p in Training.TapeTrainingStep<T>.CollectParameters(new ILayer<T>[] { layer }))
            {
                bool exempt = layer is LayerNormalizationLayer<T> || biasValues.Contains(layer) || (layer is DenseLayer<T> && p.Rank == 1);
                parameters.Add(p);
                for (int i = 0; i < p.Length; i++)
                {
                    rates.Add(r);
                    decays.Add(exempt ? 0.0 : 1.0);
                }
            }
        }
        var options = (AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>)optimizer.GetOptions();
        options.WeightDecayMask = new Vector<T>(decays.Select(ops.FromDouble).ToArray());
        options.LearningRateMask = new Vector<T>(rates.Select(ops.FromDouble).ToArray());
        options.MaskParameters = parameters;
        return parameters;
    }
}
