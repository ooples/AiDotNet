using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.NeuralNetworks.Options;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Implements a RecurrentGemma language model: embedding + N RGLR blocks + layer norm + LM head.
/// </summary>
/// <remarks>
/// <para>
/// RecurrentGemma is Google's production recurrent language model based on the Griffin architecture.
/// It uses Real-Gated Linear Recurrence (RGLR) blocks for O(n) complexity and O(1) per-token generation.
/// </para>
/// <para><b>For Beginners:</b> RecurrentGemma is Google's production model that uses recurrence instead of
/// attention, giving O(n) complexity and constant memory per token during text generation.</para>
/// <para><b>Reference:</b> Botev et al., "RecurrentGemma: Moving Past Transformers for Efficient Open Language Models", 2024.</para>
/// </remarks>
/// <example>
/// <code>
/// var options = new RecurrentGemmaOptions { };
/// var model = new RecurrentGemmaLanguageModel&lt;float&gt;(options);
/// var tokens = Tensor&lt;float&gt;.CreateRandom(new[] { 1, 128 });
/// var logits = model.Predict(tokens);
/// </code>
/// </example>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
[ModelDomain(ModelDomain.Language)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.RecurrentNetwork)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("RecurrentGemma: Moving Past Transformers for Efficient Open Language Models", "https://arxiv.org/abs/2404.07839", Year = 2024, Authors = "Aleksandar Botev, Soham De, Samuel L. Smith, Anushan Fernando, George-Cristian Muraru, Ruba Haroun, Leonard Berrada, Razvan Pascanu, Pier Giuseppe Sessa, Robert Dadashi, Leonard Hussenot, Johan Ferret, Sertan Girgin, Olivier Bachem, Alek Andreev, Kathleen Kenealy, Thomas Mesnard, Cassidy Hardin, Surya Bhupatiraju, Shreya Pathak, Laurent Sifre, Morgane Riviere, Mihir Sanjay Kale, Juliette Love, Pouya Tafti, Armand Joulin, Noah Fiedel, Evan Senter, Yutian Chen, Srivatsan Srinivasan, Guillaume Desjardins, David Budden, Arnaud Doucet, Koray Kavukcuoglu, Nando De Freitas")]
public partial class RecurrentGemmaLanguageModel<T> : TokenLanguageModelLayoutBase<T>
{
    private readonly RecurrentGemmaOptions _options;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private readonly int _vocabSize;
    private readonly int _modelDimension;
    private readonly int _numLayers;
    private readonly int _maxSeqLength;

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <inheritdoc />
    public override ModelOptions GetOptions() => _options;

    /// <summary>Gets the vocabulary size.</summary>
    public int VocabSize => _vocabSize;

    /// <summary>Gets the model dimension (d_model).</summary>
    public int ModelDimension => _modelDimension;

    /// <summary>Gets the number of RecurrentGemma blocks.</summary>
    public int NumLayers => _numLayers;

    #region Constructors

    public RecurrentGemmaLanguageModel(
        NeuralNetworkArchitecture<T> architecture,
        int vocabSize = 256000,
        int modelDimension = 256,
        int numLayers = 4,
        int maxSeqLength = 512,
        ILossFunction<T>? lossFunction = null,
        RecurrentGemmaOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture,
            // The recurrent Gemma LM head emits raw logits. Use the paper-faithful
            // fused log-softmax/NLL objective rather than categorical CE, which expects
            // probabilities and can produce non-finite gradients when fed logits.
            lossFunction ?? new AiDotNet.LossFunctions.CrossEntropyWithLogitsLoss<T>())
    {
        _options = options ?? new RecurrentGemmaOptions();
        Options = _options;
        _vocabSize = vocabSize;
        _modelDimension = modelDimension;
        _numLayers = numLayers;
        _maxSeqLength = maxSeqLength;
        _optimizer = optimizer ?? CreateDefaultOptimizer();
        InitializeLayers();
    }

    #endregion

    #region Initialization

    protected override void InitializeLayers()
    {
        if (Architecture.Layers != null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
        }
        else
        {
            Layers.AddRange(LayerHelper<T>.CreateRecurrentGemmaLayers(
                _vocabSize, _modelDimension, _numLayers, _maxSeqLength));
        }

        // RecurrentGemma, Section 2: "multiply the input embeddings by a constant equal to the square
        // root of model width." Set on the embedding only -- the paper states the constant is not
        // applied to the output, so the LM head is left alone. EmbeddingLayer already implements the
        // scaling (Vaswani et al. 2017 Section 3.4); it just defaults to off.
        if (_options.ScaleEmbeddingsBySqrtWidth)
        {
            var embedding = Layers.OfType<EmbeddingLayer<T>>().FirstOrDefault();

            // Say so, rather than doing nothing. The option is only reachable through an
            // EmbeddingLayer, so with a caller-supplied architecture that has none -- pre-embedded
            // inputs, or a custom layer stack -- asking for the scaling used to be accepted and then
            // silently skipped. The model then trained WITHOUT the sqrt(d_model) factor the paper
            // requires while reporting the option as enabled, which is exactly the kind of quiet
            // substitution that only shows up as slightly-wrong convergence much later.
            if (embedding is null)
            {
                throw new InvalidOperationException(
                    $"{nameof(RecurrentGemmaOptions.ScaleEmbeddingsBySqrtWidth)} is enabled, but this " +
                    $"model has no {nameof(EmbeddingLayer<T>)} to apply it to. RecurrentGemma applies " +
                    "the sqrt(model width) constant to the input embeddings (Section 2), so the option " +
                    "cannot be honoured by any other layer. Either supply an architecture that includes " +
                    "an embedding layer, or leave the option off if the inputs are already embedded.");
            }

            embedding.ScaleBySqrtDimension = true;
        }
    }

    #endregion

    #region NeuralNetworkBase Overrides

    /// <summary>
    /// Uses the constructor-selected optimizer, as the Griffin and Hawk siblings do.
    /// </summary>
    /// <remarks>
    /// Without this the model fell through to the base default, so none of the training configuration
    /// its options describe reached the optimizer at all.
    /// </remarks>
    protected override IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> GetOrCreateBaseOptimizer()
        => _optimizer;

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> CreateDefaultOptimizer()
    {
        // Validated HERE, at the boundary where the caller's numbers turn into an optimizer. Neither
        // AdamWOptimizerOptions nor AdamWOptimizer range-checks any of these, so a negative or NaN
        // learning rate, or a beta at/above 1, was accepted and produced silently invalid training --
        // NaN moments, or a bias correction dividing by zero. The failure surfaced much later as a
        // non-finite loss with nothing pointing back at the option that caused it.
        AiDotNet.Validation.Guard.Positive(_options.LearningRate);
        AiDotNet.Validation.Guard.NonNegative(_options.WeightDecay);
        AiDotNet.Validation.Guard.Positive(_options.Epsilon);

        // Half-open [0, 1): Adam's bias correction divides by (1 - beta^t), so beta == 1 is a division
        // by zero on the first step, and the running averages never decay.
        AiDotNet.Validation.Guard.InRange(_options.Beta1, 0.0, NearestBelowOne);
        AiDotNet.Validation.Guard.InRange(_options.Beta2, 0.0, NearestBelowOne);

        // Only meaningful when clipping is on; an unset default must not be rejected.
        if (_options.EnableGradientClipping)
            AiDotNet.Validation.Guard.Positive(_options.MaxGradientNorm);

        return new AiDotNet.Optimizers.AdamWOptimizer<T, Tensor<T>, Tensor<T>>(
            this,
            new AiDotNet.Models.Options.AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = _options.LearningRate,
                WeightDecay = _options.WeightDecay,
                Beta1 = _options.Beta1,
                Beta2 = _options.Beta2,
                Epsilon = _options.Epsilon,
                EnableGradientClipping = _options.EnableGradientClipping,
                MaxGradientNorm = _options.MaxGradientNorm
            });
    }

    /// <summary>
    /// The largest double strictly below 1, used as the inclusive upper bound for the Adam betas.
    /// </summary>
    /// <remarks>
    /// <see cref="AiDotNet.Validation.Guard.InRange(double, double, double, string?)"/> is inclusive on
    /// both ends, and the betas need a half-open [0, 1). Bounding at the representable neighbour below
    /// 1 expresses that exactly, rather than rejecting at some arbitrary epsilon short of it.
    ///
    /// Written as a literal rather than <c>Math.BitDecrement(1.0)</c> because that API does not exist
    /// on net471, which this project still targets.
    /// </remarks>
    private const double NearestBelowOne = 0.99999999999999989;
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        SetTrainingMode(false);
        return Accelerate(input, () =>
        {
            var output = input;
            for (int i = 0; i < Layers.Count; i++)
            {
                output = Layers[i].Forward(output);
            }
            return output;
        });
    }

    // UpdateParameters validated the length and distributed the vector across Layers, both of which
    // the base does. Removed under AIDN082.

    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            AdditionalInfo = new Dictionary<string, object>
            {
                { "Architecture", "RecurrentGemma" },
                { "VocabSize", _vocabSize },
                { "ModelDimension", _modelDimension },
                { "NumLayers", _numLayers },
                { "MaxSeqLength", _maxSeqLength },
                { "LayerCount", Layers.Count }
            },
            ModelData = SerializeForMetadata()
        };
    }





    #endregion
}
