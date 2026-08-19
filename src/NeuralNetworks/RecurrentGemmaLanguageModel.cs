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
/// var options = new RecurrentGemmaOptions { VocabSize = 256000, ModelDim = 2560, NumLayers = 26 };
/// var model = new RecurrentGemmaLanguageModel&lt;float&gt;(options);
/// var tokens = Tensor&lt;float&gt;.Random(new[] { 1, 128 });
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
public class RecurrentGemmaLanguageModel<T> : TokenLanguageModelLayoutBase<T>
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
            foreach (var layer in Layers)
            {
                if (layer is EmbeddingLayer<T> embedding)
                {
                    embedding.ScaleBySqrtDimension = true;
                    break;
                }
            }
        }
    }

    #endregion

    #region NeuralNetworkBase Overrides

    /// <summary>
    /// Trains through the eager tape rather than the fused compiled step, as Griffin and Hawk do.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The three models build the SAME RG-LRU stack, and both siblings already opt out here. This one
    /// did not, so it inherited the base default of <c>true</c> and was the only member of the family
    /// training through the fused path.
    /// </para>
    /// <para>
    /// That path is what produced the non-finite parameters. Measured on this model at its test scale
    /// (vocab 4096, width 256, four layers): the forward is finite with logits in [-1.41, 1.33], the
    /// loss is finite at 376.27, and <c>ComputeGradients</c> — which runs the eager tape — returns
    /// 0 of 3,417,600 entries non-finite with a largest magnitude of 0.12. A single <c>Train</c> call
    /// through the fused step on the identical model returns 3,417,600 of 3,417,600 non-finite. The
    /// gradients are not being computed wrongly; the fused training step is the only stage that turns
    /// them into NaN, which is also why every downstream invariant (parameter finiteness, forward
    /// after training, loss decrease, clone equality) failed together.
    /// </para>
    /// </remarks>
    protected override bool SupportsFusedCompiledTraining => false;

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
        => new AiDotNet.Optimizers.AdamWOptimizer<T, Tensor<T>, Tensor<T>>(
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

    protected override void SerializeNetworkSpecificData(BinaryWriter writer)
    {
        writer.Write(_vocabSize);
        writer.Write(_modelDimension);
        writer.Write(_numLayers);
        writer.Write(_maxSeqLength);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _ = reader.ReadInt32();
        _ = reader.ReadInt32();
        _ = reader.ReadInt32();
        _ = reader.ReadInt32();
    }

    protected override IFullModel<T, Tensor<T>, Tensor<T>> CreateNewInstance()
    {
        return new RecurrentGemmaLanguageModel<T>(
            Architecture, _vocabSize, _modelDimension, _numLayers, _maxSeqLength,
            LossFunction, _options);
    }

    #endregion
}
