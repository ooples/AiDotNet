using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks.Options;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Implements a Jamba language model: embedding + HybridBlockScheduler (Mamba + Attention) + RMS norm + LM head.
/// </summary>
/// <remarks>
/// <para>
/// Jamba from AI21 Labs is a hybrid SSM-Attention model that interleaves Mamba blocks with
/// full attention blocks using the Jamba schedule pattern (every Nth block is attention).
/// This achieves strong quality by leveraging attention's exact retrieval with Mamba's efficient long-range processing.
/// </para>
/// <para><b>For Beginners:</b> Jamba combines Mamba's efficient long-range processing with
/// Transformer attention's precise token interactions for the best of both worlds.</para>
/// <para><b>Reference:</b> Lieber et al., "Jamba: A Hybrid Transformer-Mamba Language Model", 2024.</para>
/// </remarks>
/// <example>
/// <code>
/// var options = new JambaOptions { };
/// var model = new JambaLanguageModel&lt;float&gt;(options);
/// var tokens = Tensor&lt;float&gt;.CreateRandom(new[] { 1, 128 });
/// var logits = model.Predict(tokens);
/// </code>
/// </example>
/// <typeparam name="T">The numeric type used for calculations, typically float or double.</typeparam>
[ModelDomain(ModelDomain.Language)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Transformer)]
[ModelCategory(ModelCategory.RecurrentNetwork)]
[ModelTask(ModelTask.Generation)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Jamba: A Hybrid Transformer-Mamba Language Model", "https://arxiv.org/abs/2403.19887", Year = 2024, Authors = "Opher Lieber, Barak Lenz, Hofit Bata, Gal Cohen, Jhonathan Osin, Itay Dalmedigos, Erez Safahi, Shaked Meirom, Yonatan Belinkov, Shai Shalev-Shwartz, Omri Abend, Raz Alon, Tomer Asida, Amir Bergman, Roman Glozman, Michael Gokhman, Avashalom Manevich, Nir Ratner, Noam Rozen, Erez Shwartz, Mor Zusman, Yoav Shoham")]
public partial class JambaLanguageModel<T> : TokenLanguageModelLayoutBase<T>
{
    private readonly JambaOptions _options;
    private readonly int _vocabSize;
    private readonly int _modelDimension;
    private readonly int _numLayers;
    private readonly int _stateDimension;
    private readonly int _attentionInterval;
    private readonly int _maxSeqLength;

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <inheritdoc />
    public override ModelOptions GetOptions() => _options;

    /// <summary>Gets the vocabulary size.</summary>
    public int VocabSize => _vocabSize;

    /// <summary>Gets the model dimension (d_model).</summary>
    public int ModelDimension => _modelDimension;

    /// <summary>Gets the number of Jamba blocks.</summary>
    public int NumLayers => _numLayers;

    #region Constructors

    public JambaLanguageModel(
        NeuralNetworkArchitecture<T> architecture,
        int vocabSize = 65536,
        int modelDimension = 256,
        int numLayers = 8,
        int stateDimension = 16,
        int attentionInterval = 8,
        int maxSeqLength = 512,
        ILossFunction<T>? lossFunction = null,
        JambaOptions? options = null)
        : base(architecture,
            // The LM head emits raw vocabulary logits. Match PyTorch's
            // nn.CrossEntropyLoss contract by applying log-softmax inside
            // the loss instead of treating logits as probabilities.
            lossFunction ?? new LossFunctions.CrossEntropyWithLogitsLoss<T>())
    {
        _options = options ?? new JambaOptions();
        Options = _options;
        _vocabSize = vocabSize;
        _modelDimension = modelDimension;
        _numLayers = numLayers;
        _stateDimension = stateDimension;
        _attentionInterval = attentionInterval;
        _maxSeqLength = maxSeqLength;
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
            Layers.AddRange(LayerHelper<T>.CreateJambaLayers(
                _vocabSize, _modelDimension, _numLayers, _stateDimension, _attentionInterval, _maxSeqLength));
        }
    }

    #endregion

    #region NeuralNetworkBase Overrides

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

    // UpdateParameters applied a GRADIENT STEP, but its one-argument form is the value setter and every caller passes values -- the override corrupted the model. Removed under AIDN082.
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            AdditionalInfo = new Dictionary<string, object>
            {
                { "Architecture", "Jamba" },
                { "VocabSize", _vocabSize },
                { "ModelDimension", _modelDimension },
                { "NumLayers", _numLayers },
                { "StateDimension", _stateDimension },
                { "AttentionInterval", _attentionInterval },
                { "MaxSeqLength", _maxSeqLength },
                { "LayerCount", Layers.Count }
            },
            ModelData = SerializeForMetadata()
        };
    }





    #endregion
}
