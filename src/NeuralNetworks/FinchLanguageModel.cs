using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks.Options;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Implements a full RWKV-6 "Finch" language model: token embedding + N RWKVLayer blocks + RMS normalization + LM head.
/// </summary>
/// <remarks>
/// <para>
/// Finch extends Eagle (RWKV-5) with data-dependent token shifting via a LoRA-based mechanism,
/// allowing the model to dynamically adjust how much to blend current and previous tokens.
/// </para>
/// <para><b>For Beginners:</b> Finch (RWKV-6) builds on Eagle by adding the ability to
/// dynamically decide how much context from previous tokens to blend into the current one.
/// Think of it as a reader who can dynamically adjust their focus: sometimes reading word
/// by word, other times absorbing whole phrases. This adaptive blending helps it better
/// capture complex language patterns while maintaining the same memory-efficient inference
/// as its predecessor.</para>
/// <para>
/// <b>Reference:</b> Peng et al., "Eagle and Finch: RWKV with Matrix-Valued States and Dynamic Recurrence", 2024.
/// https://arxiv.org/abs/2404.05892
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var options = new FinchOptions { VocabSize = 65536, ModelDim = 2560, NumLayers = 32, NumHeads = 40 };
/// var model = new FinchLanguageModel&lt;float&gt;(options);
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
[ResearchPaper("Eagle and Finch: RWKV with Matrix-Valued States and Dynamic Recurrence", "https://arxiv.org/abs/2404.05892", Year = 2024, Authors = "Bo Peng, Daniel Goldstein, Quentin Anthony, Alon Albalak, Eric Alcaide, Stella Biderman, Eugene Cheah, Teddy Ferdinan, Haowen Hou, Przemyslaw Kazienko, Kranthi Kiran GV, Jan Kocon, Bartlomiej Koptyra, Satyapriya Krishna, Ronald McClelland Jr., Niklas Muennighoff, Fares Obeid, Atsushi Saito, Guangyu Song, Haoqin Tu, Stanislaw Wozniak, Ruichong Zhang, Bingchen Zhao, Qihang Zhao, Peng Zhou, Jian Zhu, Rui-Jie Zhu")]
public class FinchLanguageModel<T> : TokenLanguageModelLayoutBase<T>
{
    private readonly FinchOptions _options;
    private readonly int _vocabSize;
    private readonly int _modelDimension;
    private readonly int _numLayers;
    private readonly int _numHeads;
    private readonly int _maxSeqLength;
    private readonly double _learningRate;

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <inheritdoc />
    public override ModelOptions GetOptions() => _options;

    /// <summary>Gets the vocabulary size.</summary>
    public int VocabSize => _vocabSize;

    /// <summary>Gets the model dimension (d_model).</summary>
    public int ModelDimension => _modelDimension;

    /// <summary>Gets the number of Finch blocks.</summary>
    public int NumLayers => _numLayers;

    #region Constructors

    public FinchLanguageModel(
        NeuralNetworkArchitecture<T> architecture,
        int vocabSize = 65536,
        int modelDimension = 256,
        int numLayers = 4,
        int numHeads = 8,
        int maxSeqLength = 512,
        ILossFunction<T>? lossFunction = null,
        FinchOptions? options = null,
        double learningRate = 0.001)
        : base(architecture,
            // Raw-logit LM head → cross-entropy-with-logits (fused log-softmax + NLL), not the
            // TextGeneration default CategoricalCrossEntropy (which log()s un-normalized logits and
            // diverges). Mirrors the RWKV4 fix; same root cause. #1622 follow-on.
            lossFunction ?? new AiDotNet.LossFunctions.CrossEntropyWithLogitsLoss<T>())
    {
        _options = options ?? new FinchOptions();
        Options = _options;
        _vocabSize = vocabSize;
        _modelDimension = modelDimension;
        _numLayers = numLayers;
        _numHeads = numHeads;
        _maxSeqLength = maxSeqLength;
        _learningRate = learningRate;
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
            Layers.AddRange(LayerHelper<T>.CreateFinchLayers(
                _vocabSize, _modelDimension, _numLayers, _numHeads, _maxSeqLength));
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

    // UpdateParameters validated the length and distributed the vector across Layers. The base does
    // both. Its trailing "did the loop consume the whole vector" guard is not lost either -- it
    // protected against sum(layer.ParameterCount) drifting from ParameterCount, and the base derives
    // the count and the distribution from ONE enumeration, so they cannot drift apart.
    // (This model previously applied a second hard-coded SGD step here with a private learning rate
    // and clip bound; that was already corrected, and the clipping now lives in FinchOptions where a
    // caller can reach it.) Removed under AIDN082.

    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            AdditionalInfo = new Dictionary<string, object>
            {
                { "Architecture", "RWKV-6-Finch" },
                { "VocabSize", _vocabSize },
                { "ModelDimension", _modelDimension },
                { "NumLayers", _numLayers },
                { "NumHeads", _numHeads },
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
        writer.Write(_numHeads);
        writer.Write(_maxSeqLength);
    }

    protected override void DeserializeNetworkSpecificData(BinaryReader reader)
    {
        _ = reader.ReadInt32();
        _ = reader.ReadInt32();
        _ = reader.ReadInt32();
        _ = reader.ReadInt32();
        _ = reader.ReadInt32();
    }

    #endregion
}
