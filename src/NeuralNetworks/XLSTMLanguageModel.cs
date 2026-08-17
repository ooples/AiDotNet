using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks.Options;
using AiDotNet.Optimizers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.NeuralNetworks;

/// <summary>
/// Implements a full xLSTM language model: token embedding + N ExtendedLSTMLayer blocks + RMS normalization + LM head.
/// </summary>
/// <remarks>
/// <para>
/// xLSTM (Extended LSTM) modernizes the classic LSTM architecture with exponential gating, new memory
/// structures, and residual block stacking to achieve competitive language modeling performance.
/// </para>
/// <para><b>For Beginners:</b> xLSTM is a modern version of the classic LSTM that uses stronger gates
/// and richer memory to achieve quality competitive with Transformers and Mamba while maintaining
/// linear-time inference.</para>
/// <para><b>Reference:</b> Beck et al., "xLSTM: Extended Long Short-Term Memory", 2024.</para>
/// </remarks>
/// <example>
/// <code>
/// var options = new XLSTMOptions { VocabSize = 32000, ModelDim = 2048, NumLayers = 24 };
/// var model = new XLSTMLanguageModel&lt;float&gt;(options);
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
[ResearchPaper("xLSTM: Extended Long Short-Term Memory", "https://arxiv.org/abs/2405.04517", Year = 2024, Authors = "Maximilian Beck, Korbinian Poppel, Markus Spanring, Andreas Auer, Oleksandra Prudnikova, Michael Kopp, Gunter Klambauer, Johannes Brandstetter, Sepp Hochreiter")]
public partial class XLSTMLanguageModel<T> : TokenLanguageModelLayoutBase<T>
{
    private readonly XLSTMOptions _options;
    private readonly int _vocabSize;
    private readonly int _modelDimension;
    private readonly int _numLayers;
    private readonly int _numHeads;
    private readonly int _maxSeqLength;
    private readonly IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? _optimizer;

    /// <inheritdoc />
    public override bool SupportsTraining => true;

    /// <inheritdoc />
    public override ModelOptions GetOptions() => _options;

    /// <summary>Gets the vocabulary size.</summary>
    public int VocabSize => _vocabSize;

    /// <summary>Gets the model dimension (d_model).</summary>
    public int ModelDimension => _modelDimension;

    /// <summary>Gets the number of xLSTM blocks.</summary>
    public int NumLayers => _numLayers;

    #region Constructors

    public XLSTMLanguageModel(
        NeuralNetworkArchitecture<T> architecture,
        int vocabSize = 50277,
        int modelDimension = 256,
        int numLayers = 4,
        int numHeads = 8,
        int maxSeqLength = 512,
        ILossFunction<T>? lossFunction = null,
        XLSTMOptions? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null)
        : base(architecture,
            // xLSTM is a next-token language model and its LM head emits raw logits.
            // Keep the softmax inside the loss, where max-shifted log-sum-exp is
            // numerically stable and tape/compiled-graph safe (the same contract as
            // PyTorch CrossEntropyLoss). A caller-supplied loss still takes precedence.
            lossFunction ?? new AiDotNet.LossFunctions.CrossEntropyWithLogitsLoss<T>())
    {
        _options = options ?? new XLSTMOptions();
        Options = _options;
        _vocabSize = vocabSize;
        _modelDimension = modelDimension;
        _numLayers = numLayers;
        _numHeads = numHeads;
        _maxSeqLength = maxSeqLength;
        // xLSTM (Beck et al., 2024, S4.1) trains with AdamW. No optimizer was wired here at all, so
        // training fell through to the framework default and barely moved: across the memorization
        // task the loss drifted only 0.13% between one and two iterations, leaving the more-data
        // invariant inside its own noise floor.
        _optimizer = optimizer ?? new AdamWOptimizer<T, Tensor<T>, Tensor<T>>(this,
            new AiDotNet.Models.Options.AdamWOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = _options.LearningRate,
            });
        InitializeLayers();
    }

    /// <inheritdoc/>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        SetTrainingMode(true);
        try
        {
            TrainWithTape(input, expectedOutput, _optimizer);
        }
        finally
        {
            SetTrainingMode(false);
        }
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
            Layers.AddRange(LayerHelper<T>.CreateXLSTMLayers(
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

    // UpdateParameters restated the base verbatim; ModelBase routes it to SetParameters.
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            AdditionalInfo = new Dictionary<string, object>
            {
                { "Architecture", "xLSTM" },
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
