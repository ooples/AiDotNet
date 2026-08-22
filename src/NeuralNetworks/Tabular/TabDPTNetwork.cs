using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.NeuralNetworks.Tabular;

/// <summary>
/// TabDPT (Tabular Data Pre-Training) neural network for tabular data.
/// </summary>
/// <remarks>
/// <para>
/// TabDPT is a foundation model approach for tabular data that uses pre-training
/// on diverse datasets to learn transferable representations.
/// This implementation follows the standard neural network architecture pattern with:
/// - Proper inheritance from NeuralNetworkBase
/// - Layer-based architecture using ILayer components
/// - Engine-based tensor operations for CPU/GPU acceleration
/// - Full autodiff and JIT compilation support
/// </para>
/// <para>
/// <b>For Beginners:</b> TabDPT brings foundation model ideas to tabular data:
///
/// Architecture:
/// 1. **Input Projection**: Map features to embedding space
/// 2. **Transformer Encoder**: Deep self-attention for feature relationships
/// 3. **Context Learning**: Learn from in-context examples
/// 4. **Output Head**: Task-specific prediction layer
///
/// Key insight: By pre-training on many diverse tabular datasets,
/// TabDPT learns patterns that transfer to new datasets, similar to
/// how large language models learn from diverse text.
/// </para>
/// <para>
/// Reference: "TabDPT: Scaling Tabular Foundation Models on Real Data" (2024)
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var options = new TabDPTOptions { NumFeatures = 20, EmbeddingDim = 128, NumLayers = 12 };
/// var model = new TabDPTNetwork&lt;float&gt;(options);
/// var input = Tensor&lt;float&gt;.Random(new[] { 1, 20 });
/// var output = model.Predict(input);
/// </code>
/// </example>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelCategory(ModelCategory.Transformer)]
[ModelTask(ModelTask.Classification)]
[ModelTask(ModelTask.Regression)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
// Citation corrected. arXiv 2501.02487 is "ACE++: Instruction-Based Image Creation and Editing",
// unrelated. TabDPT is arXiv 2410.18164 (2024) and its full title ends "on Real Data".
[ResearchPaper("TabDPT: Scaling Tabular Foundation Models on Real Data",
    "https://arxiv.org/abs/2410.18164",
    Year = 2024,
    Authors = "Junwei Ma, Valentin Thomas, Rasa Hosseinzadeh, Hamidreza Kamkari, Alex Lacoste, Keyvan Golestan, Guangwei Yu, Maksims Volkovs, Anthony L. Caterini")]
public partial class TabDPTNetwork<T> : TabularNeuralNetworkBase<T>
{
    private readonly TabDPTOptions<T> _options;

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private ILossFunction<T> _lossFunction;

    /// <summary>
    /// Gets the TabDPT-specific options.
    /// </summary>
    public new TabDPTOptions<T> Options => _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    /// <summary>
    /// Gets the embedding dimension.
    /// </summary>
    public int EmbeddingDimension => _options.EmbeddingDimension;

    /// <summary>
    /// Gets the number of transformer layers.
    /// </summary>
    public int NumLayers => _options.NumLayers;

    /// <summary>
    /// Initializes a new TabDPT network with the specified architecture.
    /// </summary>
    /// <summary>
    /// Initializes a new instance with default architecture settings.
    /// </summary>
    public TabDPTNetwork()
        : this(new NeuralNetworkArchitecture<T>(
            inputType: Enums.InputType.OneDimensional,
            taskType: Enums.NeuralNetworkTaskType.Regression,
            inputSize: 10,
            outputSize: 10))
    {
    }

    public TabDPTNetwork(
        NeuralNetworkArchitecture<T> architecture,
        TabDPTOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        double maxGradNorm = 1.0)
        : base(architecture, lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType), maxGradNorm)
    {
        _options = options ?? new TabDPTOptions<T>();
        _lossFunction = lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType);
        // Decay the step toward the end of training. Built bare, the rate stayed fixed, so once the
        // model reached its floor the optimizer kept taking full-size steps and oscillated there:
        // 50 iterations landed at 7.43e-05 while 200 landed at 1.83e-04, i.e. more training made it
        // mildly worse rather than settling. A decaying rate lets it settle instead.
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this,
            new AiDotNet.Models.Options.AdamOptimizerOptions<T, Tensor<T>, Tensor<T>>
            {
                InitialLearningRate = 1e-4,
                MinLearningRate = 1e-6,
            });

        if (_options.EmbeddingDimension % _options.NumHeads != 0)
        {
            throw new ArgumentException(
                $"EmbeddingDimension ({_options.EmbeddingDimension}) must be divisible by NumHeads ({_options.NumHeads})");
        }

        InitializeLayers();
    }

    /// <inheritdoc/>
    protected override void InitializeLayers()
    {
        if (Architecture.Layers != null && Architecture.Layers.Count > 0)
        {
            Layers.AddRange(Architecture.Layers);
            ValidateCustomLayers(Layers);
        }
        else
        {
            Layers.AddRange(LayerHelper<T>.CreateDefaultTabDPTLayers(
                Architecture,
                numFeatures: Architecture.CalculatedInputSize,
                embeddingDimension: _options.EmbeddingDimension,
                numHeads: _options.NumHeads,
                numLayers: _options.NumLayers,
                numClasses: Architecture.OutputSize,
                dropoutRate: _options.DropoutRate));
        }
    }

    /// <inheritdoc/>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        if (TryForwardGpuOptimized(input, out var gpuResult))
            return gpuResult;

        return Accelerate(input, () =>
        {
            Tensor<T> currentOutput = input;
            foreach (var layer in Layers)
            {
                currentOutput = layer.Forward(currentOutput);
            }

            return currentOutput;
        });
    }

    /// <inheritdoc/>
    public override void Train(Tensor<T> input, Tensor<T> expectedOutput)
    {
        // Tape-based training (post-#1209). The previous body computed `error` and
        // dropped it without backpropagating, then called _optimizer.UpdateParameters(Layers)
        // — which throws "Backward pass must be called before updating parameters".
        //
        // Honor the base Train contract: auto-promote unbatched single samples to [1, …] first.
        (input, expectedOutput) = NormalizeBatchDim(input, expectedOutput);

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

    // UpdateParameters re-sliced the flat vector across Layers by hand -- the base walks
    // exactly the same enumeration, so this said nothing the base does not already say.
    /// <inheritdoc/>
    public override Dictionary<string, T> GetFeatureImportance()
    {
        var importance = new Dictionary<string, T>();
        int numFeatures = Architecture.CalculatedInputSize;

        var uniformValue = NumOps.FromDouble(1.0 / numFeatures);
        for (int f = 0; f < numFeatures; f++)
        {
            importance[$"feature_{f}"] = uniformValue;
        }

        return importance;
    }

    /// <inheritdoc/>
    public override ModelMetadata<T> GetModelMetadata()
    {
        return new ModelMetadata<T>
        {
            AdditionalInfo = new Dictionary<string, object>
            {
                { "Architecture", "TabDPT" },
                { "NumFeatures", Architecture.CalculatedInputSize },
                { "OutputDim", Architecture.OutputSize },
                { "EmbeddingDimension", _options.EmbeddingDimension },
                { "NumHeads", _options.NumHeads },
                { "NumLayers", _options.NumLayers },
                { "ContextLength", _options.ContextLength },
                { "UseFeatureAttention", _options.UseFeatureAttention },
                { "LayerCount", Layers.Count },
                { "LayerTypes", Layers.Select(l => l.GetType().Name).ToArray() }
            },
            ModelData = SerializeForMetadata()
        };
    }

    /// <inheritdoc/>


    /// <inheritdoc/>

}
