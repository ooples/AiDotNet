using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.NeuralNetworks.Tabular;

/// <summary>
/// Mambular (State Space Model for Tabular Data) neural network.
/// </summary>
/// <remarks>
/// <para>
/// Mambular applies the Mamba state space model architecture to tabular data,
/// treating features as a sequence and using selective state spaces for processing.
/// This implementation follows the standard neural network architecture pattern with:
/// - Proper inheritance from NeuralNetworkBase
/// - Layer-based architecture using ILayer components
/// - Engine-based tensor operations for CPU/GPU acceleration
/// - Full autodiff and JIT compilation support
/// </para>
/// <para>
/// <b>For Beginners:</b> Mambular treats features like a sequence:
///
/// Architecture:
/// 1. **Feature Embedding**: Convert features to learned representations
/// 2. **State Space Layers**: Process features sequentially with memory
/// 3. **Selective Mechanism**: Learn which features to remember/forget
/// 4. **MLP Head**: Final prediction from processed features
///
/// Key insight: State space models (like Mamba) are more efficient than
/// transformers for long sequences. For tabular data with many features,
/// this can provide both better scaling and learned sequential relationships.
/// </para>
/// <para>
/// Reference: "Mambular: A Sequential Model for Tabular Deep Learning" (2024)
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;float&gt;(inputFeatures: 8, outputSize: 4);
/// var options = new MambularOptions&lt;double&gt; { NumFeatures = 20, NumLayers = 4 };
/// var model = new MambularNetwork&lt;float&gt;(architecture);
/// var input = Tensor&lt;float&gt;.CreateRandom(new[] { 1, 20 });
/// var output = model.Predict(input);
/// </code>
/// </example>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Classification)]
[ModelTask(ModelTask.Regression)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Mambular: A Sequential Model for Tabular Deep Learning",
    "https://arxiv.org/abs/2408.06291",
    Year = 2024,
    Authors = "Thielmann, A., Kruse, R., Samiee, S., & Kleyko, D.")]
public partial class MambularNetwork<T> : TabularNeuralNetworkBase<T>
{
    private readonly MambularOptions<T> _options;

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private ILossFunction<T> _lossFunction;

    /// <summary>
    /// Gets the Mambular-specific options.
    /// </summary>
    public new MambularOptions<T> Options => _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    /// <summary>
    /// Gets the embedding dimension.
    /// </summary>
    public int EmbeddingDimension => _options.EmbeddingDimension;

    /// <summary>
    /// Gets the state dimension for the SSM.
    /// </summary>
    public int StateDimension => _options.StateDimension;

    /// <summary>
    /// Gets the number of layers.
    /// </summary>
    public int NumLayers => _options.NumLayers;

    /// <summary>
    /// Initializes a new instance with default architecture settings.
    /// </summary>
    public MambularNetwork()
        : this(new NeuralNetworkArchitecture<T>(
            inputType: Enums.InputType.OneDimensional,
            taskType: Enums.NeuralNetworkTaskType.Regression,
            inputSize: 10,
            outputSize: 10))
    {
    }

    /// <summary>
    /// Initializes a new Mambular network with the specified architecture.
    /// </summary>
    public MambularNetwork(
        NeuralNetworkArchitecture<T> architecture,
        MambularOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        double maxGradNorm = 1.0)
        : base(architecture, lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType), maxGradNorm)
    {
        _options = options ?? new MambularOptions<T>();
        _lossFunction = lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType);
        _optimizer = optimizer ?? new AdamOptimizer<T, Tensor<T>, Tensor<T>>(this);

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
            Layers.AddRange(LayerHelper<T>.CreateDefaultMambularLayers(
                Architecture,
                numFeatures: Architecture.CalculatedInputSize,
                embeddingDimension: _options.EmbeddingDimension,
                stateDimension: _options.StateDimension,
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
                { "Architecture", "Mambular" },
                { "NumFeatures", Architecture.CalculatedInputSize },
                { "OutputDim", Architecture.OutputSize },
                { "EmbeddingDimension", _options.EmbeddingDimension },
                { "StateDimension", _options.StateDimension },
                { "NumLayers", _options.NumLayers },
                { "ExpansionFactor", _options.ExpansionFactor },
                { "UseBidirectional", _options.UseBidirectional },
                { "LayerCount", Layers.Count },
                { "LayerTypes", Layers.Select(l => l.GetType().Name).ToArray() }
            },
            ModelData = SerializeForMetadata()
        };
    }

    /// <inheritdoc/>


    /// <inheritdoc/>

}
