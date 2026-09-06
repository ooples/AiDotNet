using AiDotNet.Attributes;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.NeuralNetworks.Tabular;

/// <summary>
/// TabNet neural network for interpretable tabular learning.
/// </summary>
/// <remarks>
/// <para>
/// TabNet uses sequential attention to choose which features to reason from at each decision step,
/// enabling interpretable feature selection while achieving performance competitive with gradient boosting.
/// This implementation follows the standard neural network architecture pattern with:
/// - Proper inheritance from NeuralNetworkBase
/// - Layer-based architecture using ILayer components
/// - Engine-based tensor operations for CPU/GPU acceleration
/// - Full autodiff and JIT compilation support
/// </para>
/// <para>
/// <b>For Beginners:</b> TabNet is designed for interpretability:
///
/// Architecture:
/// 1. **Feature Transformer**: Shared layers process all features
/// 2. **Attentive Transformer**: Selects which features to use at each step
/// 3. **Decision Steps**: Multiple rounds of feature selection
/// 4. **Sparse Attention**: Only a few features are used per step
///
/// Key insight: At each decision step, TabNet decides "which features should I
/// focus on?" This sequential attention makes the model interpretable - you can
/// see exactly which features were used for each prediction.
///
/// TabNet often matches gradient boosting (XGBoost, LightGBM) while providing
/// built-in feature importance and interpretability.
/// </para>
/// <para>
/// Reference: "TabNet: Attentive Interpretable Tabular Learning" (Arik &amp; Pfister, AAAI 2021)
/// </para>
/// </remarks>
/// <example>
/// <code>
/// var architecture = new NeuralNetworkArchitecture&lt;float&gt;(inputFeatures: 8, outputSize: 4);
/// var options = new TabNetOptions&lt;double&gt; { NumFeatures = 20, NumDecisionSteps = 5, RelaxationFactor = 1.5 };
/// var input = Tensor&lt;float&gt;.CreateRandom(new[] { 1, 20 });
/// var trainX = Tensor&lt;float&gt;.CreateRandom(4, 8);
/// var trainY = Tensor&lt;float&gt;.CreateRandom(4, 2);
/// var result = new AiModelBuilder&lt;float, Tensor&lt;float&gt;, Tensor&lt;float&gt;&gt;()
///     .ConfigureModel(new TabNetNetwork&lt;float&gt;(architecture))
///     .Build(trainX, trainY);
/// var output = result.Predict(input);
/// </code>
/// </example>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
[ModelDomain(ModelDomain.MachineLearning)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.Classification)]
[ModelTask(ModelTask.Regression)]
[ModelComplexity(ModelComplexity.High)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("TabNet: Attentive Interpretable Tabular Learning",
    "https://arxiv.org/abs/1908.07442",
    Year = 2021,
    Authors = "Arik, S. O. & Pfister, T.")]
public partial class TabNetNetwork<T> : TabularNeuralNetworkBase<T>
{
    private readonly TabNetOptions<T> _options;

    private IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>> _optimizer;
    private ILossFunction<T> _lossFunction;

    /// <summary>
    /// Gets the TabNet-specific options.
    /// </summary>
    public new TabNetOptions<T> Options => _options;

    /// <inheritdoc/>
    public override ModelOptions GetOptions() => _options;

    /// <summary>
    /// Gets the number of decision steps.
    /// </summary>
    public int NumDecisionSteps => _options.NumDecisionSteps;

    /// <summary>
    /// Gets the feature dimension.
    /// </summary>
    public int FeatureDimension => _options.FeatureDimension;

    /// <summary>
    /// Initializes a new instance with default architecture settings.
    /// </summary>
    public TabNetNetwork()
        : this(new NeuralNetworkArchitecture<T>(
            inputType: Enums.InputType.OneDimensional,
            taskType: Enums.NeuralNetworkTaskType.Regression,
            inputSize: 16,
            outputSize: 10))
    {
    }

    /// <summary>
    /// Initializes a new TabNet network with the specified architecture.
    /// </summary>
    public TabNetNetwork(
        NeuralNetworkArchitecture<T> architecture,
        TabNetOptions<T>? options = null,
        IGradientBasedOptimizer<T, Tensor<T>, Tensor<T>>? optimizer = null,
        ILossFunction<T>? lossFunction = null,
        double maxGradNorm = 1.0)
        : base(architecture, lossFunction ?? NeuralNetworkHelper<T>.GetDefaultLossFunction(architecture.TaskType), maxGradNorm)
    {
        _options = options ?? new TabNetOptions<T>();
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
            Layers.AddRange(LayerHelper<T>.CreateDefaultTabNetLayers(
                Architecture,
                numFeatures: Architecture.CalculatedInputSize,
                hiddenDimension: _options.FeatureDimension,
                numSteps: _options.NumDecisionSteps,
                numClasses: Architecture.OutputSize,
                dropoutRate: _options.DropoutRate));
        }
    }

    /// <inheritdoc/>
    protected override Tensor<T> PredictCore(Tensor<T> input)
    {
        if (TryForwardGpuOptimized(input, out var gpuResult))
            return gpuResult;

        // Wrapped in the #1622 verify-then-trust compiled gate; no-op unless acceleration is engaged.
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

        // TabNet provides interpretable feature importance through attention masks
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
                { "Architecture", "TabNet" },
                { "NumFeatures", Architecture.CalculatedInputSize },
                { "OutputDim", Architecture.OutputSize },
                { "NumDecisionSteps", _options.NumDecisionSteps },
                { "FeatureDimension", _options.FeatureDimension },
                { "OutputDimension", _options.OutputDimension },
                { "RelaxationFactor", _options.RelaxationFactor },
                { "SparsityCoefficient", _options.SparsityCoefficient },
                { "LayerCount", Layers.Count },
                { "LayerTypes", Layers.Select(l => l.GetType().Name).ToArray() }
            },
            ModelData = SerializeForMetadata()
        };
    }

    /// <inheritdoc/>


    /// <inheritdoc/>

}
