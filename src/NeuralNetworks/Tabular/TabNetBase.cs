using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Layers;

namespace AiDotNet.NeuralNetworks.Tabular;

/// <summary>
/// Base implementation of the TabNet architecture for attentive interpretable tabular learning.
/// </summary>
/// <remarks>
/// <para>
/// TabNet is a deep learning architecture specifically designed for tabular data. It uses
/// sequential attention to choose which features to reason from at each decision step,
/// enabling both high performance and interpretability.
/// </para>
/// <para>
/// <b>For Beginners:</b> TabNet is a neural network designed specifically for tables of data
/// (like spreadsheets or databases). Unlike traditional neural networks that process all
/// features at once, TabNet makes decisions in steps:
///
/// 1. **Step 1**: "Which features should I look at first?"
///    - Selects a subset of features using sparse attention
///    - Processes those features through the Feature Transformer
///
/// 2. **Step 2**: "Based on what I learned, which features next?"
///    - Uses information from Step 1 to select new features
///    - Features used before are less likely to be selected again
///
/// 3. **Steps 3, 4, ...**: Continue refining the decision
///
/// At each step, TabNet accumulates knowledge that gets combined into the final output.
///
/// This approach provides several benefits:
/// - **Interpretability**: You can see which features were selected at each step
/// - **Feature Selection**: Automatically ignores irrelevant features
/// - **Efficiency**: Doesn't need to process all features for every prediction
/// - **Performance**: Often matches or beats gradient boosting on tabular data
/// </para>
/// <para>
/// Reference: "TabNet: Attentive Interpretable Tabular Learning" (Arik &amp; Pfister, AAAI 2021)
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
public abstract class TabNetBase<T>
{
    /// <summary>
    /// Numeric operations provider for generic type T.
    /// </summary>
    protected readonly INumericOperations<T> NumOps;

    /// <summary>
    /// Configuration options for TabNet.
    /// </summary>
    protected readonly TabNetOptions<T> Options;

    /// <summary>
    /// Number of input features.
    /// </summary>
    protected readonly int InputDim;

    /// <summary>
    /// Number of output values.
    /// </summary>
    protected readonly int OutputDim;

    // Initial embedding layer (for batch normalization of raw input)
    private readonly GhostBatchNormalization<T> _initialBN;

    // Shared layers across all decision steps
    private readonly List<FullyConnectedLayer<T>> _sharedFCLayers;
    private readonly List<GhostBatchNormalization<T>> _sharedBNLayers;

    // Feature Transformers for each decision step
    private readonly List<FeatureTransformer<T>> _featureTransformers;

    // Attentive Transformers for each decision step
    private readonly List<AttentiveTransformer<T>> _attentiveTransformers;

    // Final output layer
    private readonly FullyConnectedLayer<T> _outputLayer;

    // Cached attention masks for interpretability
    private readonly List<Tensor<T>> _stepAttentionMasks = [];
    private readonly List<Tensor<T>> _stepOutputs = [];

    // Training state
    private bool _isTraining = true;

    /// <summary>
    /// Initializes a new instance of the TabNetBase class.
    /// </summary>
    /// <param name="inputDim">Number of input features.</param>
    /// <param name="outputDim">Number of output values.</param>
    /// <param name="options">TabNet configuration options.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> When creating a TabNet model:
    /// - inputDim: The number of features in your data (columns in your table)
    /// - outputDim: What you want to predict (1 for regression, N for N-class classification)
    /// - options: Configuration settings like number of decision steps, hidden dimensions, etc.
    ///
    /// Example for house price prediction:
    /// - inputDim = 10 (bedrooms, bathrooms, sqft, year built, etc.)
    /// - outputDim = 1 (the price)
    ///
    /// Example for customer churn classification:
    /// - inputDim = 20 (usage patterns, demographics, etc.)
    /// - outputDim = 2 (churn / no churn)
    /// </para>
    /// </remarks>
    protected TabNetBase(int inputDim, int outputDim, TabNetOptions<T>? options = null)
    {
        NumOps = MathHelper.GetNumericOperations<T>();
        Options = options ?? new TabNetOptions<T>();
        InputDim = inputDim;
        OutputDim = outputDim;

        // Initialize batch normalization for raw input
        _initialBN = new GhostBatchNormalization<T>(
            inputDim,
            Options.VirtualBatchSize,
            Options.BatchNormalizationMomentum,
            Options.Epsilon);

        // Initialize shared layers
        _sharedFCLayers = [];
        _sharedBNLayers = [];
        InitializeSharedLayers();

        // Initialize decision step components
        _featureTransformers = [];
        _attentiveTransformers = [];
        InitializeDecisionSteps();

        // Initialize output layer
        _outputLayer = new FullyConnectedLayer<T>(
            Options.OutputDimension,
            outputDim,
            (IActivationFunction<T>?)null);
    }

    /// <summary>
    /// Initializes the shared layers used across all decision steps.
    /// </summary>
    private void InitializeSharedLayers()
    {
        int currentDim = InputDim;
        int hiddenDim = Options.FeatureDimension * 2; // GLU doubles then halves

        for (int i = 0; i < Options.NumSharedLayers; i++)
        {
            var fc = new FullyConnectedLayer<T>(currentDim, hiddenDim, (IActivationFunction<T>?)null);
            _sharedFCLayers.Add(fc);

            var bn = new GhostBatchNormalization<T>(
                hiddenDim,
                Options.VirtualBatchSize,
                Options.BatchNormalizationMomentum,
                Options.Epsilon);
            _sharedBNLayers.Add(bn);

            currentDim = Options.FeatureDimension;
        }
    }

    /// <summary>
    /// Initializes the Feature and Attentive Transformers for each decision step.
    /// </summary>
    private void InitializeDecisionSteps()
    {
        for (int step = 0; step < Options.NumDecisionSteps; step++)
        {
            // Feature Transformer: processes masked features
            // Uses shared layers for efficiency
            var featureTransformer = new FeatureTransformer<T>(
                InputDim,
                Options.FeatureDimension + Options.OutputDimension, // Split between attention and output
                _sharedFCLayers.Count > 0 ? _sharedFCLayers : null,
                _sharedBNLayers.Count > 0 ? _sharedBNLayers : null,
                Options.NumSharedLayers,
                Options.NumStepSpecificLayers,
                Options.VirtualBatchSize,
                Options.BatchNormalizationMomentum,
                Options.Epsilon);
            _featureTransformers.Add(featureTransformer);

            // Attentive Transformer: generates attention mask for next step
            // (not needed for the last step)
            if (step < Options.NumDecisionSteps - 1)
            {
                var attentiveTransformer = new AttentiveTransformer<T>(
                    Options.FeatureDimension,
                    InputDim,
                    Options.RelaxationFactor,
                    Options.VirtualBatchSize,
                    Options.BatchNormalizationMomentum,
                    Options.Epsilon);
                _attentiveTransformers.Add(attentiveTransformer);
            }
        }
    }

    /// <summary>
    /// Performs the forward pass through the TabNet encoder.
    /// </summary>
    /// <param name="input">Input tensor of shape [batch_size, input_dim].</param>
    /// <returns>Output tensor before final activation.</returns>
    /// <remarks>
    /// <para>
    /// The forward pass processes data through multiple decision steps:
    /// 1. Apply initial batch normalization to the raw input
    /// 2. Initialize prior scales to ones (all features equally available)
    /// 3. For each decision step:
    ///    a. Compute attention mask using Attentive Transformer
    ///    b. Apply mask to get selected features
    ///    c. Process through Feature Transformer
    ///    d. Split output into "attention" part and "decision" part
    ///    e. Accumulate decision outputs using ReLU
    ///    f. Update prior scales to discourage feature reuse
    /// 4. Pass aggregated output through final layer
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Here's what happens step by step:
    ///
    /// **Initial Setup:**
    /// - Normalize the input data (helps training stability)
    /// - Set up "prior scales" = all 1s (all features equally available at first)
    ///
    /// **For each decision step:**
    /// 1. **Attention**: Decide which features to focus on
    ///    - Use information from previous step to select features
    ///    - Features used before are penalized (prior scales are lower)
    ///
    /// 2. **Masking**: Apply the attention to the input
    ///    - multiply input features by their attention weights
    ///    - Features with 0 attention are completely ignored
    ///
    /// 3. **Transform**: Process the selected features
    ///    - Extract useful information from the selected features
    ///    - Part of the output helps with next step's attention
    ///    - Part of the output contributes to the final answer
    ///
    /// 4. **Update**: Prepare for the next step
    ///    - Reduce prior scales for features that were heavily used
    ///    - This encourages looking at new features in the next step
    ///
    /// **Final Output:**
    /// - Combine the decision outputs from all steps
    /// - Apply the final layer to get predictions
    /// </para>
    /// </remarks>
    public virtual Tensor<T> Forward(Tensor<T> input)
    {
        if (input.Rank != 2)
        {
            throw new ArgumentException($"Expected 2D input [batch_size, features], got {input.Rank}D", nameof(input));
        }

        int batchSize = input.Shape[0];
        int features = input.Shape[1];

        if (features != InputDim)
        {
            throw new ArgumentException($"Expected {InputDim} features, got {features}", nameof(input));
        }

        _stepAttentionMasks.Clear();
        _stepOutputs.Clear();

        // Step 1: Initial batch normalization
        var normalizedInput = _initialBN.Forward(input);

        // Step 2: Initialize prior scales to ones
        var priorScales = CreateOnes(batchSize, InputDim);

        // Aggregated decision output (accumulated across steps)
        var aggregatedOutput = CreateZeros(batchSize, Options.OutputDimension);

        // Processed features from previous step (for first step, use zeros)
        var processedFeatures = CreateZeros(batchSize, Options.FeatureDimension);

        // Step 3: Process each decision step
        for (int step = 0; step < Options.NumDecisionSteps; step++)
        {
            Tensor<T> attentionMask;

            if (step == 0)
            {
                // First step: uniform attention (use all features equally)
                attentionMask = CreateOnes(batchSize, InputDim);
                // Normalize to sum to 1
                var sum = NumOps.FromDouble(InputDim);
                for (int i = 0; i < attentionMask.Length; i++)
                {
                    attentionMask[i] = NumOps.Divide(attentionMask[i], sum);
                }
            }
            else
            {
                // Subsequent steps: compute attention using Attentive Transformer
                attentionMask = _attentiveTransformers[step - 1].Forward(processedFeatures, priorScales);
            }

            _stepAttentionMasks.Add(attentionMask);

            // Apply attention mask to get selected features
            var maskedInput = ApplyMask(normalizedInput, attentionMask);

            // Process through Feature Transformer
            var transformerOutput = _featureTransformers[step].Forward(maskedInput);

            // Split output: first part for attention, second part for decision
            var (attentionPart, decisionPart) = SplitTensor(
                transformerOutput,
                Options.FeatureDimension,
                Options.OutputDimension);

            processedFeatures = attentionPart;

            // Accumulate decision output with ReLU
            var reluDecision = ApplyReLU(decisionPart);
            aggregatedOutput = AddTensors(aggregatedOutput, reluDecision);

            _stepOutputs.Add(decisionPart);

            // Update prior scales (except for last step)
            if (step < Options.NumDecisionSteps - 1)
            {
                priorScales = _attentiveTransformers[step].UpdatePriorScales(priorScales, attentionMask);
            }
        }

        // Step 4: Final output layer
        var output = _outputLayer.Forward(aggregatedOutput);

        return output;
    }

    /// <summary>
    /// Performs the backward pass through the TabNet model.
    /// </summary>
    /// <param name="outputGradient">Gradient of the loss with respect to the output.</param>
    /// <returns>Gradient with respect to the input.</returns>
    public virtual Tensor<T> Backward(Tensor<T> outputGradient)
    {
        // Backward through output layer
        var aggregatedGrad = _outputLayer.Backward(outputGradient);

        // Initialize input gradient accumulator
        Tensor<T>? inputGrad = null;

        // Backward through decision steps (reverse order)
        for (int step = Options.NumDecisionSteps - 1; step >= 0; step--)
        {
            // Apply ReLU derivative
            var stepOutput = _stepOutputs[step];
            var stepGrad = ApplyReLUDerivative(aggregatedGrad, stepOutput);

            // Backward through Feature Transformer
            var maskedInputGrad = _featureTransformers[step].Backward(stepGrad);

            // Backward through mask (gradient flows to input through attention)
            var attentionMask = _stepAttentionMasks[step];
            var unmaskGrad = ApplyMaskBackward(maskedInputGrad, attentionMask);

            // Accumulate input gradient
            if (inputGrad == null)
            {
                inputGrad = unmaskGrad;
            }
            else
            {
                inputGrad = AddTensors(inputGrad, unmaskGrad);
            }
        }

        // Backward through initial batch normalization
        return _initialBN.Backward(inputGrad ?? throw new InvalidOperationException("No gradients computed"));
    }

    /// <summary>
    /// Computes the sparsity regularization loss.
    /// </summary>
    /// <returns>The total sparsity loss across all decision steps.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The sparsity loss encourages TabNet to be selective about features.
    ///
    /// Without this loss, the model might spread attention across many features.
    /// With this loss, it learns to focus on fewer, more important features.
    ///
    /// This improves:
    /// - Interpretability (clearer which features matter)
    /// - Generalization (less overfitting to noisy features)
    /// - Efficiency (fewer features actively processed)
    /// </para>
    /// </remarks>
    public T ComputeSparsityLoss()
    {
        var totalLoss = NumOps.Zero;
        for (int step = 0; step < _attentiveTransformers.Count; step++)
        {
            var mask = _stepAttentionMasks[step + 1]; // Skip first uniform mask
            var stepLoss = _attentiveTransformers[step].ComputeSparsityLoss(mask);
            totalLoss = NumOps.Add(totalLoss, stepLoss);
        }
        return NumOps.Multiply(totalLoss, NumOps.FromDouble(Options.SparsityCoefficient));
    }

    /// <summary>
    /// Gets the feature importance from all decision steps.
    /// </summary>
    /// <returns>Aggregated feature importance scores.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This shows you which features TabNet considers most important.
    ///
    /// The importance is computed by averaging the attention masks across all decision steps.
    /// Features with higher importance scores were selected more often during inference.
    ///
    /// This is one of TabNet's key advantages - built-in interpretability without needing
    /// additional explanation methods like SHAP or LIME.
    /// </para>
    /// </remarks>
    public Tensor<T> GetFeatureImportance()
    {
        if (_stepAttentionMasks.Count == 0)
        {
            throw new InvalidOperationException("Forward pass must be called first to compute feature importance.");
        }

        int batchSize = _stepAttentionMasks[0].Shape[0];
        var importance = CreateZeros(batchSize, InputDim);

        foreach (var mask in _stepAttentionMasks)
        {
            importance = AddTensors(importance, mask);
        }

        // Normalize by number of steps
        var numSteps = NumOps.FromDouble(_stepAttentionMasks.Count);
        for (int i = 0; i < importance.Length; i++)
        {
            importance[i] = NumOps.Divide(importance[i], numSteps);
        }

        return importance;
    }

    /// <summary>
    /// Gets the attention masks from each decision step.
    /// </summary>
    /// <returns>List of attention masks, one per decision step.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This provides detailed interpretability showing exactly
    /// which features were selected at each decision step.
    ///
    /// You can use this to understand the model's reasoning process:
    /// - Step 1 might focus on basic features (e.g., age, income)
    /// - Step 2 might refine using related features (e.g., employment status)
    /// - Step 3 might incorporate contextual features (e.g., location, time)
    ///
    /// This level of detail is unique to TabNet and helps build trust in the model's predictions.
    /// </para>
    /// </remarks>
    public IReadOnlyList<Tensor<T>> GetStepAttentionMasks() => _stepAttentionMasks.AsReadOnly();

    #region Helper Methods

    /// <summary>
    /// Creates a tensor filled with ones.
    /// </summary>
    protected Tensor<T> CreateOnes(int batchSize, int dim)
    {
        var tensor = new Tensor<T>([batchSize, dim]);
        for (int i = 0; i < tensor.Length; i++)
        {
            tensor[i] = NumOps.One;
        }
        return tensor;
    }

    /// <summary>
    /// Creates a tensor filled with zeros.
    /// </summary>
    protected Tensor<T> CreateZeros(int batchSize, int dim)
    {
        var tensor = new Tensor<T>([batchSize, dim]);
        for (int i = 0; i < tensor.Length; i++)
        {
            tensor[i] = NumOps.Zero;
        }
        return tensor;
    }

    /// <summary>
    /// Applies attention mask to input features.
    /// </summary>
    protected Tensor<T> ApplyMask(Tensor<T> input, Tensor<T> mask)
    {
        var result = new Tensor<T>(input.Shape);
        for (int i = 0; i < input.Length; i++)
        {
            result[i] = NumOps.Multiply(input[i], mask[i]);
        }
        return result;
    }

    /// <summary>
    /// Backward pass for mask application.
    /// </summary>
    protected Tensor<T> ApplyMaskBackward(Tensor<T> gradient, Tensor<T> mask)
    {
        var result = new Tensor<T>(gradient.Shape);
        for (int i = 0; i < gradient.Length; i++)
        {
            result[i] = NumOps.Multiply(gradient[i], mask[i]);
        }
        return result;
    }

    /// <summary>
    /// Splits a tensor along the feature dimension.
    /// </summary>
    protected (Tensor<T> first, Tensor<T> second) SplitTensor(Tensor<T> tensor, int firstDim, int secondDim)
    {
        int batchSize = tensor.Shape[0];
        var first = new Tensor<T>([batchSize, firstDim]);
        var second = new Tensor<T>([batchSize, secondDim]);

        for (int b = 0; b < batchSize; b++)
        {
            for (int i = 0; i < firstDim; i++)
            {
                first[b * firstDim + i] = tensor[b * (firstDim + secondDim) + i];
            }
            for (int i = 0; i < secondDim; i++)
            {
                second[b * secondDim + i] = tensor[b * (firstDim + secondDim) + firstDim + i];
            }
        }

        return (first, second);
    }

    /// <summary>
    /// Applies ReLU activation.
    /// </summary>
    protected Tensor<T> ApplyReLU(Tensor<T> input)
    {
        var result = new Tensor<T>(input.Shape);
        for (int i = 0; i < input.Length; i++)
        {
            result[i] = NumOps.GreaterThan(input[i], NumOps.Zero) ? input[i] : NumOps.Zero;
        }
        return result;
    }

    /// <summary>
    /// Applies ReLU derivative for backward pass.
    /// </summary>
    protected Tensor<T> ApplyReLUDerivative(Tensor<T> gradient, Tensor<T> original)
    {
        var result = new Tensor<T>(gradient.Shape);
        for (int i = 0; i < gradient.Length; i++)
        {
            result[i] = NumOps.GreaterThan(original[i], NumOps.Zero) ? gradient[i] : NumOps.Zero;
        }
        return result;
    }

    /// <summary>
    /// Adds two tensors element-wise.
    /// </summary>
    protected Tensor<T> AddTensors(Tensor<T> a, Tensor<T> b)
    {
        var result = new Tensor<T>(a.Shape);
        for (int i = 0; i < a.Length; i++)
        {
            result[i] = NumOps.Add(a[i], b[i]);
        }
        return result;
    }

    #endregion

    #region Parameter Management

    /// <summary>
    /// Gets all trainable parameters.
    /// </summary>
    public Vector<T> GetParameters()
    {
        var allParams = new List<T>();

        // Initial BN
        AddVectorToList(allParams, _initialBN.GetParameters());

        // Feature Transformers
        foreach (var ft in _featureTransformers)
        {
            AddVectorToList(allParams, ft.GetParameters());
        }

        // Attentive Transformers
        foreach (var at in _attentiveTransformers)
        {
            AddVectorToList(allParams, at.GetParameters());
        }

        // Output layer
        AddVectorToList(allParams, _outputLayer.GetParameters());

        return CreateVectorFromList(allParams);
    }

    /// <summary>
    /// Sets the trainable parameters.
    /// </summary>
    public void SetParameters(Vector<T> parameters)
    {
        int offset = 0;

        // Initial BN
        offset = SetComponentParameters(_initialBN, parameters, offset);

        // Feature Transformers
        foreach (var ft in _featureTransformers)
        {
            offset = SetComponentParameters(ft, parameters, offset);
        }

        // Attentive Transformers
        foreach (var at in _attentiveTransformers)
        {
            offset = SetComponentParameters(at, parameters, offset);
        }

        // Output layer
        SetComponentParameters(_outputLayer, parameters, offset);
    }

    private void AddVectorToList(List<T> list, Vector<T> vector)
    {
        for (int i = 0; i < vector.Length; i++)
        {
            list.Add(vector[i]);
        }
    }

    private Vector<T> CreateVectorFromList(List<T> list)
    {
        var vector = new Vector<T>(list.Count);
        for (int i = 0; i < list.Count; i++)
        {
            vector[i] = list[i];
        }
        return vector;
    }

    private int SetComponentParameters(ILayer<T> layer, Vector<T> parameters, int offset)
    {
        int count = layer.ParameterCount;
        var componentParams = new Vector<T>(count);
        for (int i = 0; i < count; i++)
        {
            componentParams[i] = parameters[offset + i];
        }
        layer.SetParameters(componentParams);
        return offset + count;
    }

    private int SetComponentParameters(GhostBatchNormalization<T> bn, Vector<T> parameters, int offset)
    {
        int count = bn.ParameterCount;
        var componentParams = new Vector<T>(count);
        for (int i = 0; i < count; i++)
        {
            componentParams[i] = parameters[offset + i];
        }
        bn.SetParameters(componentParams);
        return offset + count;
    }

    private int SetComponentParameters(FeatureTransformer<T> ft, Vector<T> parameters, int offset)
    {
        int count = ft.ParameterCount;
        var componentParams = new Vector<T>(count);
        for (int i = 0; i < count; i++)
        {
            componentParams[i] = parameters[offset + i];
        }
        ft.SetParameters(componentParams);
        return offset + count;
    }

    private int SetComponentParameters(AttentiveTransformer<T> at, Vector<T> parameters, int offset)
    {
        int count = at.ParameterCount;
        var componentParams = new Vector<T>(count);
        for (int i = 0; i < count; i++)
        {
            componentParams[i] = parameters[offset + i];
        }
        at.SetParameters(componentParams);
        return offset + count;
    }

    /// <summary>
    /// Gets the total number of trainable parameters.
    /// </summary>
    public int ParameterCount
    {
        get
        {
            int count = _initialBN.ParameterCount;
            count += _featureTransformers.Sum(ft => ft.ParameterCount);
            count += _attentiveTransformers.Sum(at => at.ParameterCount);
            count += _outputLayer.ParameterCount;
            return count;
        }
    }

    /// <summary>
    /// Updates parameters using the specified learning rate.
    /// </summary>
    public void UpdateParameters(T learningRate)
    {
        foreach (var ft in _featureTransformers)
        {
            ft.UpdateParameters(learningRate);
        }
        foreach (var at in _attentiveTransformers)
        {
            at.UpdateParameters(learningRate);
        }
        _outputLayer.UpdateParameters(learningRate);
    }

    /// <summary>
    /// Sets training mode.
    /// </summary>
    public void SetTrainingMode(bool isTraining)
    {
        _isTraining = isTraining;
        foreach (var ft in _featureTransformers)
        {
            ft.SetTrainingMode(isTraining);
        }
        foreach (var at in _attentiveTransformers)
        {
            at.SetTrainingMode(isTraining);
        }
        _outputLayer.SetTrainingMode(isTraining);
    }

    /// <summary>
    /// Resets the internal state.
    /// </summary>
    public void ResetState()
    {
        _stepAttentionMasks.Clear();
        _stepOutputs.Clear();
        foreach (var ft in _featureTransformers)
        {
            ft.ResetState();
        }
        foreach (var at in _attentiveTransformers)
        {
            at.ResetState();
        }
        _outputLayer.ResetState();
    }

    #endregion
}
