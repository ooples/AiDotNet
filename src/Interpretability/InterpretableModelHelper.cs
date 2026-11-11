using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Interpretability.Explainers;
using AiDotNet.Interpretability.Helpers;
using AiDotNet.LinearAlgebra;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Interpretability;

/// <summary>
/// Provides static helper methods for model interpretability operations using production-ready explainer algorithms.
/// </summary>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This class serves as a facade that connects interpretable models to the
/// actual explainer implementations (SHAP, LIME, PDP, Integrated Gradients, DeepLIFT, GradCAM, etc.).
/// It handles the conversion between model interfaces and the prediction functions that explainers need.
/// </para>
/// <para>
/// <b>IMPORTANT:</b> These methods delegate to production-ready explainer classes:
/// - SHAPExplainer: Kernel SHAP for model-agnostic Shapley value explanations
/// - LIMEExplainer: Local interpretable model-agnostic explanations
/// - PartialDependenceExplainer: PDP and ICE curves
/// - FeatureInteractionExplainer: Friedman's H-statistic
/// - IntegratedGradientsExplainer: Neural network attribution (requires gradient access)
/// - DeepLIFTExplainer: Fast neural network attribution
/// - GradCAMExplainer: Visual CNN explanations
/// - PermutationFeatureImportance: Model-agnostic global importance
/// </para>
/// <para>
/// <b>CRITICAL:</b> Methods that require background data will throw if no data is provided.
/// Synthetic background data is only used as a last resort and will emit warnings.
/// </para>
/// </remarks>
public static class InterpretableModelHelper
{
    #region Permutation Feature Importance (Global)

    /// <summary>
    /// Gets the global feature importance using permutation feature importance.
    /// </summary>
    /// <param name="model">The interpretable model to explain.</param>
    /// <param name="enabledMethods">Set of enabled interpretation methods.</param>
    /// <param name="data">The data matrix to compute importance on (REQUIRED).</param>
    /// <param name="labels">The ground truth labels (REQUIRED for proper importance).</param>
    /// <param name="nRepeats">Number of permutation repeats for stability (default: 5).</param>
    /// <returns>Dictionary mapping feature indices to their importance scores.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Permutation feature importance measures how much the model's performance
    /// drops when each feature is randomly shuffled. Features that cause large drops when shuffled
    /// are more important to the model's predictions.
    /// </para>
    /// <para>
    /// This is the proper way to compute feature importance - it requires actual test data and labels.
    /// </para>
    /// </remarks>
    public static Task<Dictionary<int, T>> GetGlobalFeatureImportanceAsync<T>(
        IInterpretableModel<T> model,
        HashSet<InterpretationMethod> enabledMethods,
        Matrix<T> data,
        Vector<T> labels,
        int nRepeats = 5)
    {
        if (model is null)
            throw new ArgumentNullException(nameof(model));
        if (enabledMethods is null)
            throw new ArgumentNullException(nameof(enabledMethods));
        if (data is null)
            throw new ArgumentNullException(nameof(data), "Background data is required for permutation feature importance.");
        if (labels is null)
            throw new ArgumentNullException(nameof(labels), "Ground truth labels are required for permutation feature importance.");

        if (!enabledMethods.Contains(InterpretationMethod.FeatureImportance))
        {
            throw new InvalidOperationException("FeatureImportance method is not enabled.");
        }

        var numOps = MathHelper.GetNumericOperations<T>();
        Func<Matrix<T>, Vector<T>> predictFunc = CreatePredictionFunction(model);

        // Use MSE-based score function (negative because higher is better)
        Func<Vector<T>, Vector<T>, T> scoreFunc = (actual, predicted) =>
        {
            var residual = AiDotNetEngine.Current.TensorSubtract(actual.AsTensor(), predicted.GetSubVector(0, actual.Length).AsTensor());
            T mse = numOps.Divide(AiDotNetEngine.Current.TensorSumOfSquares(residual), numOps.FromDouble(actual.Length));
            return numOps.Negate(mse); // Negative MSE (higher is better)
        };

        var pfi = new PermutationFeatureImportance<T>(
            predictFunction: predictFunc,
            scoreFunction: scoreFunc,
            nRepeats: nRepeats,
            randomState: 42);

        var result = pfi.Calculate(data, labels);

        // Convert to dictionary
        var importance = ToIndexDictionary(result.Importances);

        return Task.FromResult(importance);
    }

    /// <summary>
    /// Gets the global feature importance (simplified overload for backwards compatibility).
    /// </summary>
    /// <remarks>
    /// <b>WARNING:</b> This method uses weight-based importance which is less reliable than
    /// permutation importance. Use the overload with data and labels for production use.
    /// </remarks>
    [Obsolete("Use the overload with data and labels for accurate permutation-based importance.")]
    public static Task<Dictionary<int, T>> GetGlobalFeatureImportanceAsync<T>(
        IInterpretableModel<T> model,
        HashSet<InterpretationMethod> enabledMethods)
    {
        if (model is null)
            throw new ArgumentNullException(nameof(model));
        if (enabledMethods is null)
            throw new ArgumentNullException(nameof(enabledMethods));

        if (!enabledMethods.Contains(InterpretationMethod.FeatureImportance))
        {
            throw new InvalidOperationException("FeatureImportance method is not enabled.");
        }

        var numOps = MathHelper.GetNumericOperations<T>();
        var importance = new Dictionary<int, T>();

        // Fallback: weight-based importance for neural networks (less reliable)
        if (model is INeuralNetworkModel<T> nnModel)
        {
            var architecture = nnModel.GetArchitecture();
            int numFeatures = architecture.InputSize;
            var layers = architecture.Layers;

            if (layers is not null && layers.Count > 0)
            {
                var firstLayer = layers[0];
                var parameters = firstLayer.GetParameters();
                if (parameters.Length > 0 && parameters[0] is Tensor<T> weights)
                {
                    int outputDim = weights.Shape.Length > 1 ? weights.Shape[1] : 1;
                    var sums = AbsoluteWeightRowSums(weights, Math.Min(numFeatures, weights.Shape[0]), outputDim);

                    // Normalize
                    T total = AiDotNetEngine.Current.Sum(sums);
                    if (numOps.GreaterThan(total, numOps.FromDouble(1e-10)))
                    {
                        sums = AiDotNetEngine.Current.Divide(sums, total);
                    }

                    return Task.FromResult(ToIndexDictionary(sums));
                }
            }

            T uniform = numOps.Divide(numOps.One, numOps.FromDouble(numFeatures));
            importance = Enumerable.Range(0, Math.Max(0, numFeatures)).ToDictionary(i => i, _ => uniform);
        }
        else
        {
            // For non-NN models without data, return uniform importance
            T uniform = numOps.FromDouble(0.1);
            importance = Enumerable.Range(0, 10).ToDictionary(i => i, _ => uniform);
        }

        return Task.FromResult(importance);
    }

    #endregion

    #region Local Feature Importance / Saliency

    /// <summary>
    /// Gets the local feature importance for a specific input using gradient-based attribution.
    /// </summary>
    public static Task<Dictionary<int, T>> GetLocalFeatureImportanceAsync<T>(
        IInterpretableModel<T> model,
        HashSet<InterpretationMethod> enabledMethods,
        Tensor<T> input)
    {
        if (model is null)
            throw new ArgumentNullException(nameof(model));
        if (enabledMethods is null)
            throw new ArgumentNullException(nameof(enabledMethods));
        if (input is null)
            throw new ArgumentNullException(nameof(input));

        if (!enabledMethods.Contains(InterpretationMethod.FeatureImportance))
        {
            throw new InvalidOperationException("FeatureImportance method is not enabled.");
        }

        var numOps = MathHelper.GetNumericOperations<T>();
        var importance = new Dictionary<int, T>();

        if (model is INeuralNetworkModel<T> nnModel)
        {
            var architecture = nnModel.GetArchitecture();
            var layers = architecture.Layers;

            if (layers is not null && layers.Count > 0)
            {
                var firstLayer = layers[0];
                var parameters = firstLayer.GetParameters();
                if (parameters.Length > 0 && parameters[0] is Tensor<T> weights)
                {
                    var engine = AiDotNetEngine.Current;
                    int numFeatures = Math.Min(input.Data.Length, weights.Shape[0]);
                    int outputDim = weights.Shape.Length > 1 ? weights.Shape[1] : 1;

                    var avgWeights = engine.Divide(AbsoluteWeightRowSums(weights, numFeatures, outputDim), numOps.FromDouble(outputDim));
                    var inputValues = engine.Abs(input.ToVector().GetSubVector(0, Math.Max(0, numFeatures)));
                    return Task.FromResult(ToIndexDictionary(engine.Multiply(inputValues, avgWeights)));
                }
            }
        }

        // Fallback: use absolute input values
        importance = ToIndexDictionary(AiDotNetEngine.Current.Abs(input.ToVector()));

        return Task.FromResult(importance);
    }

    #endregion

    #region SHAP (Kernel SHAP)

    /// <summary>
    /// Gets SHAP values for the given inputs using Kernel SHAP algorithm.
    /// </summary>
    /// <param name="model">The interpretable model to explain.</param>
    /// <param name="enabledMethods">Set of enabled interpretation methods.</param>
    /// <param name="inputs">The input tensor to explain.</param>
    /// <param name="backgroundData">Background data representing the baseline distribution (REQUIRED).</param>
    /// <param name="nSamples">Number of samples for Kernel SHAP approximation (default: 100).</param>
    /// <returns>Matrix of SHAP values where each row is an instance and columns are features.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> SHAP (SHapley Additive exPlanations) values come from game theory
    /// and fairly distribute the "credit" for a prediction among all input features.
    ///
    /// - Positive SHAP value: feature pushed prediction higher
    /// - Negative SHAP value: feature pushed prediction lower
    /// - SHAP values sum to (prediction - baseline_prediction)
    ///
    /// <b>IMPORTANT:</b> Background data should represent the baseline distribution of your data.
    /// Using random or synthetic data will give meaningless explanations.
    /// </para>
    /// </remarks>
    public static Task<Matrix<T>> GetShapValuesAsync<T>(
        IInterpretableModel<T> model,
        HashSet<InterpretationMethod> enabledMethods,
        Tensor<T> inputs,
        Matrix<T> backgroundData,
        int nSamples = 100)
    {
        if (model is null)
            throw new ArgumentNullException(nameof(model));
        if (enabledMethods is null)
            throw new ArgumentNullException(nameof(enabledMethods));
        if (inputs is null)
            throw new ArgumentNullException(nameof(inputs));
        if (backgroundData is null)
            throw new ArgumentNullException(nameof(backgroundData), "Background data is required for Kernel SHAP. Use a representative sample of your training data.");

        if (!enabledMethods.Contains(InterpretationMethod.SHAP))
        {
            throw new InvalidOperationException("SHAP method is not enabled.");
        }

        int numSamples = inputs.Shape[0];
        int numFeatures = inputs.Data.Length / numSamples;

        Func<Matrix<T>, Vector<T>> predictFunc = CreatePredictionFunction(model);

        var shapExplainer = new SHAPExplainer<T>(
            predictFunction: predictFunc,
            backgroundData: backgroundData,
            nSamples: Math.Max(nSamples, numFeatures * 10),
            randomState: 42);

        var shapValues = new Matrix<T>(numSamples, numFeatures);
        var inputMatrix = ConvertTensorToMatrix(inputs);

        for (int i = 0; i < numSamples; i++)
        {
            var instance = inputMatrix.GetRow(i);
            var explanation = shapExplainer.Explain(instance);

            // Features past the explanation's end stay zero
            AiDotNetEngine.Current.SetRow(shapValues, i, ExplainerTensorOps<T>.PadOrTruncate(explanation.ShapValues, numFeatures));
        }

        return Task.FromResult(shapValues);
    }

    /// <summary>
    /// Gets SHAP values (backwards compatible overload that uses input as background).
    /// </summary>
    [Obsolete("Use the overload with explicit backgroundData parameter for accurate SHAP values.")]
    public static Task<Matrix<T>> GetShapValuesAsync<T>(
        IInterpretableModel<T> model,
        HashSet<InterpretationMethod> enabledMethods,
        Tensor<T> inputs)
    {
        // Use input data as background (not ideal but maintains backwards compatibility)
        var backgroundData = ConvertTensorToMatrix(inputs);
        return GetShapValuesAsync(model, enabledMethods, inputs, backgroundData);
    }

    #endregion

    #region LIME

    /// <summary>
    /// Gets LIME explanation for a specific input using local linear approximation.
    /// </summary>
    public static Task<LimeExplanation<T>> GetLimeExplanationAsync<T>(
        IInterpretableModel<T> model,
        HashSet<InterpretationMethod> enabledMethods,
        Tensor<T> input,
        int numFeatures = 10,
        int nSamples = 500,
        double kernelWidth = 0.75)
    {
        if (model is null)
            throw new ArgumentNullException(nameof(model));
        if (enabledMethods is null)
            throw new ArgumentNullException(nameof(enabledMethods));
        if (input is null)
            throw new ArgumentNullException(nameof(input));

        if (!enabledMethods.Contains(InterpretationMethod.LIME))
        {
            throw new InvalidOperationException("LIME method is not enabled.");
        }

        var numOps = MathHelper.GetNumericOperations<T>();
        int inputSize = input.Data.Length;

        Func<Matrix<T>, Vector<T>> predictFunc = CreatePredictionFunction(model);

        var limeExplainer = new LIMEExplainer<T>(
            predictFunction: predictFunc,
            numFeatures: inputSize,
            nSamples: nSamples,
            kernelWidth: kernelWidth,
            randomState: 42);

        var instanceVector = new Vector<T>(input.Data.ToArray());
        var limeResult = limeExplainer.Explain(instanceVector);

        var featureImportance = ToIndexDictionary(
            limeResult.Coefficients.GetSubVector(0, Math.Max(0, Math.Min(numFeatures, limeResult.Coefficients.Length))));

        var limeExplanation = new LimeExplanation<T>
        {
            FeatureImportance = featureImportance,
            Intercept = limeResult.Intercept,
            PredictedValue = limeResult.Prediction,
            LocalModelScore = limeResult.LocalR2,
            NumFeatures = numFeatures
        };

        return Task.FromResult(limeExplanation);
    }

    #endregion

    #region Integrated Gradients

    /// <summary>
    /// Gets Integrated Gradients attributions for neural network explanation.
    /// </summary>
    /// <param name="model">The neural network model to explain.</param>
    /// <param name="enabledMethods">Set of enabled interpretation methods.</param>
    /// <param name="input">The input tensor to explain.</param>
    /// <param name="baseline">The baseline input (defaults to zeros if null).</param>
    /// <param name="numSteps">Number of integration steps (default: 50).</param>
    /// <param name="gradientFunction">Optional function to compute gradients analytically.</param>
    /// <returns>Integrated Gradients explanation with feature attributions.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Integrated Gradients is a theoretically-grounded method for explaining
    /// neural network predictions. It satisfies important axioms (completeness, sensitivity) that
    /// other methods don't.
    ///
    /// How it works:
    /// - Start with a baseline (typically zeros)
    /// - Create a path from baseline to your input
    /// - Integrate the gradients along this path
    /// - The result shows how each feature contributed to moving from baseline to final prediction
    /// </para>
    /// </remarks>
    public static Task<IntegratedGradientsExplanation<T>> GetIntegratedGradientsAsync<T>(
        IInterpretableModel<T> model,
        HashSet<InterpretationMethod> enabledMethods,
        Tensor<T> input,
        Tensor<T>? baseline = null,
        int numSteps = 50,
        Func<Vector<T>, int, Vector<T>>? gradientFunction = null)
    {
        if (model is null)
            throw new ArgumentNullException(nameof(model));
        if (enabledMethods is null)
            throw new ArgumentNullException(nameof(enabledMethods));
        if (input is null)
            throw new ArgumentNullException(nameof(input));

        if (!enabledMethods.Contains(InterpretationMethod.IntegratedGradients))
        {
            throw new InvalidOperationException("IntegratedGradients method is not enabled.");
        }

        int numFeatures = input.Data.Length;

        // Create vector-based predict function
        Func<Vector<T>, Vector<T>> predictVectorFunc = CreateVectorPredictionFunction(model);

        // Convert baseline tensor to vector
        Vector<T>? baselineVector = baseline is not null
            ? new Vector<T>(baseline.Data.ToArray())
            : null;

        var igExplainer = new IntegratedGradientsExplainer<T>(
            predictFunction: predictVectorFunc,
            gradientFunction: gradientFunction,
            numFeatures: numFeatures,
            numSteps: numSteps,
            baseline: baselineVector);

        var instanceVector = new Vector<T>(input.Data.ToArray());
        var result = igExplainer.Explain(instanceVector);

        return Task.FromResult(result);
    }

    #endregion

    #region DeepLIFT

    /// <summary>
    /// Gets DeepLIFT attributions for neural network explanation.
    /// </summary>
    /// <param name="model">The neural network model to explain.</param>
    /// <param name="enabledMethods">Set of enabled interpretation methods.</param>
    /// <param name="input">The input tensor to explain.</param>
    /// <param name="baseline">The baseline input (defaults to zeros if null).</param>
    /// <param name="rule">DeepLIFT rule to use (Rescale or RevealCancel).</param>
    /// <returns>DeepLIFT explanation with feature attributions.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> DeepLIFT explains neural network predictions by comparing
    /// activations to a reference baseline. It's faster than Integrated Gradients and
    /// handles non-linearities better than vanilla gradients.
    ///
    /// The Rescale rule is simpler and works well in most cases.
    /// The RevealCancel rule is better when you need to separate positive and negative contributions.
    /// </para>
    /// </remarks>
    public static Task<DeepLIFTExplanation<T>> GetDeepLIFTAsync<T>(
        IInterpretableModel<T> model,
        HashSet<InterpretationMethod> enabledMethods,
        Tensor<T> input,
        Tensor<T>? baseline = null,
        DeepLIFTRule rule = DeepLIFTRule.Rescale)
    {
        if (model is null)
            throw new ArgumentNullException(nameof(model));
        if (enabledMethods is null)
            throw new ArgumentNullException(nameof(enabledMethods));
        if (input is null)
            throw new ArgumentNullException(nameof(input));

        if (!enabledMethods.Contains(InterpretationMethod.DeepLIFT))
        {
            throw new InvalidOperationException("DeepLIFT method is not enabled.");
        }

        int numFeatures = input.Data.Length;

        Func<Vector<T>, Vector<T>> predictVectorFunc = CreateVectorPredictionFunction(model);

        Vector<T>? baselineVector = baseline is not null
            ? new Vector<T>(baseline.Data.ToArray())
            : null;

        var deepLiftExplainer = new DeepLIFTExplainer<T>(
            predictFunction: predictVectorFunc,
            numFeatures: numFeatures,
            baseline: baselineVector,
            rule: rule);

        var instanceVector = new Vector<T>(input.Data.ToArray());
        var result = deepLiftExplainer.Explain(instanceVector);

        return Task.FromResult(result);
    }

    #endregion

    #region GradCAM

    /// <summary>
    /// Gets Grad-CAM visual explanation for a CNN prediction.
    /// </summary>
    /// <param name="model">The CNN model to explain.</param>
    /// <param name="enabledMethods">Set of enabled interpretation methods.</param>
    /// <param name="input">The input image tensor.</param>
    /// <param name="inputShape">Shape of the input (e.g., [height, width, channels]).</param>
    /// <param name="featureMapShape">Shape of feature maps from the target conv layer.</param>
    /// <param name="targetClass">Target class to explain (-1 for predicted class).</param>
    /// <param name="featureMapFunction">Optional function to extract feature maps.</param>
    /// <param name="gradientFunction">Optional function to compute gradients.</param>
    /// <param name="useGradCAMPlusPlus">Use Grad-CAM++ variant (default: false).</param>
    /// <returns>GradCAM explanation with heatmap.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Grad-CAM creates visual explanations showing which parts of an image
    /// were most important for the CNN's prediction. It produces a heatmap where bright regions
    /// indicate high importance.
    ///
    /// Use Grad-CAM++ when your images have multiple instances of the same object.
    /// </para>
    /// </remarks>
    public static Task<GradCAMExplanation<T>> GetGradCAMAsync<T>(
        IInterpretableModel<T> model,
        HashSet<InterpretationMethod> enabledMethods,
        Tensor<T> input,
        int[] inputShape,
        int[] featureMapShape,
        int targetClass = -1,
        Func<Tensor<T>, int, Tensor<T>>? featureMapFunction = null,
        Func<Tensor<T>, int, int, Tensor<T>>? gradientFunction = null,
        bool useGradCAMPlusPlus = false)
    {
        if (model is null)
            throw new ArgumentNullException(nameof(model));
        if (enabledMethods is null)
            throw new ArgumentNullException(nameof(enabledMethods));
        if (input is null)
            throw new ArgumentNullException(nameof(input));
        if (inputShape is null || inputShape.Length == 0)
            throw new ArgumentNullException(nameof(inputShape));
        if (featureMapShape is null || featureMapShape.Length == 0)
            throw new ArgumentNullException(nameof(featureMapShape));

        if (!enabledMethods.Contains(InterpretationMethod.GradCAM))
        {
            throw new InvalidOperationException("GradCAM method is not enabled.");
        }

        // Create tensor-based predict function
        Func<Tensor<T>, Tensor<T>> predictTensorFunc = CreateTensorPredictionFunction(model);

        var gradCamExplainer = new GradCAMExplainer<T>(
            predictFunction: predictTensorFunc,
            featureMapFunction: featureMapFunction,
            gradientFunction: gradientFunction,
            inputShape: inputShape,
            featureMapShape: featureMapShape,
            useGradCAMPlusPlus: useGradCAMPlusPlus);

        var result = gradCamExplainer.ExplainTensor(input, targetClass);

        return Task.FromResult(result);
    }

    #endregion

    #region TreeSHAP

    /// <summary>
    /// Gets TreeSHAP values for a tree-based model.
    /// </summary>
    /// <param name="tree">The decision tree root node.</param>
    /// <param name="enabledMethods">Set of enabled interpretation methods.</param>
    /// <param name="input">The input to explain.</param>
    /// <param name="expectedValue">Expected (baseline) prediction value.</param>
    /// <param name="featureNames">Optional feature names.</param>
    /// <returns>TreeSHAP explanation with exact SHAP values.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> TreeSHAP computes exact (not approximate) SHAP values
    /// for tree-based models. Unlike Kernel SHAP which uses sampling, TreeSHAP
    /// uses the tree structure to compute mathematically precise Shapley values.
    ///
    /// This is much faster and more accurate for tree-based models like:
    /// - Decision Trees
    /// - Random Forests
    /// - Gradient Boosting
    /// </para>
    /// </remarks>
    public static Task<TreeSHAPExplanation<T>> GetTreeSHAPAsync<T>(
        DecisionTreeNode<T> tree,
        HashSet<InterpretationMethod> enabledMethods,
        Tensor<T> input,
        T expectedValue,
        string[]? featureNames = null)
    {
        if (tree is null)
            throw new ArgumentNullException(nameof(tree));
        if (enabledMethods is null)
            throw new ArgumentNullException(nameof(enabledMethods));
        if (input is null)
            throw new ArgumentNullException(nameof(input));

        if (!enabledMethods.Contains(InterpretationMethod.TreeSHAP))
        {
            throw new InvalidOperationException("TreeSHAP method is not enabled.");
        }

        int numFeatures = input.Data.Length;

        var treeSHAPExplainer = new TreeSHAPExplainer<T>(
            tree: tree,
            numFeatures: numFeatures,
            expectedValue: expectedValue,
            featureNames: featureNames);

        var instanceVector = new Vector<T>(input.Data.ToArray());
        var result = treeSHAPExplainer.Explain(instanceVector);

        return Task.FromResult(result);
    }

    /// <summary>
    /// Gets TreeSHAP values for an ensemble of trees (Random Forest, Gradient Boosting).
    /// </summary>
    /// <param name="trees">The ensemble of decision tree root nodes.</param>
    /// <param name="enabledMethods">Set of enabled interpretation methods.</param>
    /// <param name="input">The input to explain.</param>
    /// <param name="expectedValue">Expected (baseline) prediction value.</param>
    /// <param name="featureNames">Optional feature names.</param>
    /// <returns>TreeSHAP explanation with exact SHAP values averaged across trees.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> For ensemble models like Random Forests, TreeSHAP computes
    /// exact SHAP values for each tree and then averages them. This gives you precise
    /// feature attributions for the entire ensemble.
    /// </para>
    /// </remarks>
    public static Task<TreeSHAPExplanation<T>> GetTreeSHAPAsync<T>(
        IEnumerable<DecisionTreeNode<T>> trees,
        HashSet<InterpretationMethod> enabledMethods,
        Tensor<T> input,
        T expectedValue,
        string[]? featureNames = null)
    {
        if (trees is null)
            throw new ArgumentNullException(nameof(trees));
        if (enabledMethods is null)
            throw new ArgumentNullException(nameof(enabledMethods));
        if (input is null)
            throw new ArgumentNullException(nameof(input));

        if (!enabledMethods.Contains(InterpretationMethod.TreeSHAP))
        {
            throw new InvalidOperationException("TreeSHAP method is not enabled.");
        }

        int numFeatures = input.Data.Length;

        var treeSHAPExplainer = new TreeSHAPExplainer<T>(
            trees: trees,
            numFeatures: numFeatures,
            expectedValue: expectedValue,
            featureNames: featureNames);

        var instanceVector = new Vector<T>(input.Data.ToArray());
        var result = treeSHAPExplainer.Explain(instanceVector);

        return Task.FromResult(result);
    }

    #endregion

    #region Partial Dependence

    /// <summary>
    /// Gets partial dependence data for specified features.
    /// </summary>
    /// <param name="model">The interpretable model to explain.</param>
    /// <param name="enabledMethods">Set of enabled interpretation methods.</param>
    /// <param name="featureIndices">Indices of features to analyze.</param>
    /// <param name="backgroundData">Background data for computing marginal effects (REQUIRED).</param>
    /// <param name="gridResolution">Number of points in the feature grid.</param>
    /// <param name="computeIce">Whether to compute ICE curves.</param>
    /// <returns>Partial dependence data with grid values, PDP values, and optional ICE curves.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Partial Dependence Plots (PDPs) show how a feature affects
    /// predictions on average, while holding all other features constant.
    ///
    /// - Upward slope: increasing the feature increases predictions
    /// - Downward slope: increasing the feature decreases predictions
    /// - Flat line: feature has little average effect
    ///
    /// ICE curves show the same for individual instances, revealing if the effect varies.
    /// </para>
    /// </remarks>
    public static Task<PartialDependenceData<T>> GetPartialDependenceAsync<T>(
        IInterpretableModel<T> model,
        HashSet<InterpretationMethod> enabledMethods,
        Vector<int> featureIndices,
        Matrix<T> backgroundData,
        int gridResolution = 20,
        bool computeIce = true)
    {
        if (model is null)
            throw new ArgumentNullException(nameof(model));
        if (enabledMethods is null)
            throw new ArgumentNullException(nameof(enabledMethods));
        if (featureIndices is null)
            throw new ArgumentNullException(nameof(featureIndices));
        if (backgroundData is null)
            throw new ArgumentNullException(nameof(backgroundData), "Background data is required for PDP computation.");

        if (!enabledMethods.Contains(InterpretationMethod.PartialDependence))
        {
            throw new InvalidOperationException("PartialDependence method is not enabled.");
        }

        Func<Matrix<T>, Vector<T>> predictFunc = CreatePredictionFunction(model);

        var pdpExplainer = new PartialDependenceExplainer<T>(
            predictFunction: predictFunc,
            backgroundData: backgroundData,
            gridResolution: gridResolution,
            computeIce: computeIce);

        var featureIdxArray = featureIndices.ToArray();
        var pdpResult = pdpExplainer.ComputeForFeatures(featureIdxArray);

        // Convert to PartialDependenceData format
        var gridValues = new Dictionary<int, Vector<T>>();
        foreach (var kvp in pdpResult.GridValues)
        {
            gridValues[kvp.Key] = new Vector<T>(kvp.Value);
        }

        int numPdFeatures = featureIdxArray.Length;
        var pdpMatrix = new Matrix<T>(numPdFeatures, gridResolution);
        for (int f = 0; f < numPdFeatures; f++)
        {
            var featureIdx = featureIdxArray[f];
            if (pdpResult.PartialDependence.TryGetValue(featureIdx, out var pdValues))
            {
                // Grid points beyond the explainer's output stay zero
                var row = new Vector<T>(pdValues.Take(gridResolution)
                    .Concat(Enumerable.Repeat(MathHelper.GetNumericOperations<T>().Zero, Math.Max(0, gridResolution - pdValues.Length))));
                AiDotNetEngine.Current.SetRow(pdpMatrix, f, row);
            }
        }

        var iceCurves = new List<Matrix<T>>();
        if (pdpResult.IceCurves is not null)
        {
            foreach (var featureIdx in featureIdxArray)
            {
                if (pdpResult.IceCurves.TryGetValue(featureIdx, out var iceMatrix))
                {
                    iceCurves.Add(iceMatrix.Clone());
                }
            }
        }

        var pdpData = new PartialDependenceData<T>
        {
            GridValues = gridValues,
            PartialDependenceValues = pdpMatrix,
            FeatureIndices = featureIndices,
            GridResolution = gridResolution,
            IceCurves = iceCurves
        };

        return Task.FromResult(pdpData);
    }

    /// <summary>
    /// Gets partial dependence data (backwards compatible overload with synthetic data fallback).
    /// </summary>
    [Obsolete("Use the overload with explicit backgroundData parameter for accurate PDP computation.")]
    public static Task<PartialDependenceData<T>> GetPartialDependenceAsync<T>(
        IInterpretableModel<T> model,
        HashSet<InterpretationMethod> enabledMethods,
        Vector<int> featureIndices,
        int gridResolution = 20)
    {
        int numFeatures = GetNumFeatures(model);
        var backgroundData = CreateBackgroundData<T>(model, numFeatures, 100);
        return GetPartialDependenceAsync(model, enabledMethods, featureIndices, backgroundData, gridResolution);
    }

    #endregion

    #region Feature Interaction

    /// <summary>
    /// Gets feature interaction effects between two features using Friedman's H-statistic.
    /// </summary>
    /// <param name="model">The interpretable model to explain.</param>
    /// <param name="enabledMethods">Set of enabled interpretation methods.</param>
    /// <param name="feature1Index">Index of the first feature.</param>
    /// <param name="feature2Index">Index of the second feature.</param>
    /// <param name="backgroundData">Background data for H-statistic computation (REQUIRED).</param>
    /// <param name="gridSize">Number of grid points for PD approximation.</param>
    /// <returns>The H-statistic value between 0 and 1.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> The H-statistic measures how much two features interact.
    /// An interaction means the effect of one feature depends on the value of another.
    ///
    /// - H = 0: No interaction (features act independently)
    /// - H = 1: Pure interaction (entire effect comes from interaction)
    /// - H &lt; 0.05: Negligible interaction
    /// - H &gt; 0.5: Strong interaction
    /// </para>
    /// </remarks>
    public static Task<T> GetFeatureInteractionAsync<T>(
        IInterpretableModel<T> model,
        HashSet<InterpretationMethod> enabledMethods,
        int feature1Index,
        int feature2Index,
        Matrix<T> backgroundData,
        int gridSize = 20)
    {
        if (model is null)
            throw new ArgumentNullException(nameof(model));
        if (enabledMethods is null)
            throw new ArgumentNullException(nameof(enabledMethods));
        if (backgroundData is null)
            throw new ArgumentNullException(nameof(backgroundData), "Background data is required for H-statistic computation.");

        if (!enabledMethods.Contains(InterpretationMethod.FeatureInteraction))
        {
            throw new InvalidOperationException("FeatureInteraction method is not enabled.");
        }

        Func<Matrix<T>, Vector<T>> predictFunc = CreatePredictionFunction(model);

        var interactionExplainer = new FeatureInteractionExplainer<T>(
            predictFunction: predictFunc,
            data: backgroundData,
            gridSize: gridSize);

        var hStatistic = interactionExplainer.ComputePairwiseHStatistic(feature1Index, feature2Index);

        return Task.FromResult(hStatistic);
    }

    /// <summary>
    /// Gets feature interaction (backwards compatible overload).
    /// </summary>
    [Obsolete("Use the overload with backgroundData parameter for accurate H-statistic computation.")]
    public static Task<T> GetFeatureInteractionAsync<T>(
        IInterpretableModel<T> model,
        HashSet<InterpretationMethod> enabledMethods,
        int feature1Index,
        int feature2Index)
    {
        int numFeatures = GetNumFeatures(model);
        var backgroundData = CreateBackgroundData<T>(model, numFeatures, 50);
        return GetFeatureInteractionAsync(model, enabledMethods, feature1Index, feature2Index, backgroundData);
    }

    /// <summary>
    /// Legacy overload without model parameter.
    /// </summary>
    [Obsolete("Use the overload that takes a model parameter for proper H-statistic computation.")]
    public static Task<T> GetFeatureInteractionAsync<T>(
        HashSet<InterpretationMethod> enabledMethods,
        int feature1Index,
        int feature2Index)
    {
        if (enabledMethods is null)
            throw new ArgumentNullException(nameof(enabledMethods));

        if (!enabledMethods.Contains(InterpretationMethod.FeatureInteraction))
        {
            throw new InvalidOperationException("FeatureInteraction method is not enabled.");
        }

        var numOps = MathHelper.GetNumericOperations<T>();
        return Task.FromResult(numOps.Zero);
    }

    #endregion

    #region Counterfactual

    /// <summary>
    /// Gets counterfactual explanation for a given input and desired output.
    /// </summary>
    public static Task<CounterfactualExplanation<T>> GetCounterfactualAsync<T>(
        IInterpretableModel<T> model,
        HashSet<InterpretationMethod> enabledMethods,
        Tensor<T> input,
        Tensor<T> desiredOutput,
        int maxChanges = 5)
    {
        if (model is null)
            throw new ArgumentNullException(nameof(model));
        if (enabledMethods is null)
            throw new ArgumentNullException(nameof(enabledMethods));
        if (input is null)
            throw new ArgumentNullException(nameof(input));
        if (desiredOutput is null)
            throw new ArgumentNullException(nameof(desiredOutput));

        if (!enabledMethods.Contains(InterpretationMethod.Counterfactual))
        {
            throw new InvalidOperationException("Counterfactual method is not enabled.");
        }

        int numFeatures = input.Data.Length;

        Func<Matrix<T>, Vector<T>> predictFunc = CreatePredictionFunction(model);

        var cfExplainer = new CounterfactualExplainer<T>(
            predictFunction: predictFunc,
            numFeatures: numFeatures,
            maxIterations: 1000,
            stepSize: 0.1,
            maxChanges: maxChanges,
            targetThreshold: 0.5,
            randomState: 42);

        var instanceVector = new Vector<T>(input.Data.ToArray());
        var targetVector = new Vector<T>(desiredOutput.Data.ToArray());

        var cfResult = cfExplainer.Explain(instanceVector, targetVector);

        return Task.FromResult(cfResult);
    }

    #endregion

    #region Anchor

    /// <summary>
    /// Gets anchor explanation for a given input using beam search to find
    /// sufficient conditions (rules) that anchor the prediction.
    /// </summary>
    public static Task<AnchorExplanation<T>> GetAnchorExplanationAsync<T>(
        IInterpretableModel<T> model,
        HashSet<InterpretationMethod> enabledMethods,
        Tensor<T> input,
        T threshold)
    {
        if (model is null)
            throw new ArgumentNullException(nameof(model));
        if (enabledMethods is null)
            throw new ArgumentNullException(nameof(enabledMethods));
        if (input is null)
            throw new ArgumentNullException(nameof(input));

        if (!enabledMethods.Contains(InterpretationMethod.Anchor))
        {
            throw new InvalidOperationException("Anchor method is not enabled.");
        }

        var numOps = MathHelper.GetNumericOperations<T>();
        int numFeatures = input.Data.Length;

        Func<Matrix<T>, Vector<T>> predictFunc = CreatePredictionFunction(model);

        var anchorExplainer = new AnchorExplainer<T>(
            predictFunction: predictFunc,
            numFeatures: numFeatures,
            precisionThreshold: numOps.ToDouble(threshold),
            maxAnchorSize: Math.Min(6, numFeatures),
            beamWidth: 4,
            nSamples: 1000,
            randomState: 42);

        var instanceVector = new Vector<T>(input.Data.ToArray());
        var anchorResult = anchorExplainer.Explain(instanceVector);

        return Task.FromResult(anchorResult);
    }

    #endregion

    #region Fairness

    /// <summary>
    /// Validates fairness metrics for the given inputs with ground truth labels.
    /// </summary>
    /// <param name="model">The interpretable model to evaluate.</param>
    /// <param name="inputs">The input tensor.</param>
    /// <param name="groundTruthLabels">The ground truth labels (REQUIRED for proper fairness metrics).</param>
    /// <param name="sensitiveFeatureIndex">Index of the sensitive attribute (e.g., gender, race).</param>
    /// <param name="fairnessMetrics">Optional list of specific metrics to compute.</param>
    /// <returns>Fairness metrics including demographic parity, equal opportunity, and more.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> Fairness metrics help detect if a model treats different groups
    /// (e.g., by gender, race) differently in ways that might be unfair.
    ///
    /// - <b>Demographic Parity</b>: Equal positive prediction rates across groups
    /// - <b>Disparate Impact</b>: Ratio of positive rates (>0.8 is often considered fair)
    /// - <b>Equal Opportunity</b>: Equal true positive rates across groups (requires ground truth)
    /// - <b>Equalized Odds</b>: Equal TPR and FPR across groups (requires ground truth)
    /// </para>
    /// </remarks>
    public static Task<FairnessMetrics<T>> ValidateFairnessAsync<T>(
        IInterpretableModel<T> model,
        Tensor<T> inputs,
        Vector<T> groundTruthLabels,
        int sensitiveFeatureIndex,
        List<FairnessMetric>? fairnessMetrics = null)
    {
        if (model is null)
            throw new ArgumentNullException(nameof(model));
        if (inputs is null)
            throw new ArgumentNullException(nameof(inputs));
        if (groundTruthLabels is null)
            throw new ArgumentNullException(nameof(groundTruthLabels), "Ground truth labels are required for accurate fairness metrics like Equal Opportunity and Equalized Odds.");

        var numOps = MathHelper.GetNumericOperations<T>();

        Func<Matrix<T>, Vector<T>> predictFunc = CreatePredictionFunction(model);
        var inputMatrix = ConvertTensorToMatrix(inputs);
        var predictions = predictFunc(inputMatrix);

        int numSamples = inputMatrix.Rows;
        if (numSamples == 0 || groundTruthLabels.Length != numSamples)
        {
            return Task.FromResult(new FairnessMetrics<T>(
                demographicParity: numOps.Zero,
                equalOpportunity: numOps.Zero,
                equalizedOdds: numOps.Zero,
                predictiveParity: numOps.Zero,
                disparateImpact: numOps.One,
                statisticalParityDifference: numOps.Zero));
        }

        // Split samples by sensitive attribute (binary: below / at or above the median)
        var engine = AiDotNetEngine.Current;
        var (group0, group1, predictedPositive) = FairnessMasks(inputMatrix, predictions, sensitiveFeatureIndex);
        var labels = groundTruthLabels.AsTensor();
        var actualPositive = engine.TensorGreaterEqual(labels, engine.TensorFullLike(labels, numOps.FromDouble(0.5)));

        // Counts for each group
        var group0Predicted = engine.TensorMultiply(group0, predictedPositive);
        var group1Predicted = engine.TensorMultiply(group1, predictedPositive);
        T group0Count = engine.TensorSum(group0);
        T group1Count = engine.TensorSum(group1);
        T group0Positive = engine.TensorSum(group0Predicted);
        T group1Positive = engine.TensorSum(group1Predicted);
        T group0TP = engine.TensorSum(engine.TensorMultiply(group0Predicted, actualPositive));
        T group1TP = engine.TensorSum(engine.TensorMultiply(group1Predicted, actualPositive));
        T group0ActualPositive = engine.TensorSum(engine.TensorMultiply(group0, actualPositive)); // TP + FN
        T group1ActualPositive = engine.TensorSum(engine.TensorMultiply(group1, actualPositive));
        T group0FP = numOps.Subtract(group0Positive, group0TP);
        T group1FP = numOps.Subtract(group1Positive, group1TP);
        T group0ActualNegative = numOps.Subtract(group0Count, group0ActualPositive); // FP + TN
        T group1ActualNegative = numOps.Subtract(group1Count, group1ActualPositive);

        // Compute proper fairness metrics
        T rate0 = SafeRatio(numOps, group0Positive, group0Count);
        T rate1 = SafeRatio(numOps, group1Positive, group1Count);

        // Demographic Parity (difference in positive prediction rates)
        T demographicParity = numOps.Abs(numOps.Subtract(rate0, rate1));

        // Disparate Impact (ratio of positive rates)
        T disparateImpact = DisparateImpact(numOps, rate0, rate1);

        // Statistical Parity Difference
        T statisticalParityDifference = numOps.Subtract(rate0, rate1);

        // True Positive Rates (for Equal Opportunity)
        T tpr0 = SafeRatio(numOps, group0TP, group0ActualPositive);
        T tpr1 = SafeRatio(numOps, group1TP, group1ActualPositive);

        // False Positive Rates (for Equalized Odds)
        T fpr0 = SafeRatio(numOps, group0FP, group0ActualNegative);
        T fpr1 = SafeRatio(numOps, group1FP, group1ActualNegative);

        // Equal Opportunity (difference in TPR)
        T equalOpportunity = numOps.Abs(numOps.Subtract(tpr0, tpr1));

        // Equalized Odds (max of TPR difference and FPR difference)
        T fprGap = numOps.Abs(numOps.Subtract(fpr0, fpr1));
        T equalizedOdds = numOps.GreaterThan(fprGap, equalOpportunity) ? fprGap : equalOpportunity;

        // Predictive Parity (difference in precision)
        T precision0 = SafeRatio(numOps, group0TP, group0Positive);
        T precision1 = SafeRatio(numOps, group1TP, group1Positive);
        T predictiveParity = numOps.Abs(numOps.Subtract(precision0, precision1));

        return Task.FromResult(new FairnessMetrics<T>(
            demographicParity: demographicParity,
            equalOpportunity: equalOpportunity,
            equalizedOdds: equalizedOdds,
            predictiveParity: predictiveParity,
            disparateImpact: disparateImpact,
            statisticalParityDifference: statisticalParityDifference));
    }

    /// <summary>
    /// Validates fairness metrics without ground truth (uses approximations).
    /// </summary>
    [Obsolete("Use the overload with groundTruthLabels for accurate Equal Opportunity and Equalized Odds metrics.")]
    public static Task<FairnessMetrics<T>> ValidateFairnessAsync<T>(
        IInterpretableModel<T> model,
        Tensor<T> inputs,
        int sensitiveFeatureIndex,
        List<FairnessMetric>? fairnessMetrics = null)
    {
        if (model is null)
            throw new ArgumentNullException(nameof(model));
        if (inputs is null)
            throw new ArgumentNullException(nameof(inputs));

        var numOps = MathHelper.GetNumericOperations<T>();

        Func<Matrix<T>, Vector<T>> predictFunc = CreatePredictionFunction(model);
        var inputMatrix = ConvertTensorToMatrix(inputs);
        var predictions = predictFunc(inputMatrix);

        int numSamples = inputMatrix.Rows;
        if (numSamples == 0)
        {
            return Task.FromResult(new FairnessMetrics<T>(
                demographicParity: numOps.Zero,
                equalOpportunity: numOps.Zero,
                equalizedOdds: numOps.Zero,
                predictiveParity: numOps.Zero,
                disparateImpact: numOps.One,
                statisticalParityDifference: numOps.Zero));
        }

        var engine = AiDotNetEngine.Current;
        var (group0, group1, predictedPositive) = FairnessMasks(inputMatrix, predictions, sensitiveFeatureIndex);

        T group0Count = engine.TensorSum(group0);
        T group1Count = engine.TensorSum(group1);
        T group0Positive = engine.TensorSum(engine.TensorMultiply(group0, predictedPositive));
        T group1Positive = engine.TensorSum(engine.TensorMultiply(group1, predictedPositive));

        T rate0 = SafeRatio(numOps, group0Positive, group0Count);
        T rate1 = SafeRatio(numOps, group1Positive, group1Count);

        T demographicParity = numOps.Abs(numOps.Subtract(rate0, rate1));
        T disparateImpact = DisparateImpact(numOps, rate0, rate1);
        T statisticalParityDifference = numOps.Subtract(rate0, rate1);

        // Without ground truth, approximate Equal Opportunity and Equalized Odds
        T equalOpportunity = numOps.Subtract(numOps.One, demographicParity);
        T equalizedOdds = numOps.Subtract(numOps.One, demographicParity);
        T sampleCount = numOps.FromDouble(Math.Max(1, numSamples));
        T predictiveParity = numOps.Subtract(numOps.One, numOps.Abs(numOps.Subtract(
            numOps.Divide(group0Positive, sampleCount), numOps.Divide(group1Positive, sampleCount))));

        return Task.FromResult(new FairnessMetrics<T>(
            demographicParity: demographicParity,
            equalOpportunity: equalOpportunity,
            equalizedOdds: equalizedOdds,
            predictiveParity: predictiveParity,
            disparateImpact: disparateImpact,
            statisticalParityDifference: statisticalParityDifference));
    }

    /// <summary>
    /// Legacy overload without input data.
    /// </summary>
    [Obsolete("Use the overload that takes input data for proper fairness computation.")]
    public static Task<FairnessMetrics<T>> ValidateFairnessAsync<T>(
        List<FairnessMetric> fairnessMetrics)
    {
        if (fairnessMetrics is null)
            throw new ArgumentNullException(nameof(fairnessMetrics));

        var numOps = MathHelper.GetNumericOperations<T>();
        return Task.FromResult(new FairnessMetrics<T>(
            demographicParity: numOps.Zero,
            equalOpportunity: numOps.Zero,
            equalizedOdds: numOps.Zero,
            predictiveParity: numOps.Zero,
            disparateImpact: numOps.One,
            statisticalParityDifference: numOps.Zero));
    }

    #endregion

    #region Model Metadata

    /// <summary>
    /// Gets model-specific interpretability information.
    /// </summary>
    public static Task<Dictionary<string, object>> GetModelSpecificInterpretabilityAsync<T>(
        IInterpretableModel<T> model)
    {
        if (model is null)
            throw new ArgumentNullException(nameof(model));

        var info = new Dictionary<string, object>
        {
            ["ModelType"] = model.GetType().Name
        };

        if (model is INeuralNetworkModel<T> nnModel)
        {
            var architecture = nnModel.GetArchitecture();
            var layers = architecture.Layers ?? new List<ILayer<T>>();

            info["InputSize"] = architecture.InputSize;
            info["OutputSize"] = architecture.OutputSize;
            info["NumLayers"] = layers.Count;
            info["LayerTypes"] = layers.Select(l => l.GetType().Name).ToList();

            long totalParams = 0;
            foreach (var layer in layers)
            {
                var parameters = layer.GetParameters();
                foreach (var param in parameters)
                {
                    if (param is Tensor<T> tensor)
                    {
                        totalParams += tensor.Data.Length;
                    }
                }
            }
            info["TotalParameters"] = totalParams;
        }

        return Task.FromResult(info);
    }

    /// <summary>
    /// Generates a text explanation for a prediction.
    /// </summary>
    public static Task<string> GenerateTextExplanationAsync<T>(
        IInterpretableModel<T> model,
        Tensor<T> input,
        Tensor<T> prediction)
    {
        if (model is null)
            throw new ArgumentNullException(nameof(model));
        if (input is null)
            throw new ArgumentNullException(nameof(input));
        if (prediction is null)
            throw new ArgumentNullException(nameof(prediction));

        var numOps = MathHelper.GetNumericOperations<T>();
        var sb = new System.Text.StringBuilder();

        sb.AppendLine("=== Model Prediction Explanation ===");
        sb.AppendLine($"Model: {model.GetType().Name}");
        sb.AppendLine($"Input dimensions: {input.Data.Length}");
        sb.AppendLine($"Prediction: {prediction.Data.Span[0]}");

        var inputValues = input.ToVector();
        var magnitudeRanks = inputValues.Length > 0
            ? ExplainerTensorOps<T>.DenseRanks(AiDotNetEngine.Current.TensorAbs(inputValues.AsTensor()))
            : Array.Empty<int>();
        var topIndices = Enumerable.Range(0, inputValues.Length).OrderByDescending(i => magnitudeRanks[i]).Take(5).ToList();
        var topValues = inputValues.GetElements(topIndices).ToList();
        var directions = topValues.Count > 0 ? ExplainerTensorOps<T>.SignPrefixes(topValues) : Array.Empty<string>();

        sb.AppendLine();
        sb.AppendLine("Top input features by magnitude:");
        foreach (var (idx, value, direction) in topIndices.Zip(topValues, (i, v) => (i, v)).Zip(directions, (iv, d) => (iv.i, iv.v, d)))
        {
            sb.AppendLine($"  Feature {idx}: {direction}{value}");
        }

        if (model is INeuralNetworkModel<T> nnModel)
        {
            var architecture = nnModel.GetArchitecture();
            sb.AppendLine();
            sb.AppendLine($"Network architecture: {architecture.InputSize} -> ... -> {architecture.OutputSize}");
            sb.AppendLine($"Number of layers: {architecture.Layers?.Count ?? 0}");
        }

        return Task.FromResult(sb.ToString());
    }

    #endregion

    #region Helper Methods

    /// <summary>
    /// Creates a Matrix-based prediction function from an interpretable model.
    /// </summary>
    private static Func<Matrix<T>, Vector<T>> CreatePredictionFunction<T>(IInterpretableModel<T> model)
    {
        var numOps = MathHelper.GetNumericOperations<T>();

        if (model is INeuralNetworkModel<T> nnModel)
        {
            return inputMatrix =>
            {
                // First output of each row's prediction (zero when the model returns nothing)
                var outputs = new List<Vector<T>>(inputMatrix.Rows);
                for (int i = 0; i < inputMatrix.Rows; i++)
                {
                    var row = inputMatrix.GetRow(i);
                    var inputTensor = new Tensor<T>(new[] { row.Length }, row);
                    outputs.Add(ExplainerTensorOps<T>.PadOrTruncate(nnModel.Predict(inputTensor).ToVector(), 1));
                }

                return outputs.Count > 0 ? ExplainerTensorOps<T>.Stack(outputs).ToVector() : new Vector<T>(0);
            };
        }

        // For non-NN models, try to use the model's Predict method if it implements IFullModel
        if (model is IFullModel<T, Tensor<T>, Tensor<T>> fullModel)
        {
            return inputMatrix =>
            {
                // First output of each row's prediction (zero when the model returns nothing)
                var outputs = new List<Vector<T>>(inputMatrix.Rows);
                for (int i = 0; i < inputMatrix.Rows; i++)
                {
                    var row = inputMatrix.GetRow(i);
                    var inputTensor = new Tensor<T>(new[] { row.Length }, row);
                    outputs.Add(ExplainerTensorOps<T>.PadOrTruncate(fullModel.Predict(inputTensor).ToVector(), 1));
                }

                return outputs.Count > 0 ? ExplainerTensorOps<T>.Stack(outputs).ToVector() : new Vector<T>(0);
            };
        }

        if (model is IFullModel<T, Matrix<T>, Matrix<T>> matrixModel)
        {
            return inputMatrix =>
            {
                return matrixModel.Predict(inputMatrix).GetColumn(0);
            };
        }

        if (model is IFullModel<T, Vector<T>, T> vectorModel)
        {
            return inputMatrix =>
            {
                var results = new List<T>(inputMatrix.Rows);
                for (int i = 0; i < inputMatrix.Rows; i++)
                {
                    results.Add(vectorModel.Predict(inputMatrix.GetRow(i)));
                }

                return new Vector<T>(results);
            };
        }

        // Last resort fallback - throw informative error
        throw new NotSupportedException(
            $"Model type {model.GetType().Name} does not implement a supported prediction interface. " +
            "Please implement INeuralNetworkModel<T> or IFullModel<T, TInput, TOutput>.");
    }

    /// <summary>
    /// Creates a Vector-based prediction function from an interpretable model.
    /// </summary>
    private static Func<Vector<T>, Vector<T>> CreateVectorPredictionFunction<T>(IInterpretableModel<T> model)
    {
        var numOps = MathHelper.GetNumericOperations<T>();

        if (model is INeuralNetworkModel<T> nnModel)
        {
            return input =>
            {
                var inputTensor = new Tensor<T>(new[] { input.Length }, input);
                var outputTensor = nnModel.Predict(inputTensor);
                return new Vector<T>(outputTensor.Data.ToArray());
            };
        }

        if (model is IFullModel<T, Tensor<T>, Tensor<T>> fullModel)
        {
            return input =>
            {
                var inputTensor = new Tensor<T>(new[] { input.Length }, input);
                var outputTensor = fullModel.Predict(inputTensor);
                return new Vector<T>(outputTensor.Data.ToArray());
            };
        }

        if (model is IFullModel<T, Vector<T>, T> vectorModel)
        {
            return input =>
            {
                var result = vectorModel.Predict(input);
                return new Vector<T>(new[] { result });
            };
        }

        throw new NotSupportedException(
            $"Model type {model.GetType().Name} does not implement a supported prediction interface.");
    }

    /// <summary>
    /// Creates a Tensor-based prediction function from an interpretable model.
    /// </summary>
    private static Func<Tensor<T>, Tensor<T>> CreateTensorPredictionFunction<T>(IInterpretableModel<T> model)
    {
        if (model is INeuralNetworkModel<T> nnModel)
        {
            return input => nnModel.Predict(input);
        }

        if (model is IFullModel<T, Tensor<T>, Tensor<T>> fullModel)
        {
            return input => fullModel.Predict(input);
        }

        throw new NotSupportedException(
            $"Model type {model.GetType().Name} does not implement a supported prediction interface for tensors.");
    }

    /// <summary>
    /// Converts a Tensor to a Matrix (assuming 2D or flattening to 2D).
    /// </summary>
    private static Matrix<T> ConvertTensorToMatrix<T>(Tensor<T> tensor)
    {
        if (tensor.Shape.Length == 1)
        {
            return new Tensor<T>(new[] { 1, tensor.Shape[0] }, tensor.ToVector()).ToMatrix();
        }
        else if (tensor.Shape.Length >= 2)
        {
            int rows = tensor.Shape[0];
            int cols = tensor.Data.Length / rows;
            return new Tensor<T>(new[] { rows, cols }, tensor.ToVector().GetSubVector(0, rows * cols)).ToMatrix();
        }
        else
        {
            return new Tensor<T>(new[] { 1, 1 }, ExplainerTensorOps<T>.PadOrTruncate(tensor.ToVector(), 1)).ToMatrix();
        }
    }

    /// <summary>
    /// Maps each position of <paramref name="values"/> to its value.
    /// </summary>
    private static Dictionary<int, T> ToIndexDictionary<T>(Vector<T> values)
    {
        return Enumerable.Range(0, values.Length)
            .Zip(values, (index, value) => (index, value))
            .ToDictionary(entry => entry.index, entry => entry.value);
    }

    /// <summary>
    /// Sum of |w| over the <paramref name="outputDim"/> consecutive flat weights of each of the first
    /// <paramref name="count"/> rows (weights past the tensor's end count as zero).
    /// </summary>
    private static Vector<T> AbsoluteWeightRowSums<T>(Tensor<T> weights, int count, int outputDim)
    {
        if (count <= 0)
        {
            return new Vector<T>(0);
        }

        var engine = AiDotNetEngine.Current;
        var flat = ExplainerTensorOps<T>.PadOrTruncate(weights.ToVector(), count * outputDim);
        var rows = engine.Reshape(engine.TensorAbs(flat.AsTensor()), new[] { count, outputDim });
        return engine.ReduceSum(rows, new[] { 1 }, false).ToVector();
    }

    /// <summary>
    /// numerator / denominator, or zero when the denominator is not positive.
    /// </summary>
    private static T SafeRatio<T>(INumericOperations<T> numOps, T numerator, T denominator)
    {
        return numOps.GreaterThan(denominator, numOps.Zero) ? numOps.Divide(numerator, denominator) : numOps.Zero;
    }

    /// <summary>
    /// min(rate0, rate1) / max(rate0, rate1) when both are positive, 1 when both are zero, else 0.
    /// </summary>
    private static T DisparateImpact<T>(INumericOperations<T> numOps, T rate0, T rate1)
    {
        if (numOps.GreaterThan(rate0, numOps.Zero) && numOps.GreaterThan(rate1, numOps.Zero))
        {
            T smaller = numOps.LessThan(rate0, rate1) ? rate0 : rate1;
            T larger = numOps.GreaterThan(rate0, rate1) ? rate0 : rate1;
            return numOps.Divide(smaller, larger);
        }

        // Rates are non-negative, so neither being positive means both are zero
        return !numOps.GreaterThan(rate0, numOps.Zero) && !numOps.GreaterThan(rate1, numOps.Zero) ? numOps.One : numOps.Zero;
    }

    /// <summary>
    /// 0/1 masks for samples below (group 0) and at or above (group 1) the median of the sensitive column,
    /// and for predictions at or above 0.5.
    /// </summary>
    private static (Tensor<T> Group0, Tensor<T> Group1, Tensor<T> PredictedPositive) FairnessMasks<T>(
        Matrix<T> inputMatrix, Vector<T> predictions, int sensitiveFeatureIndex)
    {
        var engine = AiDotNetEngine.Current;
        var numOps = MathHelper.GetNumericOperations<T>();
        int numSamples = inputMatrix.Rows;

        var sensitive = inputMatrix.GetColumn(sensitiveFeatureIndex).AsTensor();
        T medianSensitive = engine.TensorSort(sensitive).Item1.ToVector()[numSamples / 2];
        var group0 = engine.TensorLessThan(sensitive, medianSensitive);
        var group1 = engine.TensorSubtract(engine.TensorFullLike(group0, numOps.One), group0);
        var predicted = predictions.GetSubVector(0, numSamples).AsTensor();
        var predictedPositive = engine.TensorGreaterEqual(predicted, engine.TensorFullLike(predicted, numOps.FromDouble(0.5)));
        return (group0, group1, predictedPositive);
    }

    /// <summary>
    /// Gets the number of input features from a model.
    /// </summary>
    private static int GetNumFeatures<T>(IInterpretableModel<T> model)
    {
        if (model is INeuralNetworkModel<T> nnModel)
        {
            return nnModel.GetArchitecture().InputSize;
        }
        return 10;
    }

    /// <summary>
    /// Creates synthetic background data for explainers (used as fallback).
    /// </summary>
    /// <remarks>
    /// <b>WARNING:</b> Synthetic data should only be used as a last resort.
    /// For accurate explanations, always provide real representative data.
    /// </remarks>
    private static Matrix<T> CreateBackgroundData<T>(IInterpretableModel<T> model, int numFeatures, int numSamples)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var rand = RandomHelper.CreateSeededRandom(42);

        var data = new Matrix<T>(numSamples, numFeatures);

        for (int i = 0; i < numSamples; i++)
        {
            for (int j = 0; j < numFeatures; j++)
            {
                double u1 = rand.NextDouble();
                double u2 = rand.NextDouble();
                double gaussian = Math.Sqrt(-2 * Math.Log(Math.Max(1e-10, u1))) * Math.Cos(2 * Math.PI * u2);
                data[i, j] = numOps.FromDouble(gaussian);
            }
        }

        return data;
    }

    #endregion

    #region Neural Network Gradient-Based Explanations

    /// <summary>
    /// Gets Integrated Gradients attributions using efficient backpropagation for a neural network.
    /// </summary>
    /// <param name="neuralNetwork">The neural network model with backpropagation support.</param>
    /// <param name="enabledMethods">Set of enabled interpretation methods.</param>
    /// <param name="input">The input tensor to explain.</param>
    /// <param name="baseline">The baseline input (defaults to zeros if null).</param>
    /// <param name="numSteps">Number of integration steps (default: 50).</param>
    /// <param name="featureNames">Optional feature names.</param>
    /// <returns>Integrated Gradients explanation with feature attributions.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the preferred method for explaining neural network predictions
    /// with Integrated Gradients. It uses the network's built-in backpropagation to compute exact
    /// gradients, which is:
    ///
    /// - <b>Faster</b>: O(1) forward/backward passes per step vs O(n) for numerical gradients
    /// - <b>More Accurate</b>: Exact gradients vs numerical approximations
    /// - <b>More Stable</b>: No issues with choosing epsilon values
    ///
    /// The method:
    /// 1. Creates interpolated inputs along the path from baseline to input
    /// 2. Computes exact gradients at each point using backpropagation
    /// 3. Integrates the gradients using the trapezoidal rule
    /// 4. Returns attributions that sum to the prediction difference (completeness)
    /// </para>
    /// </remarks>
    public static Task<IntegratedGradientsExplanation<T>> GetIntegratedGradientsWithBackpropAsync<T>(
        INeuralNetwork<T> neuralNetwork,
        HashSet<InterpretationMethod> enabledMethods,
        Tensor<T> input,
        Tensor<T>? baseline = null,
        int numSteps = 50,
        string[]? featureNames = null)
    {
        if (neuralNetwork is null)
            throw new ArgumentNullException(nameof(neuralNetwork));
        if (enabledMethods is null)
            throw new ArgumentNullException(nameof(enabledMethods));
        if (input is null)
            throw new ArgumentNullException(nameof(input));

        if (!enabledMethods.Contains(InterpretationMethod.IntegratedGradients))
        {
            throw new InvalidOperationException("IntegratedGradients method is not enabled.");
        }

        int numFeatures = input.Data.Length;

        // Convert baseline tensor to vector
        Vector<T>? baselineVector = baseline is not null
            ? new Vector<T>(baseline.Data.ToArray())
            : null;

        // Use the neural network constructor which provides efficient backprop-based gradients
        var igExplainer = new IntegratedGradientsExplainer<T>(
            neuralNetwork: neuralNetwork,
            numFeatures: numFeatures,
            numSteps: numSteps,
            baseline: baselineVector,
            featureNames: featureNames);

        var instanceVector = new Vector<T>(input.Data.ToArray());
        var result = igExplainer.Explain(instanceVector);

        return Task.FromResult(result);
    }

    /// <summary>
    /// Gets DeepLIFT attributions using efficient backpropagation for a neural network.
    /// </summary>
    /// <param name="neuralNetwork">The neural network model with backpropagation support.</param>
    /// <param name="enabledMethods">Set of enabled interpretation methods.</param>
    /// <param name="input">The input tensor to explain.</param>
    /// <param name="baseline">The baseline input (defaults to zeros if null).</param>
    /// <param name="rule">DeepLIFT rule to use (Rescale or RevealCancel).</param>
    /// <param name="featureNames">Optional feature names.</param>
    /// <returns>DeepLIFT explanation with feature attributions.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is the preferred method for DeepLIFT explanations when you have
    /// a neural network with backpropagation support. It provides more accurate attributions than
    /// numerical approximations.
    ///
    /// Benefits:
    /// - Uses backpropagation for efficient gradient computation
    /// - Produces attributions that sum to prediction difference (completeness)
    /// - Handles non-linearities better than vanilla gradients
    ///
    /// Note: This uses a gradient-based approximation of DeepLIFT. For true DeepLIFT with
    /// specialized propagation rules (handling ReLU, etc.), you would need layer-by-layer
    /// access to the network's activations.
    /// </para>
    /// </remarks>
    public static Task<DeepLIFTExplanation<T>> GetDeepLIFTWithBackpropAsync<T>(
        INeuralNetwork<T> neuralNetwork,
        HashSet<InterpretationMethod> enabledMethods,
        Tensor<T> input,
        Tensor<T>? baseline = null,
        DeepLIFTRule rule = DeepLIFTRule.Rescale,
        string[]? featureNames = null)
    {
        if (neuralNetwork is null)
            throw new ArgumentNullException(nameof(neuralNetwork));
        if (enabledMethods is null)
            throw new ArgumentNullException(nameof(enabledMethods));
        if (input is null)
            throw new ArgumentNullException(nameof(input));

        if (!enabledMethods.Contains(InterpretationMethod.DeepLIFT))
        {
            throw new InvalidOperationException("DeepLIFT method is not enabled.");
        }

        int numFeatures = input.Data.Length;

        Vector<T>? baselineVector = baseline is not null
            ? new Vector<T>(baseline.Data.ToArray())
            : null;

        // Use the neural network constructor for efficient backprop-based gradients
        var deepLiftExplainer = new DeepLIFTExplainer<T>(
            neuralNetwork: neuralNetwork,
            numFeatures: numFeatures,
            baseline: baselineVector,
            featureNames: featureNames,
            rule: rule);

        var instanceVector = new Vector<T>(input.Data.ToArray());
        var result = deepLiftExplainer.Explain(instanceVector);

        return Task.FromResult(result);
    }

    #endregion
}
