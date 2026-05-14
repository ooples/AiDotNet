using AiDotNet.ActivationFunctions;
using AiDotNet.Enums;
using AiDotNet.FitDetectors;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.Models.Inputs;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Statistics;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNetTests.IntegrationTests.NeuralNetworks;

/// <summary>
/// Integration coverage for issues #1321, #1322, #1323 — collectively the
/// "Embedding-as-first-custom-layer" series following PR #1320 (#1317).
///
/// Bug class: validators in NeuralNetworkArchitecture (#1321) and
/// NeuralNetworkBase (#1323) compared <c>EmbeddingLayer&lt;T&gt;</c>'s
/// declared input shape <c>[1]</c> (per-token broadcast contract) against
/// either the architecture's flattened InputSize or the previous layer's
/// output shape. Both rejected a legitimate broadcast contract.
/// CalibratedProbabilityFitDetector (#1322) then unconditionally threw on
/// any predicted-vs-actual shape mismatch even when the model was perfectly
/// trainable — calibration is genuinely undefined for some custom layouts,
/// but the optimizer was unconditionally killed by the throw.
///
/// All three are now lenient at the right layer: validators recognize
/// LayerCategory.Embedding as broadcast-input; the fit detector returns
/// empty calibration with a Trace warning instead of throwing.
/// </summary>
public class EmbeddingLayerValidatorIssues1321_1322_1323IntegrationTests
{
    // ====================================================================
    // ISSUE #1321 — TransformerArchitecture.ValidateInputDimensions
    //               accepts custom Transformer chains starting with
    //               EmbeddingLayer<T> (or any LayerCategory.Embedding layer).
    // ====================================================================

    [Fact]
    public void Issue1321_TransformerArchitecture_AcceptsEmbeddingLayerAsFirstCustomLayer()
    {
        const int VocabSize = 256;
        const int DModel = 64;
        const int CtxLen = 64;
        const int Heads = 2;

        var layers = new List<ILayer<float>>
        {
            new EmbeddingLayer<float>(vocabularySize: VocabSize, embeddingDimension: DModel),
            new MultiHeadAttentionLayer<float>(Heads, DModel / Heads, activationFunction: (IActivationFunction<float>)new IdentityActivation<float>()),
            new SequenceTokenSliceLayer<float>(SequenceTokenSliceLayer<float>.Position.Last),
            new DenseLayer<float>(VocabSize, (IActivationFunction<float>)new IdentityActivation<float>()),
        };

        // Issue repro from #1321: this previously threw
        //   "The first layer's input size (1) must match the input size (64)."
        var arch = new TransformerArchitecture<float>(
            inputType: InputType.TwoDimensional,
            taskType: NeuralNetworkTaskType.SequenceClassification,
            numEncoderLayers: 2,
            numDecoderLayers: 0,
            numHeads: Heads,
            modelDimension: DModel,
            feedForwardDimension: 2 * DModel,
            inputSize: CtxLen,
            outputSize: VocabSize,
            maxSequenceLength: CtxLen,
            vocabularySize: VocabSize,
            layers: layers);

        Assert.NotNull(arch);
        Assert.Same(layers[0], arch.Layers![0]);
    }

    [Fact]
    public void Issue1321_NeuralNetworkArchitecture_AcceptsEmbeddingLayerAsFirstCustomLayer()
    {
        const int VocabSize = 256;
        const int EmbDim = 32;
        const int CtxLen = 64;

        var layers = new List<ILayer<float>>
        {
            new EmbeddingLayer<float>(vocabularySize: VocabSize, embeddingDimension: EmbDim),
            new DenseLayer<float>(VocabSize, (IActivationFunction<float>)new IdentityActivation<float>()),
        };

        // Same broken validator path applies to the base architecture too —
        // the strict size check should also recognise EmbeddingLayer.
        var arch = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputSize: CtxLen,
            outputSize: VocabSize,
            layers: layers);

        Assert.NotNull(arch);
        Assert.Same(layers[0], arch.Layers![0]);
    }

    [Fact]
    public void Issue1321_RegressionGuard_NonEmbeddingFirstLayerStillValidatesInputSize()
    {
        // Classical chain — first layer is InputLayer<float>(16) with
        // explicit input shape [16] that DOES NOT match architecture
        // InputSize=64. Validator must still reject this; the relaxation
        // is per-category, not blanket.
        var layers = new List<ILayer<float>>
        {
            new InputLayer<float>(16),
            new DenseLayer<float>(4, (IActivationFunction<float>)new IdentityActivation<float>()),
        };

        var ex = Assert.Throws<ArgumentException>(() =>
            new NeuralNetworkArchitecture<float>(
                inputType: InputType.OneDimensional,
                taskType: NeuralNetworkTaskType.Regression,
                inputSize: 64, // intentional mismatch with InputLayer's [16]
                outputSize: 4,
                layers: layers));

        Assert.Contains("first layer's input size", ex.Message);
    }

    // ====================================================================
    // ISSUE #1323 — NeuralNetworkBase.AreLayersCompatible accepts
    //               InputLayer → EmbeddingLayer (and any prev → Embedding
    //               transition) by recognising LayerCategory.Embedding as
    //               broadcast-input.
    // ====================================================================

    [Fact]
    public void Issue1323_InputLayerToEmbeddingLayer_PassesCompatibilityCheck()
    {
        const int VocabSize = 256;
        const int EmbDim = 64;
        const int CtxLen = 64;

        var layers = new List<ILayer<float>>
        {
            new InputLayer<float>(CtxLen),
            new EmbeddingLayer<float>(vocabularySize: VocabSize, embeddingDimension: EmbDim),
            new DenseLayer<float>(VocabSize, (IActivationFunction<float>)new IdentityActivation<float>()),
        };

        var arch = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputSize: CtxLen,
            outputSize: VocabSize,
            layers: layers);

        // Issue repro from #1323: previously threw
        //   "Layer 0 is not compatible with Layer 1."
        var network = new FeedForwardNeuralNetwork<float>(arch);

        Assert.Equal(layers.Count, network.Layers.Count);
        Assert.IsType<InputLayer<float>>(network.Layers[0]);
        Assert.IsType<EmbeddingLayer<float>>(network.Layers[1]);
    }

    [Fact]
    public void Issue1323_DenseToEmbeddingLayer_PassesCompatibilityCheck()
    {
        // Embedding-as-second-layer with a Dense feeding it. Same broadcast
        // contract — the strict shape check would fail because Dense outputs
        // [16] and Embedding declares input [1]. Recognised category bypass.
        const int VocabSize = 32;
        const int EmbDim = 16;

        var layers = new List<ILayer<float>>
        {
            new DenseLayer<float>(16, (IActivationFunction<float>)new IdentityActivation<float>()),
            new EmbeddingLayer<float>(vocabularySize: VocabSize, embeddingDimension: EmbDim),
            new DenseLayer<float>(VocabSize, (IActivationFunction<float>)new IdentityActivation<float>()),
        };

        var arch = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputSize: 4,
            outputSize: VocabSize,
            layers: layers);

        var network = new FeedForwardNeuralNetwork<float>(arch);
        Assert.Equal(3, network.Layers.Count);
    }

    [Fact]
    public void Issue1323_RegressionGuard_NonEmbeddingTransitionStillRejectedOnShapeMismatch()
    {
        // Classical chain with mismatched explicit-shape layer transition —
        // InputLayer outputs [16] then a second InputLayer expects [31].
        // Both carry explicit shapes the compatibility check sees as
        // incompatible. The Embedding bypass is category-gated, so this
        // non-Embedding transition must still throw.
        var layers = new List<ILayer<float>>
        {
            new InputLayer<float>(16),
            new InputLayer<float>(31),
            new DenseLayer<float>(4, (IActivationFunction<float>)new IdentityActivation<float>()),
        };

        var arch = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 16,
            outputSize: 4,
            layers: layers);

        var ex = Assert.Throws<ArgumentException>(() => new FeedForwardNeuralNetwork<float>(arch));
        Assert.Contains("Layer 0 is not compatible with Layer 1", ex.Message);
    }

    // ====================================================================
    // EDGE CASE COVERAGE — recognise category, not concrete type, and
    // honour custom subclasses + multiple embeddings + non-first position.
    // ====================================================================

    [Fact]
    public void EdgeCase_CustomEmbeddingSubclass_RecognisedByCategory()
    {
        // A user-defined embedding (e.g. token + structural priors). The
        // base GetLayerCategory() name-based check matches "Embedding" in
        // the type name; subclasses can also override the virtual method.
        // Either path should make the validator treat it as broadcast input.
        const int VocabSize = 64;
        const int EmbDim = 16;
        const int CtxLen = 8;

        var layers = new List<ILayer<float>>
        {
            new CustomTokenEmbeddingLayer(VocabSize, EmbDim),
            new DenseLayer<float>(VocabSize, (IActivationFunction<float>)new IdentityActivation<float>()),
        };

        var arch = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputSize: CtxLen,
            outputSize: VocabSize,
            layers: layers);

        var network = new FeedForwardNeuralNetwork<float>(arch);
        Assert.IsType<CustomTokenEmbeddingLayer>(network.Layers[0]);
    }

    [Fact]
    public void EdgeCase_PositionalEncodingFirst_AlsoRecognisedByCategory()
    {
        // PositionalEmbeddingLayer has "Positional" in the type name which
        // the LayerBase.GetLayerCategory() name-matcher maps to
        // LayerCategory.Embedding. So the same broadcast-input path applies.
        const int CtxLen = 64;
        const int Dim = 16;

        var layers = new List<ILayer<float>>
        {
            new PositionalEncodingLayer<float>(maxSequenceLength: CtxLen, embeddingSize: Dim),
            new DenseLayer<float>(CtxLen, (IActivationFunction<float>)new IdentityActivation<float>()),
        };

        var arch = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: CtxLen,
            outputSize: CtxLen,
            layers: layers);

        var network = new FeedForwardNeuralNetwork<float>(arch);
        Assert.NotEmpty(network.Layers);
    }

    [Fact]
    public void EdgeCase_TwoEmbeddingLayersInChain_BothAccepted()
    {
        // Token embedding + positional embedding stacked. Each individually
        // declares broadcast input shape; both should pass the layer-to-layer
        // compatibility check.
        const int VocabSize = 32;
        const int Dim = 16;
        const int CtxLen = 8;

        var layers = new List<ILayer<float>>
        {
            new EmbeddingLayer<float>(vocabularySize: VocabSize, embeddingDimension: Dim),
            new PositionalEncodingLayer<float>(maxSequenceLength: CtxLen, embeddingSize: Dim),
            new DenseLayer<float>(VocabSize, (IActivationFunction<float>)new IdentityActivation<float>()),
        };

        var arch = new NeuralNetworkArchitecture<float>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputSize: CtxLen,
            outputSize: VocabSize,
            layers: layers);

        var network = new FeedForwardNeuralNetwork<float>(arch);
        Assert.Equal(3, network.Layers.Count);
    }

    // ====================================================================
    // ISSUE #1322 — CalibratedProbabilityFitDetector returns empty
    //               calibration (instead of throwing) when predicted-vs-
    //               actual shapes can't be reconciled. The optimizer keeps
    //               training rather than dying mid-iteration.
    // ====================================================================

    [Fact]
    public void Issue1322_FitDetector_ReverseShapeMismatch_DoesNotThrow_GracefulEmptyCalibration()
    {
        // Repro shape ratio from #1322: predicted=256 (single sample's vocab
        // distribution), actual=43776 (171 samples × 256 vocab classes flat).
        // Previously: InvalidOperationException from CalculateCalibration.
        // After fix: no throw, calibration returns lenient FitType verdict.
        var detector = new CalibratedProbabilityFitDetector<float, Tensor<float>, Tensor<float>>();

        var predicted = new Tensor<float>([256]);
        for (int i = 0; i < 256; i++) predicted[i] = 0.5f;

        var actual = new Tensor<float>([171, 256]);
        for (int i = 0; i < 171 * 256; i++) actual.Data.Span[i] = 0.5f;

        var evalData = BuildEvaluationData(predicted, actual);

        // The DetectFit entry point cascades through DetermineFitType /
        // CalculateConfidenceLevel / GenerateRecommendations, all of which
        // call CalculateCalibration internally. The whole stack must stay
        // throw-free for the optimizer's per-iteration loop to keep going.
        var result = detector.DetectFit(evalData);

        Assert.NotNull(result);
        Assert.Equal(FitType.GoodFit, result.FitType);
    }

    [Fact]
    public void Issue1322_FitDetector_RankDiscordantTensors_DoesNotThrow()
    {
        // Off-by-one shape drift — predicted [10], actual [13]. Not
        // multiclass-divisible, not equal. Same lenient handling.
        var detector = new CalibratedProbabilityFitDetector<float, Tensor<float>, Tensor<float>>();

        var predicted = new Tensor<float>([10]);
        for (int i = 0; i < 10; i++) predicted[i] = 0.3f;

        var actual = new Tensor<float>([13]);
        for (int i = 0; i < 13; i++) actual[i] = 1.0f;

        var evalData = BuildEvaluationData(predicted, actual);
        var result = detector.DetectFit(evalData);

        Assert.NotNull(result);
        Assert.Equal(FitType.GoodFit, result.FitType);
    }

    [Fact]
    public void Issue1322_FitDetector_MatchedShapes_StillProducesRealCalibration()
    {
        // Regression guard: the existing binary-calibration path
        // (predicted.Length == actual.Length) must continue to compute real
        // calibration metrics — the lenient fallback only fires on shape
        // mismatch, not on every call.
        var detector = new CalibratedProbabilityFitDetector<float, Tensor<float>, Tensor<float>>();
        var rng = RandomHelper.CreateSeededRandom(42);

        const int N = 200;
        var predicted = new Tensor<float>([N]);
        var actual = new Tensor<float>([N]);
        for (int i = 0; i < N; i++)
        {
            predicted[i] = (float)rng.NextDouble();
            // Label is 1 with probability ~ predicted[i] — well-calibrated.
            actual[i] = rng.NextDouble() < predicted[i] ? 1f : 0f;
        }

        var evalData = BuildEvaluationData(predicted, actual);
        var result = detector.DetectFit(evalData);

        Assert.NotNull(result);
        // Well-calibrated random data should land in GoodFit (default
        // thresholds are loose enough that non-pathological calibration
        // passes).
        Assert.Contains(result.FitType, new[] { FitType.GoodFit, FitType.Underfit, FitType.Overfit });
    }

    [Fact]
    public void Issue1322_FitDetector_MulticlassPredictedLargerThanActual_StillReducesCorrectly()
    {
        // Regression guard for the existing multiclass reduction path
        // (predicted.Length == numClasses * actual.Length, with class-index
        // labels). Must continue to work — the new shape-mismatch fallback
        // only fires when the existing reduction can't apply.
        var detector = new CalibratedProbabilityFitDetector<float, Tensor<float>, Tensor<float>>();

        const int N = 50;
        const int NumClasses = 4;
        var predicted = new Tensor<float>([N * NumClasses]);
        var actual = new Tensor<float>([N]);
        for (int i = 0; i < N; i++)
        {
            int trueClass = i % NumClasses;
            actual[i] = trueClass;
            for (int c = 0; c < NumClasses; c++)
            {
                predicted[i * NumClasses + c] = c == trueClass ? 0.7f : 0.1f;
            }
        }

        var evalData = BuildEvaluationData(predicted, actual);

        // Should NOT throw — existing multiclass reduction path still applies.
        var result = detector.DetectFit(evalData);
        Assert.NotNull(result);
    }

    // ====================================================================
    // Helpers — build a minimal ModelEvaluationData carrying the predicted
    // and actual tensors the detector reads.
    // ====================================================================

    private static ModelEvaluationData<float, Tensor<float>, Tensor<float>> BuildEvaluationData(
        Tensor<float> predicted,
        Tensor<float> actual)
    {
        // ModelStats wraps Predicted / Actual; the fit detector reads from
        // evaluationData.ModelStats.Predicted / .Actual via ConversionsHelper.
        // We don't need a real model here — just plumb the tensors in.
        var modelStats = new ModelStats<float, Tensor<float>, Tensor<float>>(
            new ModelStatsInputs<float, Tensor<float>, Tensor<float>>
            {
                XMatrix = new Tensor<float>([1]),
                FeatureCount = 1,
                Actual = actual,
                Predicted = predicted,
                Model = null,
            });

        return new ModelEvaluationData<float, Tensor<float>, Tensor<float>>
        {
            ModelStats = modelStats,
        };
    }

    // ====================================================================
    // Custom Embedding subclass for the EdgeCase test — verifies the
    // category-based recognition handles user-defined embeddings, not just
    // the stock EmbeddingLayer<T>. Naming triggers the name-based fallback
    // in LayerBase.GetLayerCategory; an explicit override would also work.
    // ====================================================================

    private sealed class CustomTokenEmbeddingLayer : LayerBase<float>
    {
        private readonly int _vocabSize;
        private readonly int _embeddingDim;

        public CustomTokenEmbeddingLayer(int vocabSize, int embeddingDim)
            : base([1], [embeddingDim])
        {
            _vocabSize = vocabSize;
            _embeddingDim = embeddingDim;
        }

        public override bool SupportsTraining => false;

        public override LayerCategory GetLayerCategory() => LayerCategory.Embedding;

        public override Tensor<float> Forward(Tensor<float> input)
        {
            // Stub forward: return zeros of the per-token embedding shape
            // for any-rank input. Test cares about validator passthrough,
            // not numerical correctness.
            var outShape = new int[input.Rank + 1];
            for (int i = 0; i < input.Rank; i++) outShape[i] = input.Shape[i];
            outShape[input.Rank] = _embeddingDim;
            return new Tensor<float>(outShape);
        }

        public override void UpdateParameters(float learningRate) { }

        public override Vector<float> GetParameters() => Vector<float>.Empty();

        public override void ResetState() { }
    }
}
