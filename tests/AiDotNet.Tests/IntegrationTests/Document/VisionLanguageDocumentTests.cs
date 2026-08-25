using AiDotNet.Document;
using AiDotNet.Document.VisionLanguage;
using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors.Helpers;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNet.Tests.IntegrationTests.Document;

/// <summary>
/// Integration tests for vision-language document models.
/// </summary>
public class VisionLanguageDocumentTests
{
    private static NeuralNetworkArchitecture<double> CreateArchitecture(int imageSize = 64)
    {
        return new NeuralNetworkArchitecture<double>(
            inputType: InputType.ThreeDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            inputHeight: imageSize,
            inputWidth: imageSize,
            inputDepth: 3,
            outputSize: 16);
    }

    private static Tensor<double> CreateSmallImage(int size = 64)
    {
        int totalSize = 1 * 3 * size * size;
        var data = new Vector<double>(totalSize);
        for (int i = 0; i < totalSize; i++)
            data[i] = 0.5;
        return new Tensor<double>(new[] { 1, 3, size, size }, data);
    }

    #region DocOwl Tests

    [Fact(Timeout = 120000)]
    public async Task DocOwl_NativeConstruction_Succeeds()
    {
        var arch = CreateArchitecture();
        var model = new DocOwl<double>(arch, imageSize: 64);
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task DocOwl_Predict_ReturnsOutput()
    {
        var arch = CreateArchitecture();
        var model = new DocOwl<double>(arch, imageSize: 64);
        var input = CreateSmallImage();
        var output = model.Predict(input);
        Assert.NotNull(output);
        Assert.True(output.Shape.Length > 0, "Output should have non-empty shape");
        Assert.True(output.Shape[0] > 0, "Output first dimension should be positive");
    }

    private static DocOwl<double> CreateSmallDocOwl()
        => new DocOwl<double>(CreateArchitecture(imageSize: 64), imageSize: 64,
            maxSequenceLength: 64, visionDim: 32, languageDim: 32,
            visionLayers: 1, languageLayers: 1, numHeads: 4, vocabSize: 64);

    private static Tensor<double> CreateTokenIds(int count, int offset = 0, int vocab = 64)
    {
        var data = new Vector<double>(count);
        for (int i = 0; i < count; i++) data[i] = (i + offset) % vocab;
        return new Tensor<double>(new[] { count }, data);
    }

    /// <summary>
    /// DocOwl's text decoder had no token embedding, so only projected visual tokens ever reached it
    /// and _languageEmbeddings sat dead. Different text with the same image must now move the output.
    /// </summary>
    [Fact(Timeout = 120000)]
    public async Task DocOwl_TextTokens_ChangeTheOutput()
    {
        await Task.Yield();
        var model = CreateSmallDocOwl();
        model.SetTrainingMode(false);
        var image = CreateSmallImage(64);

        var imageOnly = model.Predict(image);
        // Genuinely different sequences. An earlier version varied only the vocab bound, which for
        // six tokens produced the SAME ids both times and made the assertion unfalsifiable.
        var withTextA = model.Predict(image, CreateTokenIds(6, offset: 0));
        var withTextB = model.Predict(image, CreateTokenIds(6, offset: 17));

        Assert.NotNull(imageOnly);
        Assert.True(Differs(withTextA, withTextB),
            "Two different token sequences over the same image gave identical output, so the text " +
            "tokens are not reaching DocOwl's decoder.");
    }

    /// <summary>
    /// The point of routing the second modality through the ordinary forward rather than a
    /// NoGradScope side door: training must actually see it. Every per-model EncodeMultimodal in
    /// this family opens a NoGradScope, which is why LiLT's layout stream could never be trained.
    /// </summary>
    [Fact(Timeout = 120000)]
    public async Task DocOwl_TrainingWithText_ChangesParameters()
    {
        await Task.Yield();
        var model = CreateSmallDocOwl();
        var image = CreateSmallImage(64);
        var tokens = CreateTokenIds(6);

        model.SetTrainingMode(false);

        // Run one forward BEFORE snapshotting. The token table allocates lazily, so a count taken
        // first is short by its size and the comparison below would report a length change rather
        // than the weight movement it is actually testing.
        var logits = model.Predict(image, tokens);
        int nonFiniteLogits = logits.ToArray().Count(value => !IsFinite(value));
        Assert.True(nonFiniteLogits == 0,
            $"DocOwl produced {nonFiniteLogits}/{logits.Length} non-finite logits before training.");
        var before = model.Layers[^1].GetParameters().ToArray();
        Assert.True(before.Length > 0,
            "DocOwl's token embedding exposed no parameters after its first multimodal forward.");

        // DocOwl trains with CrossEntropyWithLogitsLoss, so its dense target must contain one
        // probability distribution per position. Adding a scalar to the logits (the old probe)
        // produced an invalid CE target whose gradient could vanish depending on initialization.
        // Select a different class from the current argmax at every position to guarantee a real,
        // well-posed supervised signal without assuming a particular random initialization.
        var target = CreateContrastingClassTarget(logits);

        var analyticGradients = model.ComputeGradients(image, tokens, target);
        int analyticNonFinite = analyticGradients.Count(value => !IsFinite(value));
        int analyticNonZero = analyticGradients.Count(value => IsFinite(value) && value != 0.0);
        var publishedGradients = model.GetParameterGradients();
        Assert.Equal(analyticGradients.Length, publishedGradients.Length);
        int tokenGradientOffset = publishedGradients.Length - before.Length;
        var tokenGradients = publishedGradients.GetSubVector(tokenGradientOffset, before.Length);
        int tokenNonFinite = tokenGradients.Count(value => !IsFinite(value));
        int tokenNonZero = tokenGradients.Count(value => IsFinite(value) && value != 0.0);
        if (analyticNonFinite > 0 || tokenNonFinite > 0 || tokenNonZero == 0)
        {
            var layerDiagnostics = model.Layers.Select((layer, index) =>
            {
                var layerGradients = (layer as AiDotNet.NeuralNetworks.Layers.LayerBase<double>)?
                    .ScatteredParameterGradients ?? layer.GetParameterGradients();
                int nonFinite = layerGradients.Count(value => !IsFinite(value));
                int nonZero = layerGradients.Count(value => IsFinite(value) && value != 0.0);
                return $"{index}:{layer.GetType().Name}={nonZero} nonzero/{nonFinite} nonfinite/{layerGradients.Length}";
            });
            double maxAbsLogit = logits.ToArray().Max(value => System.Math.Abs(value));
            Assert.Fail(
                $"DocOwl's auxiliary-input gradient is invalid before the optimizer step: " +
                $"loss={model.GetLastLoss():G17}, max |logit|={maxAbsLogit:G17}, " +
                $"all nonzero={analyticNonZero}, all non-finite={analyticNonFinite}/{analyticGradients.Length}, " +
                $"token nonzero={tokenNonZero}, token non-finite={tokenNonFinite}/{tokenGradients.Length}. " +
                string.Join("; ", layerDiagnostics));
        }

        // Exercise the configuration that exposed the shard-only failure. Multi-input training
        // must stay on the eager tape while the compiled cache persists only one input tensor;
        // otherwise it captures a stale auxiliary input and can report a successful no-op step.
        bool compilationWasEnabled = AiDotNet.Tensors.Engines.Optimization.TensorCodecOptions.Current.EnableCompilation;
        try
        {
            AiDotNet.Tensors.Engines.Optimization.TensorCodecOptions.Current.EnableCompilation = true;
            model.Train(image, tokens, target);
        }
        finally
        {
            AiDotNet.Tensors.Engines.Optimization.TensorCodecOptions.Current.EnableCompilation = compilationWasEnabled;
        }

        var after = model.Layers[^1].GetParameters().ToArray();
        Assert.Equal(before.Length, after.Length);

        double maxDelta = 0.0;
        int nonFiniteParameters = 0;
        for (int i = 0; i < before.Length; i++)
        {
            if (!IsFinite(after[i]))
                nonFiniteParameters++;
            else
                maxDelta = System.Math.Max(maxDelta, System.Math.Abs(before[i] - after[i]));
        }

        var gradients = model.GetParameterGradients();
        int nonFiniteGradients = 0;
        int nonZeroGradients = 0;
        for (int i = 0; i < gradients.Length; i++)
        {
            if (!IsFinite(gradients[i])) nonFiniteGradients++;
            else if (gradients[i] != 0.0) nonZeroGradients++;
        }

        Assert.True(nonFiniteParameters == 0 && maxDelta > 1e-12,
            "Training through the auxiliary-input overload left every token-embedding parameter " +
            "untouched, so the text path is not on the gradient tape. " +
            $"max |delta|={maxDelta:G17}, non-finite token parameters={nonFiniteParameters}, " +
            $"published nonzero gradients={nonZeroGradients}/{gradients.Length}, " +
            $"non-finite gradients={nonFiniteGradients}.");
    }

    private static Tensor<double> CreateContrastingClassTarget(Tensor<double> logits)
    {
        var shape = logits.Shape.ToArray();
        Assert.True(shape.Length > 0 && shape[^1] > 1,
            $"DocOwl logits must expose a final class axis; got [{string.Join(",", shape)}].");

        int classCount = shape[^1];
        int positionCount = logits.Length / classCount;
        var target = new Tensor<double>(shape);
        for (int position = 0; position < positionCount; position++)
        {
            int offset = position * classCount;
            int argmax = 0;
            for (int classIndex = 1; classIndex < classCount; classIndex++)
            {
                if (logits[offset + classIndex] > logits[offset + argmax])
                    argmax = classIndex;
            }

            target[offset + ((argmax + 1) % classCount)] = 1.0;
        }

        return target;
    }

    private static bool IsFinite(double value)
        => !double.IsNaN(value) && !double.IsInfinity(value);

    private static bool Differs(Tensor<double> a, Tensor<double> b)
    {
        if (a.Length != b.Length) return true;
        for (int i = 0; i < a.Length; i++)
        {
            if (System.Math.Abs(a.Data.Span[i] - b.Data.Span[i]) > 1e-12) return true;
        }

        return false;
    }

    [Fact(Timeout = 120000)]
    public async Task DocOwl_GetModelMetadata_ReturnsValidData()
    {
        var arch = CreateArchitecture();
        var model = new DocOwl<double>(arch, imageSize: 64);
        var meta = model.GetModelMetadata();
        Assert.Equal("DocOwl", meta.Name);
    }

    #endregion

    #region InfographicVQA Tests

    [Fact(Timeout = 120000)]
    public async Task InfographicVQA_NativeConstruction_Succeeds()
    {
        var arch = CreateArchitecture();
        var model = new InfographicVQA<double>(arch, imageSize: 64);
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task InfographicVQA_Predict_ReturnsOutput()
    {
        var arch = CreateArchitecture();
        var model = new InfographicVQA<double>(arch, imageSize: 64);
        var input = CreateSmallImage();
        var output = model.Predict(input);
        Assert.NotNull(output);
        Assert.True(output.Shape.Length > 0, "Output should have non-empty shape");
        Assert.True(output.Shape[0] > 0, "Output first dimension should be positive");
    }

    [Fact(Timeout = 120000)]
    public async Task InfographicVQA_GetModelMetadata_ReturnsValidData()
    {
        var arch = CreateArchitecture();
        var model = new InfographicVQA<double>(arch, imageSize: 64);
        var meta = model.GetModelMetadata();
        Assert.Equal("InfographicVQA", meta.Name);
    }

    #endregion

    #region UDOP Tests

    [Fact(Timeout = 120000)]
    public async Task UDOP_NativeConstruction_Succeeds()
    {
        var arch = CreateArchitecture();
        var model = new UDOP<double>(arch, imageSize: 64);
        Assert.NotNull(model);
    }

    [Fact(Timeout = 120000)]
    public async Task UDOP_Predict_ReturnsOutput()
    {
        var arch = CreateArchitecture();
        var model = new UDOP<double>(arch, imageSize: 64);
        var input = CreateSmallImage();
        var output = model.Predict(input);
        Assert.NotNull(output);
        Assert.True(output.Shape.Length > 0, "Output should have non-empty shape");
        Assert.True(output.Shape[0] > 0, "Output first dimension should be positive");
    }

    [Fact(Timeout = 120000)]
    public async Task UDOP_GetModelMetadata_ReturnsValidData()
    {
        var arch = CreateArchitecture();
        var model = new UDOP<double>(arch, imageSize: 64);
        var meta = model.GetModelMetadata();
        Assert.Equal("UDOP", meta.Name);
    }

    #endregion

    #region Cross-Model Tests

    [Fact(Timeout = 120000)]
    public async Task AllVisionLanguageModels_SupportsTraining_InNativeMode()
    {
        var arch = CreateArchitecture();
        var models = new DocumentNeuralNetworkBase<double>[]
        {
            new DocOwl<double>(arch, imageSize: 64),
            new InfographicVQA<double>(arch, imageSize: 64),
            new UDOP<double>(arch, imageSize: 64),
        };

        foreach (var model in models)
        {
            Assert.True(model.SupportsTraining);
        }
    }

    #endregion
}
