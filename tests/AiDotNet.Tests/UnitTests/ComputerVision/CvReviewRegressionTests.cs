using AiDotNet.ComputerVision;
using AiDotNet.ComputerVision.Detection.TextDetection;
using AiDotNet.ComputerVision.OCR;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Metrics;
using AiDotNet.Models;
using AiDotNet.Models.Parameters;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ComputerVision;

/// <summary>Behavioral regressions for the shared detection/OCR contracts reviewed in PR #2154.</summary>
public sealed class CvReviewRegressionTests
{
    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    public void RoiAlign_EmptyRois_PreservesEmptyOutputShape(int batch)
    {
        var features = new Tensor<double>(new[] { batch, 3, 4, 5 });
        var result = CvTensorOps<double>.RoIAlign(
            features, Array.Empty<double>(), Array.Empty<int>(), 1, 2, 2);

        Assert.Equal(new[] { 0, 3, 2, 2 }, result.Shape.ToArray());
        Assert.Equal(0, result.Length);
    }

    [Fact]
    public void CtcDecoder_CapsCharactersWithoutCountingBlanksOrRepeatedTimesteps()
    {
        var model = new DecoderProbe(2);

        Assert.Equal("ab", model.Ctc(Logits(0, 0, 1, 1, 0, 2, 2, 3)));
        Assert.Equal("a", model.Ctc(Logits(0, 1, 1, 0)));
    }

    [Fact]
    public void AttentionDecoder_CapsCharactersAndHonorsEndToken()
    {
        var model = new DecoderProbe(2);

        Assert.Equal("ab", model.Attention(Logits(1, 2, 3)));
        Assert.Equal("a", model.Attention(Logits(1, 0, 2)));
    }

    [Fact]
    public void Decoders_ZeroCharacterBudget_EmitNothing()
    {
        var model = new DecoderProbe(0);

        Assert.Empty(model.Ctc(Logits(1, 2)));
        Assert.Empty(model.Attention(Logits(1, 2)));
    }

    [Fact]
    public void CorpusErrorRates_EmptyReferencesWithInsertions_AreUndefined()
    {
        string[] references = { "", "" };
        string[] hypotheses = { "wrong", "words here" };

        Assert.True(double.IsNaN(TextRecognitionMetrics.CharacterErrorRate(references, hypotheses)));
        Assert.True(double.IsNaN(TextRecognitionMetrics.WordErrorRate(references, hypotheses)));
        Assert.Equal(0, TextRecognitionMetrics.CharacterErrorRate(references, references));
        Assert.Equal(0, TextRecognitionMetrics.WordErrorRate(references, references));
        Assert.Equal(0.5, TextRecognitionMetrics.CharacterErrorRate(new[] { "ab" }, new[] { "a" }));
        Assert.Equal(0.5, TextRecognitionMetrics.WordErrorRate(new[] { "a b" }, new[] { "a" }));
    }

    [Fact]
    public void Trainer_EmptyRegistry_FailsWithModelIdentity()
    {
        var model = new EmptyModel();
        var input = new Tensor<double>(new[] { 1 });

        var error = Assert.Throws<InvalidOperationException>(() =>
            TensorModelTrainer<double>.Step(model, input, input, 0.1, model.Predict));

        Assert.Contains(nameof(EmptyModel), error.Message);
    }

    [Fact]
    public async Task Trainer_ConcurrentFirstCalls_WarmUpOnceWithoutDuplicateKeyErrors()
    {
        var model = new EmptyModel();
        var input = new Tensor<double>(new[] { 1 });
        int warmups = 0;
        using var start = new ManualResetEventSlim();
        var calls = Enumerable.Range(0, 8).Select(_ => Task.Run(() =>
        {
            start.Wait();
            return Record.Exception(() => TensorModelTrainer<double>.Step(
                model, input, input, 0.1, value =>
                {
                    Interlocked.Increment(ref warmups);
                    Thread.Sleep(100);
                    return value;
                }));
        })).ToArray();

        start.Set();
        var errors = await Task.WhenAll(calls);

        Assert.All(errors, error => Assert.IsType<InvalidOperationException>(error));
        Assert.Equal(1, warmups);
    }

    [Fact]
    public void Trainer_FailedWarmup_IsRetried()
    {
        var model = new EmptyModel();
        var input = new Tensor<double>(new[] { 1 });
        int calls = 0;
        Tensor<double> Forward(Tensor<double> value)
        {
            if (++calls == 1)
                throw new ArithmeticException("warm-up failure");
            return value;
        }

        Assert.Throws<ArithmeticException>(() =>
            TensorModelTrainer<double>.Step(model, input, input, 0.1, Forward));
        var error = Assert.Throws<InvalidOperationException>(() =>
            TensorModelTrainer<double>.Step(model, input, input, 0.1, Forward));

        Assert.Contains(nameof(EmptyModel), error.Message);
        Assert.Equal(2, calls);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void TensorListAdapter_LayoutIdsMatchLiveChunks(bool collection)
    {
        var first = new Tensor<double>(new[] { 2 }, new Vector<double>(new[] { 3.0, 5.0 }));
        var second = new Tensor<double>(new[] { 1 }, new Vector<double>(new[] { 7.0 }));
        var source = new TensorListParameterSource<double>(
            () => new[] { first }, () => new[] { second });
        IParameterSource<double> adapter = collection
            ? new ComponentCollectionParameterSource<double>(() => new[] { source })
            : new ComponentAccessorParameterSource<double>(() => source);
        var registry = new ParameterComponentRegistry<double>();
        registry.Register("weights", adapter);

        var slots = registry.ParameterLayout.Slots.Where(slot => slot.ParameterCount > 0).ToArray();
        var chunks = registry.GetParameterStateChunks().ToArray();

        Assert.Equal(slots.Select(slot => slot.StableId), chunks.Select(chunk => chunk.StableId));
        Assert.Equal(new[] { 3.0, 5.0, 7.0 }, registry.GetParameters().ToArray());
        Assert.Equal(new[] { 2L, 1L }, slots.Select(slot => slot.ParameterCount.GetValueOrDefault()));
        Assert.Same(first, chunks[0].Tensor);
        Assert.Same(second, chunks[1].Tensor);
        Assert.All(chunks, chunk => Assert.True(chunk.IsWritableInPlace));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ChunkOnlyAdapter_FlatFallbackIsNotWritableModelStorage(bool collection)
    {
        var source = new ChunkOnlySource();
        IParameterSource<double> adapter = collection
            ? new ComponentCollectionParameterSource<double>(() => new[] { source })
            : new ComponentAccessorParameterSource<double>(() => source);
        var registry = new ParameterComponentRegistry<double>();
        registry.Register("weights", adapter);

        var chunk = Assert.Single(registry.GetParameterStateChunks());
        var slot = Assert.Single(registry.ParameterLayout.Slots);

        Assert.Equal(slot.StableId, chunk.StableId);
        Assert.Equal(new[] { 3.0, 5.0 }, chunk.Tensor.ToArray());
        Assert.NotSame(source.Weight, chunk.Tensor);
        Assert.False(chunk.IsWritableInPlace);
        chunk.Tensor[0] = 9;
        Assert.Equal(3.0, source.Weight[0]);
    }

    [Fact]
    public void FlatSource_SnapshotIsNotWritableModelStorage()
    {
        var source = new FlatOnlySource();
        var registry = new ParameterComponentRegistry<double>();
        registry.Register("weight", source);

        var chunk = Assert.Single(registry.GetParameterStateChunks());

        Assert.Equal("weight", chunk.StableId);
        Assert.Equal(new[] { 3.0, 5.0 }, chunk.Tensor.ToArray());
        Assert.False(chunk.IsWritableInPlace);
        chunk.Tensor[0] = 9;
        Assert.Equal(3.0, source.Weight[0]);
    }

    [Fact]
    public void Trainer_ReadyModel_UpdatesLiveWeightAndReusesWarmup()
    {
        var model = new ScalarTrainableModel();
        var input = new Tensor<double>(new[] { 1 }, new Vector<double>(new[] { 1.0 }));
        var target = new Tensor<double>(new[] { 1 });

        double firstLoss = TensorModelTrainer<double>.Step(model, input, target, 0.1, model.Predict);
        Assert.Equal(4.0, firstLoss, 12);
        Assert.Equal(1.6, model.Weight[0], 12);
        double secondLoss = TensorModelTrainer<double>.Step(model, input, target, 0.1, model.Predict);

        Assert.Equal(2.56, secondLoss, 12);
        Assert.Equal(1.28, model.Weight[0], 12);
        Assert.Equal(3, model.ForwardCalls);
    }

    [Fact]
    public void TextPredict_UsesTheSamePreprocessedPixelsAsDetection()
    {
        var model = new TextInputProbe();
        var input = new Tensor<double>(new[] { 1, 3, 4, 6 });
        for (int i = 0; i < input.Length; i++) input[i] = 255;

        var prediction = model.Predict(input);

        Assert.Equal(new[] { 1, 3, 2, 2 }, prediction.Shape.ToArray());
        Assert.All(prediction.ToArray(), value => Assert.Equal(1.0, value, 12));
        Assert.All(input.ToArray(), value => Assert.Equal(255.0, value));
    }

    [Fact]
    public void TextPredict_PreprocessingPreservesTheInputGradient()
    {
        var model = new TextInputProbe();
        var input = new Tensor<double>(new[] { 1, 3, 2, 2 });
        for (int i = 0; i < input.Length; i++) input[i] = 10 + i;
        using var tape = new GradientTape<double>();
        var prediction = model.Predict(input);
        var objective = AiDotNetEngine.Current.ReduceSum(prediction, null);
        var gradients = tape.ComputeGradients(objective, new[] { input });

        Assert.True(gradients.TryGetValue(input, out var gradient));
        Assert.NotNull(gradient);
        Assert.All(gradient.ToArray(), value => Assert.Equal(1.0 / 255.0, value, 12));
        for (int i = 0; i < input.Length; i++) Assert.Equal((10.0 + i) / 255.0, prediction[i], 12);
    }

    [Fact]
    public void TextCopyReplay_UsesBatchOneWithoutChangingSourceShape()
    {
        var source = new TextInputProbe();
        source.Predict(new Tensor<double>(new[] { 4, 3, 4, 6 }));
        var copy = new TextInputProbe();

        source.PrepareCopy(copy);

        Assert.Equal(new[] { 1, 3, 2, 2 }, copy.LastInputShape);
        Assert.Equal(new[] { 4, 3, 2, 2 }, source.LastInputShape);
    }

    [Fact]
    public void TextPredict_NonIntegerResizeHasIndependentPixelAndGradientOracle()
    {
        using var model = new TextInputProbe();
        var input = new Tensor<double>(new[] { 1, 3, 3, 5 });
        for (int i = 0; i < input.Length; i++) input[i] = 10 + i;
        using var tape = new GradientTape<double>();
        var prediction = model.Predict(input);
        Assert.Equal(new[] { 1, 3, 2, 2 }, prediction.Shape.ToArray());

        // Asymmetric 3x5 -> 2x2 sampling coordinates are y={0,1.5}, x={0,2.5}.
        // On this affine pixel ramp, the four interpolated offsets are exact.
        double[] offsets = { 0, 2.5, 7.5, 10 };
        double[] contributions = { 1, 0, 0.5, 0.5, 0, 0.5, 0, 0.25, 0.25, 0, 0.5, 0, 0.25, 0.25, 0 };
        for (int channel = 0; channel < 3; channel++)
            for (int pixel = 0; pixel < 4; pixel++)
                Assert.Equal((10 + channel * 15 + offsets[pixel]) / 255.0,
                    prediction[channel * 4 + pixel], 12);

        var objective = AiDotNetEngine.Current.ReduceSum(prediction, null);
        var gradients = tape.ComputeGradients(objective, new[] { input });
        Assert.True(gradients.TryGetValue(input, out var gradient));
        Assert.NotNull(gradient);
        for (int i = 0; i < input.Length; i++)
        {
            Assert.Equal(contributions[i % 15] / 255.0, gradient[i], 12);
            Assert.Equal(10 + i, input[i]);
        }
    }

    [Fact]
    public void OcrCopyReplay_UsesBatchOneWithoutChangingSourceShape()
    {
        var source = new DecoderProbe(2);
        source.Predict(new Tensor<double>(new[] { 4, 3, 4, 6 }));
        var copy = new DecoderProbe(2);

        source.PrepareCopy(copy);

        Assert.Equal(new[] { 1, 3, 4, 6 }, copy.LastInputShape);
        Assert.Equal(new[] { 4, 3, 4, 6 }, source.LastInputShape);
    }

    private static Tensor<double> Logits(params int[] ids)
    {
        var logits = new Tensor<double>(new[] { 1, ids.Length, 4 });
        for (int i = 0; i < ids.Length; i++) logits[0, i, ids[i]] = 10;
        return logits;
    }

    private sealed class EmptyModel : ModelBase<double, Tensor<double>, Tensor<double>>
    {
        public override ILossFunction<double> DefaultLossFunction => new MeanSquaredErrorLoss<double>();
        public override Tensor<double> Predict(Tensor<double> input) => input;
        public override void Train(Tensor<double> input, Tensor<double> expectedOutput) =>
            throw new NotSupportedException("The test invokes the shared trainer directly.");
        public override IFullModel<double, Tensor<double>, Tensor<double>> WithParameters(Vector<double> parameters) =>
            throw new NotSupportedException("The empty-registry probe has no parameters.");
    }

    private class FlatOnlySource : IParameterSource<double>
    {
        public Tensor<double> Weight { get; } = new(new[] { 2 }, new Vector<double>(new[] { 3.0, 5.0 }));
        public long ParameterCount => Weight.Length;
        public Vector<double> GetParameters() => new(Weight.ToArray());
        public void SetParameters(Vector<double> parameters)
        {
            if (parameters.Length != Weight.Length) throw new ArgumentException("Incorrect count.", nameof(parameters));
            for (int i = 0; i < parameters.Length; i++) Weight[i] = parameters[i];
        }
    }

    private sealed class ChunkOnlySource : FlatOnlySource, IParameterChunkSource<double>
    {
        public IEnumerable<ParameterChunk<double>> GetParameterStateChunks()
        {
            yield return new ParameterChunk<double>("part", ParameterSlotRole.Trainable, Weight);
        }
    }

    private sealed class ScalarTrainableModel : ModelBase<double, Tensor<double>, Tensor<double>>
    {
        public Tensor<double> Weight { get; } = new(new[] { 1 }, new Vector<double>(new[] { 2.0 }));
        public int ForwardCalls { get; private set; }
        public override ILossFunction<double> DefaultLossFunction => new MeanSquaredErrorLoss<double>();
        protected override void RegisterComponents() =>
            RegisterParameterComponent("weight", new TensorListParameterSource<double>(() => new[] { Weight }));
        public override Tensor<double> Predict(Tensor<double> input)
        {
            ForwardCalls++;
            return AiDotNetEngine.Current.TensorMultiply(Weight, input);
        }
        public override void Train(Tensor<double> input, Tensor<double> expectedOutput) =>
            TensorModelTrainer<double>.Step(this, input, expectedOutput, 0.1, Predict);
        public override IFullModel<double, Tensor<double>, Tensor<double>> WithParameters(Vector<double> parameters) =>
            throw new NotSupportedException("The control invokes the shared trainer directly.");
    }

    private sealed class DecoderProbe : OCRBase<double>
    {
        public DecoderProbe(int limit) : base(new OCROptions<double>
        {
            CharacterSet = "abc", MaxSequenceLength = limit, UsePretrained = false
        }) { }

        public string Ctc(Tensor<double> logits) => DecodeCTC(logits);
        public string Attention(Tensor<double> logits) => DecodeAttention(logits, 0);
        public int[] LastInputShape { get; private set; } = Array.Empty<int>();
        public void PrepareCopy(DecoderProbe copy) => PrepareCopyForStateRestore(copy);
        public override string Name => nameof(DecoderProbe);
        protected override Tensor<double> ForwardLogits(Tensor<double> image)
        {
            LastInputShape = image.Shape.ToArray();
            return image;
        }
        public override long GetParameterCount() => 0;
        public override OCRResult<double> Recognize(Tensor<double> image) => throw new NotSupportedException();
        public override (string text, double confidence) RecognizeText(Tensor<double> croppedImage) => throw new NotSupportedException();
        public override Task LoadWeightsAsync(string pathOrUrl, CancellationToken cancellationToken = default) => throw new NotSupportedException();
        public override void SaveWeights(string path) => throw new NotSupportedException();
    }

    private sealed class TextInputProbe : TextDetectorBase<double>
    {
        public TextInputProbe() : base(new TextDetectionOptions<double> { InputSize = new[] { 2, 2 } }) { }
        public override string Name => nameof(TextInputProbe);
        public int[] LastInputShape { get; private set; } = Array.Empty<int>();
        public void PrepareCopy(TextInputProbe copy) => PrepareCopyForStateRestore(copy);
        protected override List<Tensor<double>> Forward(Tensor<double> input)
        {
            LastInputShape = input.Shape.ToArray();
            return new() { input };
        }
        protected override long GetHeadParameterCount() => 0;
        protected override List<TextRegion<double>> PostProcess(List<Tensor<double>> outputs,
            int imageWidth, int imageHeight, double confidenceThreshold) => throw new NotSupportedException();
        public override TextDetectionResult<double> Detect(Tensor<double> image) => throw new NotSupportedException();
        public override TextDetectionResult<double> Detect(Tensor<double> image,
            double confidenceThreshold) => throw new NotSupportedException();
        public override Task LoadWeightsAsync(string pathOrUrl, CancellationToken cancellationToken = default) => throw new NotSupportedException();
        public override void SaveWeights(string path) => throw new NotSupportedException();
    }
}
