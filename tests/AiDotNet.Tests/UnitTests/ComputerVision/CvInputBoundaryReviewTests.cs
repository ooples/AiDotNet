using AiDotNet.ComputerVision.Detection.ObjectDetection;
using AiDotNet.ComputerVision.Detection.ObjectDetection.RCNN;
using AiDotNet.ComputerVision.Detection.TextDetection;
using AiDotNet.Models.Options;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.ComputerVision;

/// <summary>Exercises the shared input-validation boundaries without constructing a large detector.</summary>
public sealed class CvInputBoundaryReviewTests
{
    public enum DetectorEntryPoint { Serialization, Preprocessing }
    public enum TextDetectorEntryPoint { Prediction, Serialization, Preprocessing }
    public enum PyramidEntryPoint { Assignment, Pooling }

    public static TheoryData<int[], DetectorEntryPoint> InvalidInputSizes
    {
        get
        {
            var cases = new TheoryData<int[], DetectorEntryPoint>();
            foreach (int[] shape in new[]
            {
                Array.Empty<int>(), new[] { 2 }, new[] { 2, 3, 4 },
                new[] { 0, 3 }, new[] { 2, 0 }, new[] { -1, 3 }, new[] { 2, -1 }
            })
            {
                cases.Add(shape, DetectorEntryPoint.Serialization);
                cases.Add(shape, DetectorEntryPoint.Preprocessing);
            }
            return cases;
        }
    }

    [Theory]
    [MemberData(nameof(InvalidInputSizes))]
    public void Detector_RejectsInvalidConfiguredDimensionsBeforeForward(
        int[] shape, DetectorEntryPoint entryPoint)
    {
        var options = new ObjectDetectionOptions<double> { InputSize = shape, UsePretrained = false };
        using var model = new DetectorProbe(options);

        var error = Assert.Throws<ArgumentException>(() => InvokeDetector(model, entryPoint));

        Assert.Equal(nameof(options.InputSize), error.ParamName);
        Assert.Equal(0, model.ForwardCalls);
    }

    [Theory]
    [InlineData(DetectorEntryPoint.Serialization)]
    [InlineData(DetectorEntryPoint.Preprocessing)]
    public void Detector_RejectsNullConfigurationFromExternalBinding(DetectorEntryPoint entryPoint)
    {
        var options = new ObjectDetectionOptions<double> { UsePretrained = false };
        // External binding can assign null despite the non-nullable public declaration.
        var property = typeof(ObjectDetectionOptions<double>).GetProperty(nameof(options.InputSize))
            ?? throw new InvalidOperationException("The public input-size property is missing.");
        property.SetValue(options, null);
        using var model = new DetectorProbe(options);

        var error = Assert.Throws<ArgumentException>(() => InvokeDetector(model, entryPoint));

        Assert.Equal(nameof(options.InputSize), error.ParamName);
        Assert.Equal(0, model.ForwardCalls);
    }

    [Theory]
    [InlineData(1, 1)]
    [InlineData(2, 3)]
    public void Detector_ValidDeferredSerializationUsesConfiguredDimensionsOnce(int height, int width)
    {
        var options = new ObjectDetectionOptions<double>
        {
            InputSize = new[] { height, width }, UsePretrained = false
        };
        using var model = new DetectorProbe(options);

        Assert.NotEmpty(model.Serialize());
        Assert.Equal(new[] { 1, 3, height, width }, model.LastInputShape);
        Assert.NotEmpty(model.Serialize());
        Assert.Equal(1, model.ForwardCalls);
    }

    [Fact]
    public void Detector_ResolvedSerializationDoesNotReenterDeferredProbe()
    {
        var options = new ObjectDetectionOptions<double> { InputSize = new[] { 2, 3 }, UsePretrained = false };
        using var model = new DetectorProbe(options);
        model.Predict(new Tensor<double>(new[] { 2, 3, 2, 3 }));
        options.InputSize = Array.Empty<int>();

        Assert.NotEmpty(model.Serialize());
        Assert.Equal(1, model.ForwardCalls);
        Assert.Equal(new[] { 2, 3, 2, 3 }, model.LastInputShape);
    }

    public static TheoryData<int[], TextDetectorEntryPoint> InvalidTextInputSizes
    {
        get
        {
            var cases = new TheoryData<int[], TextDetectorEntryPoint>();
            foreach (int[] shape in new[]
            {
                Array.Empty<int>(), new[] { 2 }, new[] { 2, 3, 4 },
                new[] { 0, 3 }, new[] { 2, 0 }, new[] { -1, 3 }, new[] { 2, -1 }
            })
            {
                cases.Add(shape, TextDetectorEntryPoint.Prediction);
                cases.Add(shape, TextDetectorEntryPoint.Serialization);
                cases.Add(shape, TextDetectorEntryPoint.Preprocessing);
            }
            return cases;
        }
    }

    [Theory]
    [MemberData(nameof(InvalidTextInputSizes))]
    public void TextDetector_RejectsMutatedConfiguredDimensionsBeforeForward(
        int[] shape, TextDetectorEntryPoint entryPoint)
    {
        var options = new TextDetectionOptions<double> { InputSize = new[] { 2, 3 } };
        using var model = new TextDetectorProbe(options);
        options.InputSize = shape;

        var error = Assert.Throws<ArgumentException>(() => InvokeTextDetector(model, entryPoint));

        Assert.Equal(nameof(options.InputSize), error.ParamName);
        Assert.Equal(0, model.ForwardCalls);
    }

    [Theory]
    [InlineData(TextDetectorEntryPoint.Prediction)]
    [InlineData(TextDetectorEntryPoint.Serialization)]
    [InlineData(TextDetectorEntryPoint.Preprocessing)]
    public void TextDetector_RejectsNullConfigurationFromExternalBinding(TextDetectorEntryPoint entryPoint)
    {
        var options = new TextDetectionOptions<double> { InputSize = new[] { 2, 3 } };
        using var model = new TextDetectorProbe(options);
        var property = typeof(TextDetectionOptions<double>).GetProperty(nameof(options.InputSize))
            ?? throw new InvalidOperationException("The public input-size property is missing.");
        property.SetValue(options, null);

        var error = Assert.Throws<ArgumentException>(() => InvokeTextDetector(model, entryPoint));

        Assert.Equal(nameof(options.InputSize), error.ParamName);
        Assert.Equal(0, model.ForwardCalls);
    }

    [Theory]
    [InlineData(1, 1, TextDetectorEntryPoint.Prediction)]
    [InlineData(2, 3, TextDetectorEntryPoint.Prediction)]
    [InlineData(1, 1, TextDetectorEntryPoint.Preprocessing)]
    [InlineData(2, 3, TextDetectorEntryPoint.Preprocessing)]
    public void TextDetector_ValidDimensionsPreserveResizeAndNormalization(
        int height, int width, TextDetectorEntryPoint entryPoint)
    {
        var options = new TextDetectionOptions<double> { InputSize = new[] { height, width } };
        using var model = new TextDetectorProbe(options);
        var input = new Tensor<double>(new[] { 1, 3, 3, 5 });
        for (int i = 0; i < input.Length; i++) input[i] = 255.0;

        var result = entryPoint switch
        {
            TextDetectorEntryPoint.Prediction => model.Predict(input),
            TextDetectorEntryPoint.Preprocessing => model.Prepare(input),
            _ => throw new ArgumentOutOfRangeException(nameof(entryPoint))
        };

        Assert.Equal(new[] { 1, 3, height, width }, result.Shape);
        Assert.Equal(entryPoint == TextDetectorEntryPoint.Prediction ? 1 : 0, model.ForwardCalls);
        for (int i = 0; i < result.Length; i++) Assert.Equal(1.0, result[i], 12);
        for (int i = 0; i < input.Length; i++) Assert.Equal(255.0, input[i]);
    }

    [Theory]
    [InlineData(1, 1)]
    [InlineData(2, 3)]
    public void TextDetector_ValidDeferredSerializationUsesConfiguredDimensionsOnce(int height, int width)
    {
        var options = new TextDetectionOptions<double> { InputSize = new[] { height, width } };
        using var model = new TextDetectorProbe(options);

        Assert.NotEmpty(model.Serialize());
        Assert.Equal(new[] { 1, 3, height, width }, model.LastInputShape);
        Assert.NotEmpty(model.Serialize());
        Assert.Equal(1, model.ForwardCalls);
    }

    [Fact]
    public void TextDetector_ResolvedSerializationDoesNotReadUnusedConfiguredDimensions()
    {
        var options = new TextDetectionOptions<double> { InputSize = new[] { 2, 3 } };
        using var model = new TextDetectorProbe(options);
        model.Predict(new Tensor<double>(new[] { 2, 3, 2, 3 }));
        options.InputSize = Array.Empty<int>();

        Assert.NotEmpty(model.Serialize());
        Assert.Equal(1, model.ForwardCalls);
        Assert.Equal(new[] { 2, 3, 2, 3 }, model.LastInputShape);
    }

    [Theory]
    [InlineData(TextDetectorEntryPoint.Prediction)]
    [InlineData(TextDetectorEntryPoint.Preprocessing)]
    public void TextDetector_RejectsInPlaceDimensionMutationAfterAValidPrediction(
        TextDetectorEntryPoint entryPoint)
    {
        var options = new TextDetectionOptions<double> { InputSize = new[] { 2, 3 } };
        using var model = new TextDetectorProbe(options);
        model.Predict(new Tensor<double>(new[] { 1, 3, 2, 3 }));
        options.InputSize[1] = 0;

        var error = Assert.Throws<ArgumentException>(() => InvokeTextDetector(model, entryPoint));

        Assert.Equal(nameof(options.InputSize), error.ParamName);
        Assert.Equal(1, model.ForwardCalls);
    }

    public static TheoryData<int[], PyramidEntryPoint> InvalidStrides
    {
        get
        {
            var cases = new TheoryData<int[], PyramidEntryPoint>();
            foreach (int[] strides in new[]
            {
                Array.Empty<int>(), new[] { 4, 16 }, new[] { 4, 8, 32 },
                new[] { 8, 4 }, new[] { 4, 4 }, new[] { 3, 6 },
                new[] { 0, 2 }, new[] { -4, -8 }
            })
            {
                cases.Add(strides, PyramidEntryPoint.Assignment);
                cases.Add(strides, PyramidEntryPoint.Pooling);
            }
            return cases;
        }
    }

    [Theory]
    [MemberData(nameof(InvalidStrides))]
    public void Pyramid_RejectsInvalidStridesBeforeAssignmentOrPooling(
        int[] strides, PyramidEntryPoint entryPoint)
    {
        var boxes = Boxes(224);
        var error = Assert.Throws<ArgumentException>(() =>
        {
            if (entryPoint == PyramidEntryPoint.Assignment)
            {
                FpnRoIPooler<double>.AssignLevels(boxes, strides);
            }
            else
            {
                var levels = strides.Select(_ => new Tensor<double>(new[] { 1, 1, 2, 2 })).ToArray();
                FpnRoIPooler<double>.Pool(new RoIAlign<double>(1, 1), levels, strides, boxes);
            }
        });

        Assert.Equal(nameof(strides), error.ParamName);
    }

    [Theory]
    [InlineData(int.MaxValue)]
    [InlineData((1 << 30) + 1)]
    public void Pyramid_RejectsOverflowingStrideWithoutEnteringLegacyShiftLoop(int stride)
    {
        var error = Assert.Throws<ArgumentException>(() =>
            FpnRoIPooler<double>.AssignLevels(Boxes(224), new[] { stride }));

        Assert.Equal("strides", error.ParamName);
    }

    [Theory]
    [InlineData(1, 2)]
    [InlineData(4, 8)]
    [InlineData(1 << 29, 1 << 30)]
    public void Pyramid_ValidContiguousBoundaryStridesKeepIndexesInRange(int first, int second)
    {
        var assignments = FpnRoIPooler<double>.AssignLevels(Boxes(0.01, 1e12), new[] { first, second });

        Assert.Equal(new[] { 0, 1 }, assignments);
    }

    [Fact]
    public void Pyramid_MaximumSingleStrideIsValid()
    {
        var assignments = FpnRoIPooler<double>.AssignLevels(Boxes(224), new[] { 1 << 30 });

        Assert.Equal(new[] { 0 }, assignments);
    }

    [Fact]
    public void Pyramid_LevelCountMismatchIsStillRejected()
    {
        var error = Assert.Throws<ArgumentException>(() => FpnRoIPooler<double>.Pool(
            new RoIAlign<double>(1, 1), new[] { new Tensor<double>(new[] { 1, 1, 2, 2 }) },
            new[] { 4, 8 }, Boxes(224)));

        Assert.Equal("strides", error.ParamName);
    }

    private static void InvokeDetector(DetectorProbe model, DetectorEntryPoint entryPoint)
    {
        switch (entryPoint)
        {
            case DetectorEntryPoint.Serialization:
                model.Serialize();
                break;
            case DetectorEntryPoint.Preprocessing:
                model.Prepare(new Tensor<double>(new[] { 1, 3, 2, 3 }));
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(entryPoint));
        }
    }

    private static void InvokeTextDetector(TextDetectorProbe model, TextDetectorEntryPoint entryPoint)
    {
        switch (entryPoint)
        {
            case TextDetectorEntryPoint.Prediction:
                model.Predict(new Tensor<double>(new[] { 1, 3, 2, 3 }));
                break;
            case TextDetectorEntryPoint.Serialization:
                model.Serialize();
                break;
            case TextDetectorEntryPoint.Preprocessing:
                model.Prepare(new Tensor<double>(new[] { 1, 3, 2, 3 }));
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(entryPoint));
        }
    }

    private static Tensor<double> Boxes(params double[] sides)
    {
        var boxes = new Tensor<double>(new[] { sides.Length, 4 });
        for (int i = 0; i < sides.Length; i++)
        {
            boxes[i, 2] = sides[i];
            boxes[i, 3] = sides[i];
        }
        return boxes;
    }

    private sealed class DetectorProbe : ObjectDetectorBase<double>
    {
        public DetectorProbe(ObjectDetectionOptions<double> options) : base(options) { }
        public override string Name => nameof(DetectorProbe);
        public int ForwardCalls { get; private set; }
        public int[] LastInputShape { get; private set; } = Array.Empty<int>();
        public Tensor<double> Prepare(Tensor<double> image) => Preprocess(image);
        protected override List<Tensor<double>> Forward(Tensor<double> input)
        {
            ForwardCalls++;
            LastInputShape = input.Shape.ToArray();
            return new() { input };
        }
        protected override long GetHeadParameterCount() => 0;
        public override DetectionResult<double> Detect(Tensor<double> image,
            double confidenceThreshold, double nmsThreshold) => throw new NotSupportedException();
        protected override List<Detection<double>> PostProcess(List<Tensor<double>> outputs,
            int imageWidth, int imageHeight, double confidenceThreshold, double nmsThreshold) =>
            throw new NotSupportedException();
        public override Task LoadWeightsAsync(string pathOrUrl, CancellationToken cancellationToken = default) =>
            throw new NotSupportedException();
        public override void SaveWeights(string path) => throw new NotSupportedException();
    }

    private sealed class TextDetectorProbe : TextDetectorBase<double>
    {
        public TextDetectorProbe(TextDetectionOptions<double> options) : base(options) { }
        public override string Name => nameof(TextDetectorProbe);
        public int ForwardCalls { get; private set; }
        public int[] LastInputShape { get; private set; } = Array.Empty<int>();
        public Tensor<double> Prepare(Tensor<double> image) => Preprocess(image);
        protected override List<Tensor<double>> Forward(Tensor<double> input)
        {
            ForwardCalls++;
            LastInputShape = input.Shape.ToArray();
            return new() { input };
        }
        protected override long GetHeadParameterCount() => 0;
        public override TextDetectionResult<double> Detect(Tensor<double> image) =>
            throw new NotSupportedException();
        public override TextDetectionResult<double> Detect(Tensor<double> image, double confidenceThreshold) =>
            throw new NotSupportedException();
        protected override List<TextRegion<double>> PostProcess(List<Tensor<double>> outputs,
            int imageWidth, int imageHeight, double confidenceThreshold) => throw new NotSupportedException();
        public override Task LoadWeightsAsync(string pathOrUrl, CancellationToken cancellationToken = default) =>
            throw new NotSupportedException();
        public override void SaveWeights(string path) => throw new NotSupportedException();
    }
}
