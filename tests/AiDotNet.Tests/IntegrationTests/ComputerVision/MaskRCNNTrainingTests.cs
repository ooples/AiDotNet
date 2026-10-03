using AiDotNet.ComputerVision.Detection;
using AiDotNet.ComputerVision.Segmentation.InstanceSegmentation;
using AiDotNet.Interfaces;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Helpers;
using System.Threading.Tasks;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.ComputerVision;

/// <summary>
/// Mask R-CNN as the paper defines it (He et al. 2017): a 2fc box branch with class-specific regression,
/// a 14x14-RoI mask head upsampled to 28x28 by a stride-2 transposed convolution, inference that masks
/// only the final detections, and a trainable multi-task loss L_cls + L_box + L_mask.
/// </summary>
public class MaskRCNNTrainingTests
{
    private const int ImageSize = 64;

    private static InstanceSegmentationOptions<double> Options(double confidence = 0.5) => new()
    {
        Architecture = InstanceSegmentationArchitecture.MaskRCNN,
        NumClasses = 2,
        InputSize = new[] { ImageSize, ImageSize },
        ConfidenceThreshold = confidence,
        MaxDetections = 5,
        RandomSeed = 7,
    };

    private static Tensor<double> Image()
    {
        var data = new double[3 * ImageSize * ImageSize];
        var rng = RandomHelper.CreateSeededRandom(42);
        for (int i = 0; i < data.Length; i++) data[i] = rng.NextDouble();
        return new Tensor<double>(new[] { 1, 3, ImageSize, ImageSize }, new Vector<double>(data));
    }

    /// <summary>A 24x24 square object of class 1 at (16, 16), with its exact mask.</summary>
    private static List<InstanceSegmentationTrainingTarget<double>> Square()
    {
        var mask = new Tensor<double>(new[] { ImageSize, ImageSize });
        for (int y = 16; y < 40; y++)
            for (int x = 16; x < 40; x++)
                mask[y, x] = 1.0;
        var box = DetectionTrainingTarget<double>.FromPixelXywh(1, 16, 16, 24, 24, ImageSize, ImageSize);
        return new List<InstanceSegmentationTrainingTarget<double>> { new(box, mask) };
    }

    private static Vector<double> ParametersOf(object component)
        => ((IParameterSource<double>)component).GetParameters();

    private static object Field(MaskRCNN<double> model, string name)
        => typeof(MaskRCNN<double>).GetField(name, System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)!
            .GetValue(model)!;

    [Fact(Timeout = 120000)]
    public async Task MaskHead_UpsamplesA14x14RoiTo28x28PerClassLogits()
    {
        await Task.Yield();
        var head = new MaskHead<double>(256, numClasses: 3);
        Assert.Equal(14, head.RoiSize);
        var logits = head.Forward(new Tensor<double>(new[] { 2, 256, 14, 14 }));
        Assert.Equal(new[] { 2, 3, 28, 28 }, logits.Shape.ToArray());
    }

    [Fact(Timeout = 120000)]
    public async Task ResampleMask_TakesTheObjectsMaskInsideTheRoi()
    {
        await Task.Yield();
        var mask = Square()[0].Mask;
        var inside = MaskRCNN<double>.ResampleMask(mask, new double[] { 16, 16, 40, 40 }, 28);
        Assert.All(inside, v => Assert.Equal(1.0, v));
        var outside = MaskRCNN<double>.ResampleMask(mask, new double[] { 44, 44, 60, 60 }, 28);
        Assert.All(outside, v => Assert.Equal(0.0, v));

        // A RoI twice the object's width: the left half is the object, the right half is not.
        var half = MaskRCNN<double>.ResampleMask(mask, new double[] { 16, 16, 64, 40 }, 28);
        Assert.Equal(1.0, half[5 * 28 + 3]);
        Assert.Equal(0.0, half[5 * 28 + 24]);
    }

    [Fact(Timeout = 300000)]
    public async Task Segment_MasksEachDetectionWithAFullImageBinaryMask()
    {
        await Task.Yield();
        // Threshold 0: every class-proposal pair is a candidate, so an untrained model still detects.
        var model = new MaskRCNN<double>(Options(confidence: 0.0));
        var result = model.Segment(Image());

        Assert.InRange(result.Instances.Count, 1, 5);
        foreach (var instance in result.Instances)
        {
            Assert.Equal(new[] { ImageSize, ImageSize }, instance.Mask.Shape.ToArray());
            Assert.All(instance.Mask.ToArray(), v => Assert.True(v == 0.0 || v == 1.0));
            Assert.InRange(instance.ClassId, 0, 1);
        }
    }

    [Fact(Timeout = 600000)]
    public async Task TrainInstances_UpdatesEveryBranchIncludingTheMaskHead()
    {
        await Task.Yield();
        var model = new MaskRCNN<double>(Options());
        Assert.True(model.ParameterCount > 0); // resolves every lazily shaped layer, the mask head included

        string[] branches = { "_rpn", "_boxFc1", "_boxFc2", "_classHead", "_boxRegressor", "_maskHead" };
        var before = branches.ToDictionary(b => b, b => ParametersOf(Field(model, b)).ToArray());

        model.TrainInstances(Image(), Square());

        Assert.True(!double.IsNaN(model.GetLastLoss()) && !double.IsInfinity(model.GetLastLoss()), $"loss was {model.GetLastLoss()}");
        Assert.True(model.GetLastLoss() > 0);
        foreach (var branch in branches)
        {
            var after = ParametersOf(Field(model, branch)).ToArray();
            Assert.Equal(before[branch].Length, after.Length);
            Assert.True(before[branch].Zip(after, (a, b) => a != b).Any(changed => changed),
                $"{branch} did not change after a training step");
        }
    }

    [Fact(Timeout = 900000)]
    public async Task TrainInstances_LowersTheLossOnARepeatedExample()
    {
        await Task.Yield();
        var model = new MaskRCNN<double>(Options());
        var image = Image();
        var targets = Square();
        var losses = new List<double>();
        for (int step = 0; step < 8; step++)
        {
            model.TrainInstances(image, targets);
            losses.Add(model.GetLastLoss());
        }

        Assert.All(losses, l => Assert.True(!double.IsNaN(l) && !double.IsInfinity(l)));
        Assert.True(losses.Skip(5).Average() < losses.Take(3).Average(),
            $"loss did not fall: {string.Join(", ", losses.Select(l => l.ToString("G4")))}");
    }

    [Fact(Timeout = 300000)]
    public async Task Clone_ReproducesThePrediction()
    {
        await Task.Yield();
        var model = new MaskRCNN<double>(Options());
        var image = Image();
        var expected = model.Predict(image).ToArray();
        Assert.True(model.ParameterCount > 0);

        var copy = (MaskRCNN<double>)model.Clone();
        Assert.Equal(model.ParameterCount, copy.ParameterCount);
        Assert.Equal(expected, copy.Predict(image).ToArray());
    }

    [Fact(Timeout = 300000)]
    public async Task Solov2AndYoloSeg_PredictThroughTheSharedBase()
    {
        await Task.Yield();
        var image = Image();
        var solo = new SOLOv2<double>(new InstanceSegmentationOptions<double>
        {
            Architecture = InstanceSegmentationArchitecture.SOLOv2, NumClasses = 2, InputSize = new[] { ImageSize, ImageSize }
        });
        Assert.True(solo.Predict(image).Length > 0);
        Assert.True(solo.ParameterCount > 0);

        var yolo = new YOLOSeg<double>(new InstanceSegmentationOptions<double>
        {
            Architecture = InstanceSegmentationArchitecture.YOLOSeg, NumClasses = 2, InputSize = new[] { ImageSize, ImageSize }
        });
        Assert.True(yolo.Predict(image).Length > 0);
        Assert.True(yolo.ParameterCount > 0);
    }
}
