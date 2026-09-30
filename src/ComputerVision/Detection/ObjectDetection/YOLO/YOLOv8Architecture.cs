using System.IO;
using AiDotNet.Attributes;
using AiDotNet.ComputerVision.Detection.Backbones;
using AiDotNet.ComputerVision.Detection.Necks;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LossFunctions;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using System.Linq;

namespace AiDotNet.ComputerVision.Detection.ObjectDetection.YOLO;

/// <summary>
/// Ultralytics YOLO compound scaling (yolov8.yaml, yolo11.yaml): depth and width multipliers and the
/// channel cap applied before the width multiplier.
/// </summary>
internal readonly record struct YoloScale(double Depth, double Width, int MaxChannels, bool ForcesC3k = false)
{
    internal static YoloScale ForV8(ModelSize size) => size switch
    {
        ModelSize.Nano => new(0.33, 0.25, 1024),
        ModelSize.Small => new(0.33, 0.50, 1024),
        ModelSize.Medium => new(0.67, 0.75, 768),
        ModelSize.Large => new(1.00, 1.00, 512),
        ModelSize.XLarge => new(1.00, 1.25, 512),
        _ => throw new ArgumentOutOfRangeException(nameof(size), size, "YOLOv8 publishes no scale for this model size."),
    };

    // yolo11.yaml scales; m, l and x use C3k inner blocks in every C3k2 (ultralytics parse_model).
    internal static YoloScale ForV11(ModelSize size) => size switch
    {
        ModelSize.Nano => new(0.50, 0.25, 1024),
        ModelSize.Small => new(0.50, 0.50, 1024),
        ModelSize.Medium => new(0.50, 1.00, 512, true),
        ModelSize.Large => new(1.00, 1.00, 512, true),
        ModelSize.XLarge => new(1.00, 1.50, 512, true),
        _ => throw new ArgumentOutOfRangeException(nameof(size), size, "YOLO11 publishes no scale for this model size."),
    };

    /// <summary>Output channels of a layer declared with <paramref name="channels"/> at full width.</summary>
    internal int Channels(int channels) => Math.Max(1, (int)Math.Ceiling(Math.Min(channels, MaxChannels) * Width / 8.0) * 8);

    /// <summary>Repeats of a block declared with <paramref name="repeats"/> at full depth.</summary>
    internal int Repeats(int repeats) => Math.Max(1, (int)Math.Round(repeats * Depth));
}

/// <summary>
/// YOLOv8's backbone: Conv stem, four stride-2 Conv + C2f stages, then SPPF (Jocher et al. 2023).
/// </summary>
/// <remarks>
/// P3, P4 and P5 (strides 8, 16, 32) feed the neck. Every Conv is convolution + batch norm + SiLU, and the
/// backbone's C2f bottlenecks use their residual shortcut. The previous YOLOv8 ran a YOLOv4/v5-style CSP
/// backbone without batch norm and without SPPF.
/// </remarks>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.FeatureExtraction)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Ultralytics YOLOv8", "https://github.com/ultralytics/ultralytics", Year = 2023,
    Authors = "Glenn Jocher, Ayush Chaurasia, Jing Qiu")]
[ArchitectureFromPaper("https://github.com/ultralytics/ultralytics",
    "YOLOv8 is published as code; this is the backbone section of its yolov8.yaml.")]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width,
    Direction = TensorLayoutDirection.Input, BatchOptional = true)]
[TensorLayout(TensorAxis.Batch, TensorAxis.Channels, TensorAxis.Height, TensorAxis.Width,
    Direction = TensorLayoutDirection.Output, BatchOptional = true)]
public partial class YOLOv8Backbone<T> : YoloStagedBackboneBase<T>
{
    /// <summary>Creates the backbone.</summary>
    /// <param name="options">Model size and input channels; defaults to Nano over three channels.</param>
    public YOLOv8Backbone(YoloBackboneOptions? options = null)
        : base($"YOLOv8Backbone-{(options ??= new YoloBackboneOptions()).Size}", options.InChannels)
    {
        options.Validate();
        var s = YoloScale.ForV8(options.Size);
        int c64 = s.Channels(64), c128 = s.Channels(128), c256 = s.Channels(256), c512 = s.Channels(512), c1024 = s.Channels(1024);
        AddStage(new YoloConv<T>(c64, 3, 2));                       // 0  P1/2
        AddStage(new YoloConv<T>(c128, 3, 2));                      // 1  P2/4
        AddStage(new C2fBlock<T>(c128, s.Repeats(3), true));        // 2
        AddStage(new YoloConv<T>(c256, 3, 2));                      // 3  P3/8
        AddStage(new C2fBlock<T>(c256, s.Repeats(6), true));        // 4  -> P3
        AddStage(new YoloConv<T>(c512, 3, 2));                      // 5  P4/16
        AddStage(new C2fBlock<T>(c512, s.Repeats(6), true));        // 6  -> P4
        AddStage(new YoloConv<T>(c1024, 3, 2));                     // 7  P5/32
        AddStage(new C2fBlock<T>(c1024, s.Repeats(3), true));       // 8
        AddStage(new SPPFLayer<T>(c1024, c1024, 5));                // 9  -> P5
        CompleteStages(new[] { 4, 6, 9 }, new[] { c256, c512, c1024 });
    }
}

/// <summary>
/// YOLOv8's PAN-FPN neck: top-down nearest upsampling with concatenation and C2f (no shortcut), then
/// bottom-up stride-2 Conv with concatenation and C2f. Outputs P3, P4 and P5 at their own widths.
/// </summary>
[ModelDomain(ModelDomain.Vision)]
[ModelCategory(ModelCategory.NeuralNetwork)]
[ModelTask(ModelTask.FeatureExtraction)]
[ModelComplexity(ModelComplexity.Medium)]
[ModelInput(typeof(Tensor<>), typeof(Tensor<>))]
[ResearchPaper("Ultralytics YOLOv8", "https://github.com/ultralytics/ultralytics", Year = 2023,
    Authors = "Glenn Jocher, Ayush Chaurasia, Jing Qiu")]
[ArchitectureFromPaper("https://github.com/ultralytics/ultralytics",
    "YOLOv8 is published as code; this is the head (PAN-FPN) section of its yolov8.yaml.")]
public partial class YOLOv8Neck<T> : YoloPanNeckBase<T>
{
    /// <summary>Creates the neck for a model size.</summary>
    public YOLOv8Neck(ModelSize size)
        : base("YOLOv8-PAN", BuildBlocks(size))
    {
    }

    private static YoloPanBlocks<T> BuildBlocks(ModelSize size)
    {
        var s = YoloScale.ForV8(size);
        int c256 = s.Channels(256), c512 = s.Channels(512), c1024 = s.Channels(1024);
        int n = s.Repeats(3);
        return new YoloPanBlocks<T>(
            topDown4: new C2fBlock<T>(c512, n, false),
            topDown3: new C2fBlock<T>(c256, n, false),
            down3: new YoloConv<T>(c256, 3, 2),
            bottomUp4: new C2fBlock<T>(c512, n, false),
            down4: new YoloConv<T>(c512, 3, 2),
            bottomUp5: new C2fBlock<T>(c1024, n, false),
            levelChannels: new[] { c256, c512, c1024 });
    }
}