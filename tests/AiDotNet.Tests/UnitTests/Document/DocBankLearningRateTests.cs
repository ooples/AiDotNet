using System;
using System.Threading.Tasks;
using AiDotNet.Document.Analysis.PageSegmentation;
using AiDotNet.Document.Options;
using AiDotNet.Enums;
using AiDotNet.LinearAlgebra;
using AiDotNet.NeuralNetworks;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Document;

/// <summary>
/// DocBankOptions.LearningRate overrides the paper recipe (AdamW at 5e-5, Li et al. 2020, Sec. 4)
/// and must reach the optimizer that actually trains the model.
/// </summary>
/// <remarks>
/// The tape-training path once built its own Adam at a hardcoded 1e-3, so the recipe and the option
/// were both inert: every configured rate left the trained weights bit-identical. Each case here
/// starts two models from the same weights and compares what the same training steps did to them.
/// </remarks>
public class DocBankLearningRateTests
{
    private const int ImageSize = 64;
    private const int Steps = 2;
    private const double PaperLearningRate = 5e-5;

    [Fact(Timeout = 300000)]
    public async Task UnsetRate_TrainsAtThePaperRate()
    {
        var initial = InitialParameters();

        var unset = TrainedParameters(initial, learningRate: null);
        var paper = TrainedParameters(initial, PaperLearningRate);

        Assert.NotEqual(0.0, Distance(unset, initial));
        Assert.Equal(0.0, Distance(unset, paper));
    }

    [Fact(Timeout = 300000)]
    public async Task ConfiguredRate_ReachesTraining()
    {
        var initial = InitialParameters();

        var paper = TrainedParameters(initial, learningRate: null, steps: 1);
        var small = TrainedParameters(initial, 1e-4, steps: 1);
        var large = TrainedParameters(initial, 1e-3, steps: 1);

        // After one step from the same weights on the same batch, everything the optimizer does not
        // drive (normalization statistics) is identical across the three runs, so the gap to the
        // paper-rate run is the optimizer's update alone. AdamW's first update moves each weight by
        // about the learning rate, so that gap grows with (rate - 5e-5): ~19x from 1e-4 to 1e-3. A
        // rate that never reached the optimizer would leave both gaps zero.
        double smallGap = Distance(small, paper);
        double largeGap = Distance(large, paper);
        Assert.NotEqual(0.0, smallGap);
        Assert.True(largeGap > 5 * smallGap,
            $"A learning rate of 1e-3 moved the weights {largeGap:G4} from the paper-rate run, "
            + $"against {smallGap:G4} for 1e-4.");
    }

    private static NeuralNetworkArchitecture<double> Architecture() => new(
        inputType: InputType.ThreeDimensional,
        taskType: NeuralNetworkTaskType.Regression,
        inputHeight: ImageSize,
        inputWidth: ImageSize,
        inputDepth: 3,
        outputSize: 13);

    private static DocBank<double> Build(double? learningRate) =>
        new(Architecture(), options: new DocBankOptions { ImageSize = ImageSize, LearningRate = learningRate });

    private static Vector<double> InitialParameters()
    {
        using var model = Build(learningRate: null);
        return model.GetParameters();
    }

    private static Vector<double> TrainedParameters(
        Vector<double> initial, double? learningRate, int steps = Steps)
    {
        using var model = Build(learningRate);
        model.SetParameters(initial);

        var input = Image();
        var target = Target(model, input);
        model.SetTrainingMode(true);
        for (int i = 0; i < steps; i++) model.Train(input, target);
        model.SetTrainingMode(false);

        return model.GetParameters();
    }

    private static Tensor<double> Image()
    {
        int total = 3 * ImageSize * ImageSize;
        var data = new Vector<double>(total);
        for (int i = 0; i < total; i++) data[i] = (i % 17) / 17.0;
        return new Tensor<double>(new[] { 1, 3, ImageSize, ImageSize }, data);
    }

    /// <summary>A fixed one-hot labelling shaped like the model's own output.</summary>
    private static Tensor<double> Target(DocBank<double> model, Tensor<double> input)
    {
        var target = model.Predict(input).Clone();
        int classes = target.Shape[target.Rank - 1];
        var values = target.AsWritableSpan();
        values.Clear();
        for (int i = 0; i + classes <= values.Length; i += classes) values[i + (i / classes) % classes] = 1.0;
        return target;
    }

    private static double Distance(Vector<double> a, Vector<double> b)
    {
        Assert.Equal(a.Length, b.Length);
        double sum = 0;
        for (int i = 0; i < a.Length; i++) sum += (a[i] - b[i]) * (a[i] - b[i]);
        return Math.Sqrt(sum);
    }
}
