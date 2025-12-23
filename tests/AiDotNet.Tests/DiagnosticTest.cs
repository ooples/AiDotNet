using System;
using AiDotNet.ContinualLearning.Config;
using AiDotNet.Tensors.Helpers;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tests;

public class DiagnosticTest
{
    private readonly ITestOutputHelper _output;

    public DiagnosticTest(ITestOutputHelper output)
    {
        _output = output;
    }

    [Fact]
    public void DiagnoseNumericOperations()
    {
        // Test MathHelper directly
        var numOps = MathHelper.GetNumericOperations<double>();
        _output.WriteLine($"NumOps type: {numOps.GetType().FullName}");

        var result = numOps.FromDouble(0.001);
        _output.WriteLine($"FromDouble(0.001) = {result}");

        var result2 = numOps.FromDouble(1000.0);
        _output.WriteLine($"FromDouble(1000.0) = {result2}");

        // Test through config
        var config = new ContinualLearnerConfig<double>();
        var learningRate = config.LearningRate;
        _output.WriteLine($"LearningRate = {learningRate}");

        var ewcLambda = config.EwcLambda;
        _output.WriteLine($"EwcLambda = {ewcLambda}");

        Assert.Equal(0.001, result);
    }

    [Fact]
    public void DiagnoseIsValid()
    {
        var config = new ContinualLearnerConfig<double>();

        _output.WriteLine($"LearningRate: {config.LearningRate}");
        _output.WriteLine($"EpochsPerTask: {config.EpochsPerTask}");
        _output.WriteLine($"BatchSize: {config.BatchSize}");
        _output.WriteLine($"MemorySize: {config.MemorySize}");
        _output.WriteLine($"EwcLambda: {config.EwcLambda}");
        _output.WriteLine($"FisherSamples: {config.FisherSamples}");
        _output.WriteLine($"DistillationTemperature: {config.DistillationTemperature}");
        _output.WriteLine($"GemMemoryStrength: {config.GemMemoryStrength}");
        _output.WriteLine($"PackNetPruneRatio: {config.PackNetPruneRatio}");
        _output.WriteLine($"BiCValidationFraction: {config.BiCValidationFraction}");

        // Test each validation step - properties are non-nullable with defaults from constructor
        _output.WriteLine($"LR check (should be > 0): {Convert.ToDouble(config.LearningRate) > 0}");
        _output.WriteLine($"Epochs check (should be > 0): {config.EpochsPerTask > 0}");
        _output.WriteLine($"BatchSize check (should be > 0): {config.BatchSize > 0}");
        _output.WriteLine($"MemorySize check (should be >= 0): {config.MemorySize >= 0}");
        _output.WriteLine($"EwcLambda check (should be >= 0): {Convert.ToDouble(config.EwcLambda) >= 0}");
        _output.WriteLine($"FisherSamples check (should be > 0): {config.FisherSamples > 0}");
        _output.WriteLine($"DistillationTemp check (should be > 0): {Convert.ToDouble(config.DistillationTemperature) > 0}");

        bool isValid = config.IsValid();
        _output.WriteLine($"IsValid() result: {isValid}");

        Assert.True(isValid, "Default config should be valid");
    }
}
