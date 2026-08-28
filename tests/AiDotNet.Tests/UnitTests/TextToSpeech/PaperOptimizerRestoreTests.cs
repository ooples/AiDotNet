using System.Reflection;
using AiDotNet.Enums;
using AiDotNet.Helpers;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using AiDotNet.TextToSpeech.EndToEnd;
using Xunit;

namespace AiDotNet.Tests.UnitTests.TextToSpeech;

public sealed class PaperOptimizerRestoreTests
{
    [Fact]
    public void Vits_family_rebuilds_generated_optimizer_from_restored_options()
    {
        AssertRestoredOptimizer(
            options => new VITS<double>(CreateArchitecture(), options),
            new VITSOptions(),
            learningRate: 7e-5);
        AssertRestoredOptimizer(
            options => new VITS2<double>(CreateArchitecture(), options),
            new VITS2Options(),
            learningRate: 8e-5);
        AssertRestoredOptimizer(
            options => new YourTTS<double>(CreateArchitecture(), options),
            new YourTTSOptions(),
            learningRate: 9e-5);
    }

    private static void AssertRestoredOptimizer<TOptions>(
        Func<TOptions, NeuralNetworkBase<double>> create,
        TOptions sourceOptions,
        double learningRate)
        where TOptions : EndToEndTtsOptions
    {
        using var source = create(sourceOptions);
        sourceOptions.LearningRate = learningRate;
        using var target = create((TOptions)Activator.CreateInstance(typeof(TOptions))!);

        using (ModelPersistenceGuard.InternalOperation())
        {
            target.Deserialize(source.Serialize());
        }

        object optimizer = GetPrivateField(target, "_optimizer");
        object optimizerOptions = GetPrivateField(optimizer, "_options");
        var rateProperty = optimizerOptions.GetType().GetProperty("InitialLearningRate")
            ?? throw new InvalidOperationException("AdamW options do not expose InitialLearningRate.");
        Assert.Equal(learningRate, Assert.IsType<double>(rateProperty.GetValue(optimizerOptions)), 12);
    }

    private static NeuralNetworkArchitecture<double> CreateArchitecture()
        => new(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 4,
            outputSize: 4,
            layers: [new DenseLayer<double>(4)]);

    private static object GetPrivateField(object instance, string name)
    {
        for (Type? type = instance.GetType(); type is not null; type = type.BaseType)
        {
            FieldInfo? field = type.GetField(name, BindingFlags.Instance | BindingFlags.NonPublic);
            if (field is not null)
                return field.GetValue(instance)
                    ?? throw new InvalidOperationException($"Field '{name}' was null.");
        }
        throw new InvalidOperationException($"Field '{name}' was not found.");
    }
}
