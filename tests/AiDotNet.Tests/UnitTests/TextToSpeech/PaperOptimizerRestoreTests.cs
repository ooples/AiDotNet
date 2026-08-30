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

        object optimizer = GetPrivateMember(target, "_optimizer");
        object optimizerOptions = GetPrivateMember(optimizer, "_options");
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

    /// <summary>
    /// Reads a non-public member by name, whether it is stored as a field or computed by a property.
    /// </summary>
    /// <remarks>
    /// This used to look only for a field. Optimizers no longer keep a private typed copy of their
    /// options - <c>_options</c> is now a computed property over <c>OptimizerBase.Options</c>, so
    /// there is one instance rather than two to keep in step - and a field-only lookup failed with
    /// "Field '_options' was not found." What this test asserts is the VALUE the optimizer was
    /// restored with, which is unchanged by where that value is stored.
    /// </remarks>
    private static object GetPrivateMember(object instance, string name)
    {
        for (Type? type = instance.GetType(); type is not null; type = type.BaseType)
        {
            FieldInfo? field = type.GetField(name, BindingFlags.Instance | BindingFlags.NonPublic);
            if (field is not null)
                return field.GetValue(instance)
                    ?? throw new InvalidOperationException($"Member '{name}' was null.");

            PropertyInfo? property = type.GetProperty(name, BindingFlags.Instance | BindingFlags.NonPublic);
            if (property is not null)
                return property.GetValue(instance)
                    ?? throw new InvalidOperationException($"Member '{name}' was null.");
        }
        throw new InvalidOperationException($"Member '{name}' was not found.");
    }
}
