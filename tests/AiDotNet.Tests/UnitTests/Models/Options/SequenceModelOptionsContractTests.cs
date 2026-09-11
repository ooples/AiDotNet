using System;
using System.Collections.Generic;
using System.Linq;
using System.Linq.Expressions;
using System.Reflection;
using AiDotNet.Models.Options;
using AiDotNet.NeuralNetworks.Options;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Models.Options;

/// <summary>Exercises real options without allocating a model or relying on generated model fixtures.</summary>
public class SequenceModelOptionsContractTests
{
    private static readonly Type[] SequenceTypes =
    {
        typeof(EagleOptions), typeof(FalconMambaOptions), typeof(FinchOptions),
        typeof(GatedDeltaNetOptions), typeof(GLAOptions), typeof(GriffinOptions),
        typeof(HawkOptions), typeof(JambaOptions), typeof(Mamba2Options), typeof(MambaOptions),
        typeof(RecurrentGemmaOptions), typeof(RWKV4Options), typeof(RWKV7Options),
        typeof(SambaOptions), typeof(XLSTMOptions), typeof(Zamba2Options), typeof(ZambaOptions)
    };

    // Sharing state-space settings does not make an image model a tokenizer-backed language model.
    private static readonly Type[] ImageSequenceTypes = { typeof(VisionMambaOptions) };

    public static IEnumerable<object[]> CopyTypes => SequenceTypes
        .Concat(new[] { typeof(NeuralNetworkOptions), typeof(DocumentNeuralNetworkOptions) })
        .Select(type => new object[] { type });

    [Fact]
    public void SequenceRoster_CoversEveryConcreteSequenceOptionsType()
    {
        var actual = typeof(SequenceModelOptions).Assembly.GetTypes()
            .Where(type => !type.IsAbstract && typeof(SequenceModelOptions).IsAssignableFrom(type))
            .OrderBy(type => type.FullName, StringComparer.Ordinal);
        Assert.Equal(17, SequenceTypes.Length);
        Assert.Equal(typeof(VisionMambaOptions), Assert.Single(ImageSequenceTypes));
        Assert.Equal(SequenceTypes.Concat(ImageSequenceTypes)
            .OrderBy(type => type.FullName, StringComparer.Ordinal), actual);
    }

    [Fact]
    public void ImageSequenceRoster_OwnsImageGeometryWithoutLanguageRequirements()
    {
        var options = new VisionMambaOptions();
        Assert.Equal(typeof(VisionMambaOptions), Assert.Single(ImageSequenceTypes));
        Assert.IsAssignableFrom<SequenceModelOptions>(options);
        Assert.Equal(new[] { 224, 224, 16, 3, 192, 4, 10, 16 },
            new[] { options.ImageHeight, options.ImageWidth, options.PatchSize, options.Channels,
                options.ModelDimension, options.NumLayers, options.NumClasses, options.StateDimension });
        Assert.Equal(0, options.VocabSize);
        Assert.Equal(0, options.MaxSequenceLength);
        options.Validate();
    }

    [Theory]
    [MemberData(nameof(CopyTypes))]
    public void CopyConstructor_PreservesEveryDeclaredAndInheritedProperty(Type optionsType)
    {
        var original = Create(optionsType);
        var properties = WritableProperties(optionsType);
        Assert.NotEmpty(properties);
        for (var index = 0; index < properties.Length; index++)
        {
            var property = properties[index];
            var sentinel = Sentinel(property.PropertyType, property.GetValue(original), index);
            Assert.NotEqual(property.GetValue(original), sentinel);
            property.SetValue(original, sentinel);
        }

        var copy = CopyConstructor(optionsType).Invoke(new[] { original });
        Assert.NotSame(original, copy);
        // Collect every omitted property, not just the first reset-to-default value.
        var mismatches = properties.Where(property =>
            !Equals(property.GetValue(original), property.GetValue(copy)))
            .Select(property => $"{optionsType.Name}.{property.Name}").ToArray();
        Assert.True(mismatches.Length == 0, "Copy lost: " + string.Join(", ", mismatches));

        foreach (var property in properties)
        {
            var copiedValue = property.GetValue(copy);
            property.SetValue(original, Sentinel(property.PropertyType, copiedValue, 100));
            Assert.Equal(copiedValue, property.GetValue(copy));
        }
    }

    [Theory]
    [MemberData(nameof(CopyTypes))]
    public void CopyConstructor_PreservesDefaultsIncludingNullableState(Type optionsType)
    {
        var original = Create(optionsType);
        var copy = CopyConstructor(optionsType).Invoke(new[] { original });
        Assert.All(WritableProperties(optionsType), property =>
            Assert.Equal(property.GetValue(original), property.GetValue(copy)));
    }

    [Theory]
    [MemberData(nameof(CopyTypes))]
    public void CopyConstructor_RejectsNullAtTheSharedBaseBoundary(Type optionsType)
    {
        var exception = Assert.Throws<TargetInvocationException>(() =>
            CopyConstructor(optionsType).Invoke(new object?[] { null }));
        var argument = Assert.IsType<ArgumentNullException>(exception.InnerException);
        Assert.Equal("other", argument.ParamName);
    }

    [Fact]
    public void Finch_DefaultLearningRateRetainsThePreMigrationInitializer()
    {
        // 671836a347^ has an empty ctor and this 3e-4 initializer; the migration
        // accidentally overwrote it with an unused model-constructor scalar default.
        Assert.Equal(3e-4, new FinchOptions().LearningRate);
    }

    [Theory]
    [InlineData(typeof(EagleOptions), 65536, 256, 4, 8, 0, 512, 0, 0, 0.0)]
    [InlineData(typeof(FalconMambaOptions), 65024, 256, 4, 0, 16, 512, 0, 2, 0.0)]
    [InlineData(typeof(FinchOptions), 65536, 256, 4, 8, 0, 512, 0, 0, 0.0)]
    [InlineData(typeof(GatedDeltaNetOptions), 50277, 256, 4, 8, 0, 512, 0, 0, 0.0)]
    [InlineData(typeof(GLAOptions), 50277, 256, 4, 8, 0, 512, 0, 0, 0.0)]
    [InlineData(typeof(GriffinOptions), 256000, 2048, 24, 0, 0, 2048, 0, 0, 0.0)]
    [InlineData(typeof(HawkOptions), 256000, 2048, 24, 0, 0, 2048, 0, 0, 0.0)]
    [InlineData(typeof(JambaOptions), 65536, 256, 8, 0, 16, 512, 8, 0, 0.0)]
    [InlineData(typeof(Mamba2Options), 50277, 256, 4, 8, 64, 512, 0, 0, 0.0)]
    [InlineData(typeof(MambaOptions), 50277, 256, 4, 0, 16, 512, 0, 2, 0.0)]
    [InlineData(typeof(RecurrentGemmaOptions), 256000, 256, 4, 0, 0, 512, 0, 0, 0.0)]
    [InlineData(typeof(RWKV4Options), 50277, 256, 4, 0, 0, 512, 0, 0, 0.0)]
    [InlineData(typeof(RWKV7Options), 65536, 256, 4, 4, 0, 512, 0, 0, 3.5)]
    [InlineData(typeof(SambaOptions), 32000, 256, 8, 0, 16, 512, 2, 0, 0.0)]
    [InlineData(typeof(XLSTMOptions), 50277, 256, 4, 8, 0, 512, 0, 0, 0.0)]
    [InlineData(typeof(Zamba2Options), 32000, 3584, 81, 32, 64, 4096, 6, 0, 0.0)]
    [InlineData(typeof(ZambaOptions), 32000, 3712, 76, 0, 16, 4096, 6, 0, 0.0)]
    public void ShippedSequenceDefaults_RemainUnchanged(Type optionsType, int vocabulary, int width,
        int layers, int heads, int state, int sequenceLength, int attentionInterval, int expansion, double ffn)
    {
        var options = Assert.IsAssignableFrom<SequenceModelOptions>(Create(optionsType));
        Assert.Equal(new[] { vocabulary, width, layers, heads, state, sequenceLength, attentionInterval, expansion },
            new[] { options.VocabSize, options.ModelDimension, options.NumLayers, options.NumHeads,
                options.StateDimension, options.MaxSequenceLength, options.AttentionInterval, options.ExpandFactor });
        Assert.Equal(ffn, options.FfnMultiplier);
        Assert.Null(options.Seed);
        Assert.Null(options.EncoderLayerCount);
    }

    [Fact]
    public void ShippedOptimizerDefaults_RemainUnchanged()
    {
        var griffin = new GriffinOptions();
        var hawk = new HawkOptions();
        var gemma = new RecurrentGemmaOptions();
        var finch = new FinchOptions();
        Assert.Equal(2560, griffin.RecurrenceDimension);
        Assert.Equal(2560, hawk.RecurrenceDimension);
        Assert.True(gemma.ScaleEmbeddingsBySqrtWidth);
        Assert.Equal(new[] { 1e-4, 0.01, 0.9, 0.999, 1e-8, 1.0 },
            new[] { griffin.LearningRate, griffin.WeightDecay, griffin.Beta1, griffin.Beta2, griffin.Epsilon, griffin.MaxGradientNorm });
        Assert.Equal(new[] { 1e-4, 0.01, 0.9, 0.999, 1e-8, 1.0 },
            new[] { hawk.LearningRate, hawk.WeightDecay, hawk.Beta1, hawk.Beta2, hawk.Epsilon, hawk.MaxGradientNorm });
        Assert.Equal(new[] { 1e-4, 0.01, 0.9, 0.999, 1e-8, 1.0 },
            new[] { gemma.LearningRate, gemma.WeightDecay, gemma.Beta1, gemma.Beta2, gemma.Epsilon, gemma.MaxGradientNorm });
        Assert.True(griffin.EnableGradientClipping);
        Assert.True(hawk.EnableGradientClipping);
        Assert.True(gemma.EnableGradientClipping);
        Assert.Equal(new[] { 2e-5, 0.9, 0.99, 0.001, 1.0 },
            new[] { finch.MinLearningRate, finch.Beta1, finch.Beta2, finch.WeightDecay, finch.MaxGradientNorm });
        Assert.True(finch.EnableGradientClipping);
        Assert.Equal(3e-4, new GLAOptions().LearningRate);
        Assert.Equal(3e-4, new GatedDeltaNetOptions().LearningRate);
        Assert.Equal(1e-3, new XLSTMOptions().LearningRate);
    }

    public static IEnumerable<object[]> RequiredIntegers()
    {
        foreach (var type in SequenceTypes)
        {
            foreach (var property in new[]
            {
                Property<SequenceModelOptions, int>(options => options.VocabSize),
                Property<SequenceModelOptions, int>(options => options.ModelDimension),
                Property<SequenceModelOptions, int>(options => options.NumLayers),
                Property<SequenceModelOptions, int>(options => options.MaxSequenceLength)
            })
            {
                yield return new object[] { type, property, 0 };
                yield return new object[] { type, property, -1 };
            }
        }

        foreach (var type in new[] { typeof(JambaOptions), typeof(SambaOptions), typeof(ZambaOptions), typeof(Zamba2Options) })
        {
            var property = Property<SequenceModelOptions, int>(options => options.AttentionInterval);
            yield return new object[] { type, property, 0 };
            yield return new object[] { type, property, -1 };
        }
    }

    [Theory]
    [MemberData(nameof(RequiredIntegers))]
    public void RequiredDimension_RejectsExactlyTheInvalidProperty(Type optionsType, PropertyInfo property, int value)
    {
        AssertInvalid(optionsType, property, value);
    }

    public static IEnumerable<object[]> InvalidDoubles()
    {
        var positive = new[] { 0.0, -1.0, double.NaN, double.PositiveInfinity, double.NegativeInfinity };
        foreach (var row in InvalidValues<GLAOptions>(options => options.LearningRate, positive)) yield return row;
        foreach (var row in InvalidValues<GatedDeltaNetOptions>(options => options.LearningRate, positive)) yield return row;
        foreach (var row in InvalidValues<XLSTMOptions>(options => options.LearningRate, positive)) yield return row;
        foreach (var row in InvalidValues<RWKV7Options>(options => options.FfnMultiplier, positive)) yield return row;

        foreach (var row in AdamWInvalidValues<GriffinOptions>(
            options => options.LearningRate, options => options.WeightDecay, options => options.Beta1,
            options => options.Beta2, options => options.Epsilon, options => options.MaxGradientNorm)) yield return row;
        foreach (var row in AdamWInvalidValues<HawkOptions>(
            options => options.LearningRate, options => options.WeightDecay, options => options.Beta1,
            options => options.Beta2, options => options.Epsilon, options => options.MaxGradientNorm)) yield return row;
        foreach (var row in AdamWInvalidValues<RecurrentGemmaOptions>(
            options => options.LearningRate, options => options.WeightDecay, options => options.Beta1,
            options => options.Beta2, options => options.Epsilon, options => options.MaxGradientNorm)) yield return row;
    }

    [Theory]
    [MemberData(nameof(InvalidDoubles))]
    public void ConsumedNumericOption_RejectsExactlyTheInvalidProperty(Type optionsType, PropertyInfo property, double value)
    {
        AssertInvalid(optionsType, property, value);
    }

    [Theory]
    [InlineData(0.0)]
    [InlineData(-1.0)]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    public void DisabledClipping_DoesNotRequireAnUnusedThreshold(double threshold)
    {
        new GriffinOptions { EnableGradientClipping = false, MaxGradientNorm = threshold }.Validate();
        new HawkOptions { EnableGradientClipping = false, MaxGradientNorm = threshold }.Validate();
        new RecurrentGemmaOptions { EnableGradientClipping = false, MaxGradientNorm = threshold }.Validate();
    }

    [Theory]
    [InlineData(0.0)]
    [InlineData(0.99999999999999989)]
    public void AdamW_AllowsZeroDecayAndBothValidBetaEndpoints(double beta)
    {
        new GriffinOptions { Beta1 = beta, Beta2 = beta, WeightDecay = 0.0 }.Validate();
        new HawkOptions { Beta1 = beta, Beta2 = beta, WeightDecay = 0.0 }.Validate();
        new RecurrentGemmaOptions { Beta1 = beta, Beta2 = beta, WeightDecay = 0.0 }.Validate();
    }

    [Fact]
    public void ModelSpecificValidation_DoesNotRequireUnusedSharedProperties()
    {
        // A pure state-space model has no attention heads/interval or FFN ratio.
        new MambaOptions { NumHeads = 0, AttentionInterval = 0, FfnMultiplier = 0 }.Validate();
        new RWKV4Options { NumHeads = 0, StateDimension = 0, ExpandFactor = 0 }.Validate();
        // An interval larger than the stack is not zero/negative and stays supported.
        new JambaOptions { AttentionInterval = int.MaxValue }.Validate();
        new SambaOptions { AttentionInterval = int.MaxValue }.Validate();
        new ZambaOptions { AttentionInterval = int.MaxValue }.Validate();
        new Zamba2Options { AttentionInterval = int.MaxValue }.Validate();
    }

    private static IEnumerable<object[]> AdamWInvalidValues<TOptions>(
        Expression<Func<TOptions, double>> learningRate,
        Expression<Func<TOptions, double>> weightDecay,
        Expression<Func<TOptions, double>> beta1,
        Expression<Func<TOptions, double>> beta2,
        Expression<Func<TOptions, double>> epsilon,
        Expression<Func<TOptions, double>> maxGradientNorm)
    {
        var positive = new[] { 0.0, -1.0, double.NaN, double.PositiveInfinity, double.NegativeInfinity };
        foreach (var expression in new[] { learningRate, epsilon, maxGradientNorm })
            foreach (var row in InvalidValues(expression, positive)) yield return row;
        foreach (var row in InvalidValues(weightDecay, new[] { -1.0, double.NaN, double.PositiveInfinity, double.NegativeInfinity }))
            yield return row;
        foreach (var expression in new[] { beta1, beta2 })
            foreach (var row in InvalidValues(expression, new[] { -1.0, 1.0, 2.0, double.NaN, double.PositiveInfinity, double.NegativeInfinity }))
                yield return row;
    }

    private static IEnumerable<object[]> InvalidValues<TOptions>(Expression<Func<TOptions, double>> expression, double[] values)
    {
        var property = Property(expression);
        return values.Select(value => new object[] { typeof(TOptions), property, value });
    }

    private static PropertyInfo Property<TOptions, TValue>(Expression<Func<TOptions, TValue>> expression)
    {
        var member = Assert.IsAssignableFrom<MemberExpression>(expression.Body);
        return Assert.IsAssignableFrom<PropertyInfo>(member.Member);
    }

    private static void AssertInvalid(Type optionsType, PropertyInfo property, object value)
    {
        var options = Create(optionsType);
        var validate = optionsType.GetMethod(nameof(GLAOptions.Validate),
            BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic,
            binder: null, types: Type.EmptyTypes, modifiers: null)
            ?? throw new InvalidOperationException($"{optionsType.Name} has no Validate method.");
        // Prove the fixture starts valid: a different default cannot cause a false pass.
        validate.Invoke(options, Array.Empty<object>());
        property.SetValue(options, value);
        var exception = Assert.Throws<TargetInvocationException>(() => validate.Invoke(options, Array.Empty<object>()));
        var argument = Assert.IsAssignableFrom<ArgumentException>(exception.InnerException);
        Assert.Equal("options", argument.ParamName);
        Assert.Contains($"{optionsType.Name}.{property.Name}", argument.Message);
    }

    private static object Create(Type type) => Activator.CreateInstance(type)
        ?? throw new InvalidOperationException($"Could not instantiate {type.Name}.");

    private static ConstructorInfo CopyConstructor(Type type)
    {
        var constructor = type.GetConstructor(BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic,
            binder: null, types: new[] { type }, modifiers: null)
            ?? throw new InvalidOperationException($"{type.Name} has no copy constructor.");
        Assert.True(constructor.IsPublic || (type == typeof(DocumentNeuralNetworkOptions) && constructor.IsFamily));
        return constructor;
    }

    private static PropertyInfo[] WritableProperties(Type type) => type.GetProperties(BindingFlags.Instance | BindingFlags.Public)
        .Where(property => property.GetMethod?.IsPublic == true && property.SetMethod?.IsPublic == true)
        .OrderBy(property => property.Name, StringComparer.Ordinal).ToArray();

    private static object Sentinel(Type type, object? current, int index)
    {
        if (type == typeof(int) || type == typeof(int?)) return 370 + index;
        if (type == typeof(double)) return 17.125 + index;
        if (type == typeof(bool)) return !Equals(current, true);
        throw new InvalidOperationException($"Add a non-default sentinel for new option value type {type}.");
    }
}
