using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using AiDotNet.Finance.Forecasting.Transformers;
using AiDotNet.Finance.Interfaces;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tensors;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Finance;

internal static class FinanceModelTestFactory
{
    private const string FinanceNamespacePrefix = "AiDotNet.Finance";

    internal static IReadOnlyList<Type> GetFinanceModelTypes<T>()
    {
        var numericType = typeof(T);
        var tensorType = typeof(Tensor<>).MakeGenericType(numericType);
        var modelInterface = typeof(IFullModel<,,>).MakeGenericType(numericType, tensorType, tensorType);

        return GetFinanceTypes()
            .Select(type => type.IsGenericTypeDefinition ? type.MakeGenericType(numericType) : type)
            .Where(type => modelInterface.IsAssignableFrom(type))
            .OrderBy(type => type.FullName, StringComparer.Ordinal)
            .ToList();
    }

    internal static IReadOnlyList<Type> GetTradingAgentTypes<T>()
    {
        var numericType = typeof(T);
        var agentInterface = typeof(ITradingAgent<>).MakeGenericType(numericType);

        return GetFinanceTypes()
            .Select(type => type.IsGenericTypeDefinition ? type.MakeGenericType(numericType) : type)
            .Where(type => agentInterface.IsAssignableFrom(type))
            .OrderBy(type => type.FullName, StringComparer.Ordinal)
            .ToList();
    }

    internal static void RunFullModelSmokeTest<T>(Type modelType)
    {
        object? instance = null;
        object? clone = null;

        try
        {
            instance = CreateNativeModel<T>(modelType);
            Assert.NotNull(instance);

            var model = (IFullModel<T, Tensor<T>, Tensor<T>>)instance;

            if (model is IAutoMLModel<T, Tensor<T>, Tensor<T>> autoMl)
            {
                var suggestion = autoMl.SuggestNextTrialAsync().GetAwaiter().GetResult();
                Assert.NotNull(suggestion);
                return;
            }

            if (model is IFinancialNLPModel<T> nlpModel)
            {
                var input = FinanceTestHelpers.CreateTokenTensor<T>(
                    batchSize: 1,
                    sequenceLength: Math.Max(1, nlpModel.MaxSequenceLength),
                    vocabularySize: Math.Max(2, nlpModel.VocabularySize));

                var output = nlpModel.Predict(input);
                Assert.NotNull(output);

                if (nlpModel.SupportsTraining)
                {
                    nlpModel.Train(input, output);
                }

                var serialized = nlpModel.Serialize();
                clone = CreateNativeModel<T>(modelType);
                ((IFullModel<T, Tensor<T>, Tensor<T>>)clone).Deserialize(serialized);
                return;
            }

            if (model is IFinancialModel<T> financialModel)
            {
                var input = FinanceTestHelpers.CreateTimeSeriesInput<T>(
                    batchSize: 1,
                    sequenceLength: Math.Max(1, financialModel.SequenceLength),
                    numFeatures: Math.Max(1, financialModel.NumFeatures));

                var output = financialModel.Predict(input);
                Assert.NotNull(output);

                if (financialModel.SupportsTraining)
                {
                    financialModel.Train(input, output);
                }

                var serialized = financialModel.Serialize();
                clone = CreateNativeModel<T>(modelType);
                ((IFullModel<T, Tensor<T>, Tensor<T>>)clone).Deserialize(serialized);
                return;
            }

            var fallbackInput = FinanceTestHelpers.CreateTimeSeriesInput<T>(1, 1, 1);
            var fallbackOutput = model.Predict(fallbackInput);
            Assert.NotNull(fallbackOutput);

            var fallbackSerialized = model.Serialize();
            clone = CreateNativeModel<T>(modelType);
            ((IFullModel<T, Tensor<T>, Tensor<T>>)clone).Deserialize(fallbackSerialized);
        }
        finally
        {
            if (instance is IDisposable disposable)
            {
                disposable.Dispose();
            }

            if (clone is IDisposable cloneDisposable)
            {
                cloneDisposable.Dispose();
            }
        }
    }

    internal static void AssertOnnxConstructorFails<T>(Type modelType)
    {
        if (!TryInvokeOnnxConstructor<T>(modelType, out var exception))
        {
            return;
        }

        Assert.NotNull(exception);
        Assert.True(exception is FileNotFoundException || exception is InvalidOperationException,
            $"Unexpected ONNX constructor exception type: {exception!.GetType().Name}");
    }

    internal static void RunTradingAgentSmokeTest<T>(Type modelType)
    {
        (ITradingAgent<T> Agent, int StateSize, int ActionSize) result = default;

        try
        {
            result = CreateTradingAgent<T>(modelType);
            var agent = result.Agent;

            var state = FinanceTestHelpers.CreateRandomVector<T>(result.StateSize, seed: 123);
            var action = agent.SelectTradingAction(state, training: false);
            Assert.Equal(result.ActionSize, action.Length);

            var nextState = FinanceTestHelpers.CreateRandomVector<T>(result.StateSize, seed: 321);
            var numOps = MathHelper.GetNumericOperations<T>();
            var reward = numOps.FromDouble(0.01);
            var pnl = numOps.FromDouble(0.02);

            agent.StoreTradingExperience(state, action, reward, nextState, done: false, pnl: pnl);
            _ = agent.TrainOnExperiences();

            var metrics = agent.GetTradingMetrics();
            Assert.NotNull(metrics);
        }
        finally
        {
            if (result.Agent is IDisposable disposable)
            {
                disposable.Dispose();
            }
        }
    }

    private static IEnumerable<Type> GetFinanceTypes()
    {
        var assembly = typeof(PatchTST<>).Assembly;
        return assembly.GetTypes()
            .Where(type => type.IsClass
                && !type.IsAbstract
                && type.Namespace != null
                && type.Namespace.StartsWith(FinanceNamespacePrefix, StringComparison.Ordinal));
    }

    private static object CreateNativeModel<T>(Type closedModelType)
    {
        var architectureType = typeof(NeuralNetworkArchitecture<>).MakeGenericType(typeof(T));
        var constructor = SelectNativeConstructor(closedModelType, architectureType)
            ?? SelectOptionsConstructor(closedModelType);

        var parameters = constructor.GetParameters();
        object? optionsInstance = CreateOptionsInstance(parameters);

        if (optionsInstance != null)
        {
            NormalizeOptions(optionsInstance);
        }

        int inputSize = GetInputSizeFromOptions(optionsInstance) ?? 4;
        int outputSize = GetOutputSizeFromOptions(optionsInstance) ?? Math.Max(1, inputSize);
        var architecture = FinanceTestHelpers.CreateArchitecture<T>(inputSize, outputSize);

        if (optionsInstance != null)
        {
            SetArchitectureOption(optionsInstance, architecture);
        }

        var args = BuildArguments(parameters, architectureType, architecture, optionsInstance);
        return constructor.Invoke(args);
    }

    private static (ITradingAgent<T> Agent, int StateSize, int ActionSize) CreateTradingAgent<T>(Type closedModelType)
    {
        var architectureType = typeof(NeuralNetworkArchitecture<>).MakeGenericType(typeof(T));
        var constructor = SelectTradingConstructor(closedModelType);
        var parameters = constructor.GetParameters();

        object? optionsInstance = CreateOptionsInstance(parameters);
        if (optionsInstance == null)
        {
            throw new InvalidOperationException($"Trading agent {closedModelType.Name} has no options parameter.");
        }

        NormalizeOptions(optionsInstance);

        int stateSize = Math.Max(1, GetIntOption(optionsInstance, "StateSize") ?? 8);
        int actionSize = Math.Max(1, GetIntOption(optionsInstance, "ActionSize") ?? 3);

        var actorArchitecture = FinanceTestHelpers.CreateArchitecture<T>(stateSize, actionSize);
        var criticArchitecture = FinanceTestHelpers.CreateArchitecture<T>(stateSize, 1);

        var args = new object?[parameters.Length];
        for (int i = 0; i < parameters.Length; i++)
        {
            var param = parameters[i];
            if (param.ParameterType == architectureType)
            {
                string paramName = param.Name ?? string.Empty;
                if (paramName.Contains("critic", StringComparison.OrdinalIgnoreCase)
                    || paramName.Contains("value", StringComparison.OrdinalIgnoreCase))
                {
                    args[i] = criticArchitecture;
                }
                else
                {
                    args[i] = actorArchitecture;
                }
            }
            else if (optionsInstance != null && param.ParameterType.IsInstanceOfType(optionsInstance))
            {
                args[i] = optionsInstance;
            }
            else if (param.HasDefaultValue)
            {
                args[i] = param.DefaultValue;
            }
            else if (param.ParameterType.IsValueType)
            {
                args[i] = CreateDefaultValue(param.ParameterType);
            }
            else
            {
                args[i] = null;
            }
        }

        var instance = constructor.Invoke(args);
        return ((ITradingAgent<T>)instance, stateSize, actionSize);
    }

    private static ConstructorInfo? SelectNativeConstructor(Type modelType, Type architectureType)
    {
        return modelType.GetConstructors()
            .Where(ctor =>
            {
                var parameters = ctor.GetParameters();
                return parameters.Length > 0
                    && parameters[0].ParameterType == architectureType
                    && parameters.All(param => param.ParameterType != typeof(string));
            })
            .OrderBy(CountRequiredParameters)
            .ThenBy(ctor => ctor.GetParameters().Length)
            .FirstOrDefault();
    }

    private static ConstructorInfo SelectOptionsConstructor(Type modelType)
    {
        var constructor = modelType.GetConstructors()
            .Where(ctor => ctor.GetParameters().All(param => param.ParameterType != typeof(string)))
            .OrderBy(CountRequiredParameters)
            .ThenBy(ctor => ctor.GetParameters().Length)
            .FirstOrDefault();

        if (constructor == null)
        {
            throw new InvalidOperationException($"No suitable constructor found for {modelType.Name}.");
        }

        return constructor;
    }

    private static ConstructorInfo SelectTradingConstructor(Type modelType)
    {
        var constructor = modelType.GetConstructors()
            .Where(ctor => ctor.GetParameters().All(param => param.ParameterType != typeof(string)))
            .OrderByDescending(CountArchitectureParameters)
            .ThenBy(CountRequiredParameters)
            .FirstOrDefault();

        if (constructor == null)
        {
            throw new InvalidOperationException($"No trading constructor found for {modelType.Name}.");
        }

        return constructor;
    }

    private static bool TryInvokeOnnxConstructor<T>(Type closedModelType, out Exception? exception)
    {
        exception = null;
        var architectureType = typeof(NeuralNetworkArchitecture<>).MakeGenericType(typeof(T));
        var constructor = closedModelType.GetConstructors()
            .FirstOrDefault(ctor =>
            {
                var parameters = ctor.GetParameters();
                return parameters.Length > 1
                    && parameters[0].ParameterType == architectureType
                    && parameters.Any(param => param.ParameterType == typeof(string));
            });

        if (constructor == null)
        {
            return false;
        }

        var parameters = constructor.GetParameters();
        object? optionsInstance = CreateOptionsInstance(parameters);

        if (optionsInstance != null)
        {
            NormalizeOptions(optionsInstance);
        }

        int inputSize = GetInputSizeFromOptions(optionsInstance) ?? 4;
        int outputSize = GetOutputSizeFromOptions(optionsInstance) ?? Math.Max(1, inputSize);
        var architecture = FinanceTestHelpers.CreateArchitecture<T>(inputSize, outputSize);

        if (optionsInstance != null)
        {
            SetArchitectureOption(optionsInstance, architecture);
        }

        var args = BuildArguments(parameters, architectureType, architecture, optionsInstance);
        for (int i = 0; i < parameters.Length; i++)
        {
            if (parameters[i].ParameterType == typeof(string))
            {
                args[i] = "C:\\nonexistent\\model.onnx";
                break;
            }
        }

        try
        {
            _ = constructor.Invoke(args);
        }
        catch (TargetInvocationException ex)
        {
            exception = ex.InnerException ?? ex;
            return true;
        }
        catch (Exception ex)
        {
            exception = ex;
            return true;
        }

        return true;
    }

    private static int CountRequiredParameters(ConstructorInfo ctor)
    {
        return ctor.GetParameters().Count(param => !param.IsOptional);
    }

    private static int CountArchitectureParameters(ConstructorInfo ctor)
    {
        return ctor.GetParameters().Count(param => param.ParameterType.IsGenericType
            && param.ParameterType.GetGenericTypeDefinition() == typeof(NeuralNetworkArchitecture<>));
    }

    private static object? CreateOptionsInstance(ParameterInfo[] parameters)
    {
        foreach (var param in parameters)
        {
            if (!IsOptionsType(param.ParameterType))
            {
                continue;
            }

            var instance = Activator.CreateInstance(param.ParameterType);
            if (instance != null)
            {
                return instance;
            }
        }

        return null;
    }

    private static bool IsOptionsType(Type type)
    {
        if (!type.IsClass)
        {
            return false;
        }

        if (!type.Name.Contains("Options", StringComparison.Ordinal))
        {
            return false;
        }

        return type.GetConstructor(Type.EmptyTypes) != null;
    }

    private static object?[] BuildArguments(ParameterInfo[] parameters, Type architectureType, object architecture, object? optionsInstance)
    {
        var args = new object?[parameters.Length];

        for (int i = 0; i < parameters.Length; i++)
        {
            var param = parameters[i];
            if (param.ParameterType == architectureType)
            {
                args[i] = architecture;
            }
            else if (optionsInstance != null && param.ParameterType.IsInstanceOfType(optionsInstance))
            {
                args[i] = optionsInstance;
            }
            else if (param.HasDefaultValue)
            {
                args[i] = param.DefaultValue;
            }
            else if (param.ParameterType.IsValueType)
            {
                args[i] = CreateDefaultValue(param.ParameterType);
            }
            else
            {
                args[i] = null;
            }
        }

        return args;
    }

    private static object? CreateDefaultValue(Type type)
    {
        if (type == typeof(int))
        {
            return 4;
        }

        if (type == typeof(double))
        {
            return 0.1;
        }

        if (type == typeof(float))
        {
            return 0.1f;
        }

        if (type == typeof(long))
        {
            return 1L;
        }

        if (type == typeof(bool))
        {
            return false;
        }

        return Activator.CreateInstance(type);
    }

    private static int? GetInputSizeFromOptions(object? options)
    {
        return GetIntOption(options, "NumFeatures")
            ?? GetIntOption(options, "InputDim")
            ?? GetIntOption(options, "InputSize")
            ?? GetIntOption(options, "FeatureCount")
            ?? GetIntOption(options, "StateSize")
            ?? GetIntOption(options, "NumAssets")
            ?? GetIntOption(options, "NumNodes")
            ?? GetIntOption(options, "NodeCount");
    }

    private static int? GetOutputSizeFromOptions(object? options)
    {
        return GetIntOption(options, "ActionSize")
            ?? GetIntOption(options, "NumSentimentClasses")
            ?? GetIntOption(options, "OutputSize")
            ?? GetIntOption(options, "NumClasses")
            ?? GetIntOption(options, "NumOutputs")
            ?? GetIntOption(options, "NumAssets");
    }

    private static int? GetIntOption(object? options, string name)
    {
        if (options == null)
        {
            return null;
        }

        var property = options.GetType().GetProperty(name, BindingFlags.Public | BindingFlags.Instance);
        if (property == null || !property.CanRead)
        {
            return null;
        }

        var value = property.GetValue(options);
        if (value == null)
        {
            return null;
        }

        if (value is int intValue)
        {
            return intValue;
        }

        if (value is long longValue)
        {
            return (int)longValue;
        }

        if (value is short shortValue)
        {
            return shortValue;
        }

        return null;
    }

    private static void NormalizeOptions(object options)
    {
        SetInt(options, "LookbackWindow", 8);
        SetInt(options, "SequenceLength", 8);
        SetInt(options, "ContextLength", 8);
        SetInt(options, "HistoryLength", 8);
        SetInt(options, "InputLength", 8);
        SetInt(options, "WindowSize", 8);
        SetInt(options, "ForecastHorizon", 4);
        SetInt(options, "PredictionHorizon", 4);
        SetInt(options, "ForecastLength", 4);
        SetInt(options, "OutputLength", 4);
        SetInt(options, "NumFeatures", 4);
        SetInt(options, "InputDim", 4);
        SetInt(options, "FeatureCount", 4);
        SetInt(options, "NumAssets", 4);
        SetInt(options, "NumFactors", 3);
        SetInt(options, "NumNodes", 4);
        SetInt(options, "NodeCount", 4);
        SetInt(options, "NumSentimentClasses", 3);
        SetInt(options, "VocabularySize", 128);
        SetInt(options, "MaxSequenceLength", 16);
        SetInt(options, "HiddenDimension", 8);
        SetInt(options, "EmbeddingDim", 8);
        SetInt(options, "ModelDimension", 8);
        SetInt(options, "FeedForwardDimension", 16);
        SetInt(options, "IntermediateDimension", 16);
        SetInt(options, "NumLayers", 1);
        SetInt(options, "NumEncoderLayers", 1);
        SetInt(options, "NumDecoderLayers", 1);
        SetInt(options, "NumAttentionHeads", 2);
        SetInt(options, "NumHeads", 2);
        SetInt(options, "PatchSize", 2);
        SetInt(options, "Stride", 1);
        SetInt(options, "MovingAverageKernel", 3);
        SetInt(options, "AutoCorrelationFactor", 2);
        SetInt(options, "TopK", 2);
        SetInt(options, "KernelSize", 3);
        SetInt(options, "NumStacks", 1);
        SetInt(options, "NumBlocks", 1);
        SetInt(options, "NumLevels", 1);
        SetInt(options, "NumChannels", 4);
        SetInt(options, "Dilation", 1);
        SetInt(options, "BatchSize", 4);
        SetInt(options, "ReplayBufferSize", 32);
        SetInt(options, "TargetUpdateFrequency", 10);
        SetInt(options, "WarmupSteps", 0);
        SetInt(options, "StateSize", 8);
        SetInt(options, "ActionSize", 3);
        SetInt(options, "MaxInventory", 10);

        SetDouble(options, "Dropout", 0.0);
        SetDouble(options, "DropoutRate", 0.0);
        SetDouble(options, "AttentionDropout", 0.0);
        SetDouble(options, "InventoryPenalty", 0.0);
        SetDouble(options, "BaseSpread", 0.001);
        SetDouble(options, "TransactionCost", 0.0);
        SetDouble(options, "EpsilonStart", 0.1);
        SetDouble(options, "EpsilonEnd", 0.01);
        SetDouble(options, "EpsilonDecay", 0.99);

        SetNumeric(options, "LearningRate", 0.001);
        SetNumeric(options, "DiscountFactor", 0.99);
        SetNumeric(options, "InitialCapital", 10000.0);
        SetNumeric(options, "MaxPositionSize", 1.0);

        SetBool(options, "UseInstanceNormalization", false);
        SetBool(options, "ChannelIndependent", true);

        SetIntArray(options, "HiddenLayers", new[] { 16, 8 });
        SetIntArray(options, "HiddenLayerSizes", new[] { 16, 8 });
        SetIntArray(options, "LayerSizes", new[] { 16, 8 });
    }

    private static void SetArchitectureOption(object options, object architecture)
    {
        var property = options.GetType().GetProperty("Architecture", BindingFlags.Public | BindingFlags.Instance);
        if (property == null || !property.CanWrite)
        {
            return;
        }

        if (property.PropertyType.IsInstanceOfType(architecture))
        {
            property.SetValue(options, architecture);
        }
    }

    private static void SetInt(object options, string name, int value)
    {
        var property = options.GetType().GetProperty(name, BindingFlags.Public | BindingFlags.Instance);
        if (property == null || !property.CanWrite)
        {
            return;
        }

        var targetType = Nullable.GetUnderlyingType(property.PropertyType) ?? property.PropertyType;
        if (targetType == typeof(int))
        {
            property.SetValue(options, value);
        }
        else if (targetType == typeof(long))
        {
            property.SetValue(options, (long)value);
        }
        else if (targetType == typeof(short))
        {
            property.SetValue(options, (short)value);
        }
    }

    private static void SetDouble(object options, string name, double value)
    {
        var property = options.GetType().GetProperty(name, BindingFlags.Public | BindingFlags.Instance);
        if (property == null || !property.CanWrite)
        {
            return;
        }

        var targetType = Nullable.GetUnderlyingType(property.PropertyType) ?? property.PropertyType;
        if (targetType == typeof(double))
        {
            property.SetValue(options, value);
        }
        else if (targetType == typeof(float))
        {
            property.SetValue(options, (float)value);
        }
    }

    private static void SetNumeric(object options, string name, double value)
    {
        var property = options.GetType().GetProperty(name, BindingFlags.Public | BindingFlags.Instance);
        if (property == null || !property.CanWrite)
        {
            return;
        }

        var targetType = Nullable.GetUnderlyingType(property.PropertyType) ?? property.PropertyType;
        if (targetType == typeof(float))
        {
            property.SetValue(options, (float)value);
        }
        else if (targetType == typeof(double))
        {
            property.SetValue(options, value);
        }
        else if (targetType == typeof(decimal))
        {
            property.SetValue(options, (decimal)value);
        }
        else if (targetType == typeof(int))
        {
            property.SetValue(options, (int)Math.Round(value));
        }
        else if (targetType == typeof(long))
        {
            property.SetValue(options, (long)Math.Round(value));
        }
    }

    private static void SetBool(object options, string name, bool value)
    {
        var property = options.GetType().GetProperty(name, BindingFlags.Public | BindingFlags.Instance);
        if (property == null || !property.CanWrite)
        {
            return;
        }

        if (property.PropertyType == typeof(bool))
        {
            property.SetValue(options, value);
        }
    }

    private static void SetIntArray(object options, string name, int[] value)
    {
        var property = options.GetType().GetProperty(name, BindingFlags.Public | BindingFlags.Instance);
        if (property == null || !property.CanWrite)
        {
            return;
        }

        if (property.PropertyType == typeof(int[]))
        {
            property.SetValue(options, value);
        }
    }
}
