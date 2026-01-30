using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using AiDotNet.Finance.Forecasting.Transformers;
using AiDotNet.Finance.Interfaces;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.LossFunctions;
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

    internal static IReadOnlyList<Type> GetFinancialModelTypes<T>()
    {
        return GetFinanceTypesByInterface<T>(typeof(IFinancialModel<>));
    }

    internal static IReadOnlyList<Type> GetForecastingModelTypes<T>()
    {
        return GetFinanceTypesByInterface<T>(typeof(IForecastingModel<>));
    }

    internal static IReadOnlyList<Type> GetFinancialNlpModelTypes<T>()
    {
        return GetFinanceTypesByInterface<T>(typeof(IFinancialNLPModel<>));
    }

    internal static IReadOnlyList<Type> GetRiskModelTypes<T>()
    {
        return GetFinanceTypesByInterface<T>(typeof(IRiskModel<>));
    }

    internal static IReadOnlyList<Type> GetPortfolioModelTypes<T>()
    {
        return GetFinanceTypesByInterface<T>(typeof(IPortfolioOptimizer<>));
    }

    internal static IReadOnlyList<Type> GetVolatilityModelTypes<T>()
    {
        return GetFinanceTypesByInterface<T>(typeof(IVolatilityModel<>));
    }

    internal static IReadOnlyList<Type> GetFactorModelTypes<T>()
    {
        return GetFinanceTypesByInterface<T>(typeof(IFactorModel<>));
    }

    internal static IReadOnlyList<Type> GetAutoMlModelTypes<T>()
    {
        var numericType = typeof(T);
        var tensorType = typeof(Tensor<>).MakeGenericType(numericType);
        var autoMlInterface = typeof(IAutoMLModel<,,>).MakeGenericType(numericType, tensorType, tensorType);

        return GetFinanceTypes()
            .Select(type => type.IsGenericTypeDefinition ? type.MakeGenericType(numericType) : type)
            .Where(type => autoMlInterface.IsAssignableFrom(type))
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

    internal static IReadOnlyList<Type> GetFinanceModelTypesByNamespace<T>(string namespacePrefix)
    {
        if (string.IsNullOrWhiteSpace(namespacePrefix))
        {
            throw new ArgumentException("Namespace prefix cannot be null or empty.", nameof(namespacePrefix));
        }

        var numericType = typeof(T);
        var tensorType = typeof(Tensor<>).MakeGenericType(numericType);
        var modelInterface = typeof(IFullModel<,,>).MakeGenericType(numericType, tensorType, tensorType);

        return GetFinanceTypes()
            .Where(type => type.Namespace != null
                && type.Namespace.StartsWith(namespacePrefix, StringComparison.Ordinal))
            .Select(type => type.IsGenericTypeDefinition ? type.MakeGenericType(numericType) : type)
            .Where(type => modelInterface.IsAssignableFrom(type))
            .OrderBy(type => type.FullName, StringComparer.Ordinal)
            .ToList();
    }

    internal static void RunFullModelSmokeTest<T>(Type modelType)
    {
        RunFullModelSmokeTest<T>(modelType, includeQuantileForecast: false);
    }

    internal static void RunFullModelSmokeTest<T>(Type modelType, bool includeQuantileForecast)
    {
        object? instance = null;
        object? clone = null;
        object? fileClone = null;
        object? stateClone = null;

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

            RunCommonModelAssertions(model);

            if (model is IFinancialNLPModel<T> nlpModel)
            {
                var input = FinanceTestHelpers.CreateTokenTensor<T>(
                    batchSize: 1,
                    sequenceLength: Math.Max(1, nlpModel.MaxSequenceLength),
                    vocabularySize: Math.Max(2, nlpModel.VocabularySize));

                var output = nlpModel.Predict(input);
                Assert.NotNull(output);

                var sentimentFromTokens = nlpModel.AnalyzeSentiment(input);
                Assert.NotNull(sentimentFromTokens);

                var embeddings = nlpModel.GetEmbeddings(input);
                Assert.NotNull(embeddings);

                var sequenceEmbedding = nlpModel.GetSequenceEmbedding(input);
                Assert.NotNull(sequenceEmbedding);

                int safeMaxLength = Math.Max(1, nlpModel.MaxSequenceLength);
                var tokenIds = nlpModel.Tokenize("Markets are steady.", safeMaxLength);
                Assert.NotNull(tokenIds);
                _ = nlpModel.Detokenize(tokenIds);

                if (nlpModel.MaxSequenceLength > 0)
                {
                    var sentimentFromText = nlpModel.AnalyzeSentiment(new[] { "Earnings beat expectations." });
                    Assert.NotNull(sentimentFromText);
                    Assert.NotEmpty(sentimentFromText);
                }

                if (model is IFinancialModel<T> nlpFinancialModel)
                {
                    var nlpMetrics = nlpFinancialModel.GetFinancialMetrics();
                    Assert.NotNull(nlpMetrics);
                }

                if (nlpModel.SupportsTraining)
                {
                    nlpModel.Train(input, output);
                }

                var serialized = nlpModel.Serialize();
                clone = CreateNativeModel<T>(modelType);
                ((IFullModel<T, Tensor<T>, Tensor<T>>)clone).Deserialize(serialized);

                RunCheckpointRoundTrip(model, modelType, ref stateClone);
                RunFileRoundTrip(model, modelType, ref fileClone);
                return;
            }

            if (model is IFinancialModel<T> financialModel)
            {
                // Risk models and Factor models expect 2D input [batch, features], not 3D time series
                bool is2DModel = model is IRiskModel<T> || model is IFactorModel<T>;
                Tensor<T> input;

                if (is2DModel || financialModel.SequenceLength <= 1)
                {
                    // Create 2D input for risk/factor models
                    input = FinanceTestHelpers.CreateRandomTensor<T>(
                        new[] { 1, Math.Max(1, financialModel.NumFeatures) });
                }
                else
                {
                    input = FinanceTestHelpers.CreateTimeSeriesInput<T>(
                        batchSize: 1,
                        sequenceLength: Math.Max(1, financialModel.SequenceLength),
                        numFeatures: Math.Max(1, financialModel.NumFeatures));
                }

                var output = financialModel.Predict(input);
                Assert.NotNull(output);

                if (includeQuantileForecast)
                {
                    var quantileForecast = financialModel.Forecast(input, new[] { 0.1, 0.5, 0.9 });
                    Assert.NotNull(quantileForecast);
                }

                var forecast = financialModel.Forecast(input);
                Assert.NotNull(forecast);

                var financialMetrics = financialModel.GetFinancialMetrics();
                Assert.NotNull(financialMetrics);

                if (financialModel is IForecastingModel<T> forecastingModel)
                {
                    var normalized = forecastingModel.ApplyInstanceNormalization(input);
                    Assert.NotNull(normalized);

                    int steps = Math.Max(1, financialModel.PredictionHorizon);
                    var autoreg = forecastingModel.AutoregressiveForecast(input, steps);
                    Assert.NotNull(autoreg);

                    var evalMetrics = forecastingModel.Evaluate(input, output);
                    Assert.NotNull(evalMetrics);
                }

                if (financialModel is IRiskModel<T> riskModel)
                {
                    var numOps = MathHelper.GetNumericOperations<T>();
                    int assets = Math.Max(1, financialModel.NumFeatures);
                    var returns = FinanceTestHelpers.CreateReturnsMatrix<T>(samples: 8, assets: assets);
                    var weights = FinanceTestHelpers.CreateUniformWeights<T>(assets);
                    var scenarios = FinanceTestHelpers.CreateScenarioMatrix<T>(scenarios: 3, assets: assets);

                    _ = riskModel.CalculateVaR(returns, weights);
                    _ = riskModel.CalculateCVaR(returns, weights);

                    var stress = riskModel.StressTest(weights, scenarios);
                    Assert.NotNull(stress);

                    var contributions = riskModel.DecomposeRisk(returns, weights);
                    Assert.NotNull(contributions);

                    _ = riskModel.EstimateExceedanceProbability(returns, weights, numOps.FromDouble(0.01));

                    var riskMetrics = riskModel.GetRiskMetrics();
                    Assert.NotNull(riskMetrics);
                }

                if (financialModel is IPortfolioOptimizer<T> portfolioModel)
                {
                    var numOps = MathHelper.GetNumericOperations<T>();
                    int assets = Math.Max(1, portfolioModel.NumAssets);
                    var expectedReturns = FinanceTestHelpers.CreateExpectedReturns<T>(assets);
                    var covariance = FinanceTestHelpers.CreateDiagonalMatrix<T>(assets);
                    var weights = FinanceTestHelpers.CreateUniformWeights<T>(assets);

                    _ = portfolioModel.OptimizeWeights(expectedReturns, covariance);
                    _ = portfolioModel.ComputeRiskContribution(weights, covariance);
                    _ = portfolioModel.CalculateExpectedReturn(weights, expectedReturns);
                    _ = portfolioModel.CalculateVolatility(weights, covariance);
                    _ = portfolioModel.CalculateSharpeRatio(weights, expectedReturns, covariance, numOps.Zero);

                    var portfolioMetrics = portfolioModel.GetPortfolioMetrics();
                    Assert.NotNull(portfolioMetrics);
                }

                if (financialModel is IVolatilityModel<T> volatilityModel)
                {
                    int assets = Math.Max(1, financialModel.NumFeatures);
                    int lookback = Math.Max(1, financialModel.SequenceLength);
                    int horizon = Math.Max(1, financialModel.PredictionHorizon);

                    var returnsSequence = FinanceTestHelpers.CreateReturnsMatrix<T>(lookback, assets);
                    var forecastVol = volatilityModel.ForecastVolatility(returnsSequence, horizon);
                    Assert.NotNull(forecastVol);

                    var currentVol = volatilityModel.EstimateCurrentVolatility(returnsSequence);
                    Assert.NotNull(currentVol);

                    var returnsMatrix = FinanceTestHelpers.CreateReturnsMatrix<T>(Math.Max(2, lookback), assets);
                    var covariance = volatilityModel.ComputeCovarianceMatrix(returnsMatrix);
                    Assert.NotNull(covariance);

                    var correlation = volatilityModel.ComputeCorrelationMatrix(returnsMatrix);
                    Assert.NotNull(correlation);

                    var realized = volatilityModel.CalculateRealizedVolatility(returnsMatrix);
                    Assert.NotNull(realized);

                    var volatilityMetrics = volatilityModel.GetVolatilityMetrics();
                    Assert.NotNull(volatilityMetrics);
                }

                if (financialModel is IFactorModel<T> factorModel)
                {
                    int batchSize = 1;
                    int sequenceLength = Math.Max(1, financialModel.SequenceLength);
                    int assets = Math.Max(1, factorModel.NumAssets);
                    int factors = Math.Max(1, factorModel.NumFactors);

                    var returns = FinanceTestHelpers.CreateReturnsSeries<T>(batchSize, sequenceLength, assets);
                    var factorReturns = FinanceTestHelpers.CreateFactorSeries<T>(batchSize, sequenceLength, factors);
                    var exposures = FinanceTestHelpers.CreateFactorExposure<T>(batchSize, factors);

                    var extracted = factorModel.ExtractFactors(returns);
                    Assert.NotNull(extracted);

                    var loadings = factorModel.GetFactorLoadings(returns);
                    Assert.NotNull(loadings);

                    var expected = factorModel.PredictReturns(exposures);
                    Assert.NotNull(expected);

                    var covariance = factorModel.GetFactorCovariance(returns);
                    Assert.NotNull(covariance);

                    var alpha = factorModel.ComputeAlpha(returns, factorReturns);
                    Assert.NotNull(alpha);

                    var factorMetrics = factorModel.GetFactorMetrics();
                    Assert.NotNull(factorMetrics);
                }

                if (financialModel.SupportsTraining)
                {
                    financialModel.Train(input, output);
                }

                var serialized = financialModel.Serialize();
                clone = CreateNativeModel<T>(modelType);
                ((IFullModel<T, Tensor<T>, Tensor<T>>)clone).Deserialize(serialized);

                RunCheckpointRoundTrip(model, modelType, ref stateClone);
                RunFileRoundTrip(model, modelType, ref fileClone);
                return;
            }

            var fallbackInput = FinanceTestHelpers.CreateTimeSeriesInput<T>(1, 1, 1);
            var fallbackOutput = model.Predict(fallbackInput);
            Assert.NotNull(fallbackOutput);

            var fallbackSerialized = model.Serialize();
            clone = CreateNativeModel<T>(modelType);
            ((IFullModel<T, Tensor<T>, Tensor<T>>)clone).Deserialize(fallbackSerialized);

            RunCheckpointRoundTrip(model, modelType, ref stateClone);
            RunFileRoundTrip(model, modelType, ref fileClone);
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

            if (stateClone is IDisposable stateCloneDisposable)
            {
                stateCloneDisposable.Dispose();
            }

            if (fileClone is IDisposable fileCloneDisposable)
            {
                fileCloneDisposable.Dispose();
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

            var numOps = MathHelper.GetNumericOperations<T>();
            var constrained = agent.ExecuteTradeWithRiskManagement(state, numOps.FromDouble(0.5));
            Assert.Equal(result.ActionSize, constrained.Length);

            var nextState = FinanceTestHelpers.CreateRandomVector<T>(result.StateSize, seed: 321);
            var reward = numOps.FromDouble(0.01);
            var pnl = numOps.FromDouble(0.02);

            agent.StoreTradingExperience(state, action, reward, nextState, done: false, pnl: pnl);
            _ = agent.TrainOnExperiences();

            var metrics = agent.GetTradingMetrics();
            Assert.NotNull(metrics);

            agent.ResetTrading(numOps.FromDouble(10000.0));

            var metadata = agent.GetModelMetadata();
            Assert.NotNull(metadata);

            if (agent is IFullModel<T, Vector<T>, Vector<T>> fullModel)
            {
                var parameters = fullModel.GetParameters();
                Assert.Equal(fullModel.ParameterCount, parameters.Length);

                var withParams = fullModel.WithParameters(parameters);
                Assert.NotNull(withParams);

                if (withParams is IDisposable withParamsDisposable)
                {
                    withParamsDisposable.Dispose();
                }

                var serialized = fullModel.Serialize();
                fullModel.Deserialize(serialized);

                using var stateStream = new MemoryStream();
                fullModel.SaveState(stateStream);
                stateStream.Position = 0;
                fullModel.LoadState(stateStream);

                var tempPath = Path.GetTempFileName();
                try
                {
                    fullModel.SaveModel(tempPath);
                    fullModel.LoadModel(tempPath);
                }
                finally
                {
                    if (File.Exists(tempPath))
                    {
                        File.Delete(tempPath);
                    }
                }
            }
        }
        finally
        {
            if (result.Agent is IDisposable disposable)
            {
                disposable.Dispose();
            }
        }
    }

    private static IReadOnlyList<Type> GetFinanceTypesByInterface<T>(Type openInterfaceType)
    {
        var numericType = typeof(T);
        var interfaceType = openInterfaceType.IsGenericTypeDefinition
            ? openInterfaceType.MakeGenericType(numericType)
            : openInterfaceType;

        return GetFinanceTypes()
            .Select(type => type.IsGenericTypeDefinition ? type.MakeGenericType(numericType) : type)
            .Where(type => interfaceType.IsAssignableFrom(type))
            .OrderBy(type => type.FullName, StringComparer.Ordinal)
            .ToList();
    }

    private static void RunCommonModelAssertions<T>(IFullModel<T, Tensor<T>, Tensor<T>> model)
    {
        var metadata = model.GetModelMetadata();
        Assert.NotNull(metadata);

        var featureImportance = model.GetFeatureImportance();
        Assert.NotNull(featureImportance);

        var activeFeatures = model.GetActiveFeatureIndices()?.ToList() ?? new List<int>();
        model.SetActiveFeatureIndices(activeFeatures.Take(1));

        if (activeFeatures.Count > 0)
        {
            Assert.True(model.IsFeatureUsed(activeFeatures[0]));
        }

        var parameters = model.GetParameters();
        Assert.Equal(model.ParameterCount, parameters.Length);

        var withParams = model.WithParameters(parameters);
        Assert.NotNull(withParams);

        if (withParams is IDisposable disposable)
        {
            disposable.Dispose();
        }
    }

    private static void RunCheckpointRoundTrip<T>(
        IFullModel<T, Tensor<T>, Tensor<T>> model,
        Type modelType,
        ref object? stateClone)
    {
        using var stateStream = new MemoryStream();
        model.SaveState(stateStream);
        stateStream.Position = 0;

        stateClone = CreateNativeModel<T>(modelType);
        ((IFullModel<T, Tensor<T>, Tensor<T>>)stateClone).LoadState(stateStream);
    }

    private static void RunFileRoundTrip<T>(
        IFullModel<T, Tensor<T>, Tensor<T>> model,
        Type modelType,
        ref object? fileClone)
    {
        string tempPath = Path.GetTempFileName();
        try
        {
            model.SaveModel(tempPath);
            fileClone = CreateNativeModel<T>(modelType);
            ((IFullModel<T, Tensor<T>, Tensor<T>>)fileClone).LoadModel(tempPath);
        }
        finally
        {
            if (File.Exists(tempPath))
            {
                File.Delete(tempPath);
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

        // SAC agents need critic input = state + action size (Q-network takes both as input)
        bool isSACAgent = closedModelType.Name.Contains("SAC", StringComparison.OrdinalIgnoreCase);
        int criticInputSize = isSACAgent ? stateSize + actionSize : stateSize;
        var criticArchitecture = FinanceTestHelpers.CreateArchitecture<T>(criticInputSize, 1);

        var args = new object?[parameters.Length];
        for (int i = 0; i < parameters.Length; i++)
        {
            var param = parameters[i];
            if (param.ParameterType == architectureType)
            {
                string paramName = param.Name ?? string.Empty;
                if (paramName.Contains("critic", StringComparison.OrdinalIgnoreCase)
                    || paramName.Contains("value", StringComparison.OrdinalIgnoreCase)
                    || paramName.Contains("secondary", StringComparison.OrdinalIgnoreCase))
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

        // Extract input size from architecture for numFeatures parameters
        int? inputSizeFromArch = null;
        var inputSizeProp = architectureType.GetProperty("InputSize", BindingFlags.Public | BindingFlags.Instance);
        if (inputSizeProp != null)
        {
            var val = inputSizeProp.GetValue(architecture);
            if (val is int intVal)
            {
                inputSizeFromArch = intVal;
            }
        }

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
            else if (param.ParameterType == typeof(int) && param.Name != null &&
                     (param.Name.Equals("numFeatures", StringComparison.OrdinalIgnoreCase) ||
                      param.Name.Equals("features", StringComparison.OrdinalIgnoreCase) ||
                      param.Name.Equals("inputFeatures", StringComparison.OrdinalIgnoreCase)))
            {
                // Use architecture's InputSize for numFeatures parameters to match layer dimensions
                args[i] = inputSizeFromArch ?? GetIntOption(optionsInstance, "NumFeatures") ?? 1;
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
            ?? GetIntOption(options, "NodeCount")
            ?? GetIntOption(options, "HiddenDimension");  // For NLP models like FinBERT, InvestLM, etc.
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
        SetInt(options, "InputSize", 4);
        SetInt(options, "ForecastHorizon", 4);
        SetInt(options, "PredictionHorizon", 4);
        SetInt(options, "PredictionLength", 4);
        SetInt(options, "ForecastLength", 4);
        SetInt(options, "OutputLength", 4);
        SetInt(options, "OutputSize", 4);
        SetInt(options, "NumFeatures", 8);   // Must be compatible with test input shapes
        SetInt(options, "InputDim", 8);       // Must be compatible with test input shapes
        SetInt(options, "FeatureCount", 8);   // Must be compatible with test input shapes
        SetInt(options, "NumAssets", 4);
        SetInt(options, "NumFactors", 3);
        SetInt(options, "NumNodes", 4);
        SetInt(options, "NodeCount", 4);
        SetInt(options, "NumSentimentClasses", 3);
        SetInt(options, "VocabularySize", 128);
        SetInt(options, "NumTokens", 32);
        SetInt(options, "MaxSequenceLength", 16);
        SetInt(options, "HiddenDimension", 24);  // Must be divisible by 12 (NLP model attention heads)
        SetInt(options, "HiddenDim", 24);        // Must be divisible by 12 (NLP model attention heads)
        SetInt(options, "HiddenSize", 24);       // Must be divisible by 12 (NLP model attention heads)
        SetInt(options, "HiddenLayerSize", 24);  // Must be divisible by 12 (NLP model attention heads)
        SetInt(options, "EmbeddingDim", 24);     // Must be divisible by 12 (NLP model attention heads)
        SetInt(options, "EmbeddingSize", 24);    // Must be divisible by 12 (NLP model attention heads)
        SetInt(options, "LLMDimension", 24);     // Must be divisible by 12 (NLP model attention heads)
        SetInt(options, "ModelDimension", 24);   // Must be divisible by 12 (NLP model attention heads)
        SetInt(options, "ModelDim", 24);         // Must be divisible by 12 (NLP model attention heads)
        SetInt(options, "FeedForwardDimension", 48);  // Typically 2-4x hidden dimension
        SetInt(options, "FeedForwardDim", 48);       // Typically 2-4x hidden dimension
        SetInt(options, "IntermediateDimension", 48); // Typically 2-4x hidden dimension
        SetInt(options, "NumLayers", 1);
        SetInt(options, "NumHiddenLayers", 1);
        SetInt(options, "NumEncoderLayers", 1);
        SetInt(options, "NumDecoderLayers", 1);
        SetInt(options, "NumAttentionHeads", 2);
        SetInt(options, "NumHeads", 2);
        SetInt(options, "HeadCount", 2);
        SetInt(options, "PatchSize", 2);
        SetInt(options, "PatchLength", 1);
        SetInt(options, "PatchStride", 1);
        SetInt(options, "Stride", 1);
        SetInt(options, "MovingAverageKernel", 3);
        SetInt(options, "AutoCorrelationFactor", 2);
        SetInt(options, "TopK", 2);
        SetInt(options, "KernelSize", 3);
        SetInt(options, "ConvKernelSize", 3);
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
        SetInt(options, "LabelLength", 4);
        SetInt(options, "SegmentLength", 4);
        SetInt(options, "NumPrototypes", 4);

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

        // Set LossFunction for RL agents (required by ReinforcementLearningAgentBase)
        SetLossFunction(options);

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

    private static void SetLossFunction(object options)
    {
        var property = options.GetType().GetProperty("LossFunction", BindingFlags.Public | BindingFlags.Instance);
        if (property == null || !property.CanWrite || property.GetValue(options) != null)
        {
            // Property doesn't exist, can't be written, or already has a value
            return;
        }

        var propertyType = property.PropertyType;
        if (!propertyType.IsGenericType)
        {
            return;
        }

        // Get ILossFunction<T> generic argument to determine T
        var genericDef = propertyType.GetGenericTypeDefinition();
        if (genericDef != typeof(ILossFunction<>))
        {
            // Also handle nullable ILossFunction<T>?
            var underlyingType = Nullable.GetUnderlyingType(propertyType);
            if (underlyingType != null && underlyingType.IsGenericType)
            {
                genericDef = underlyingType.GetGenericTypeDefinition();
                propertyType = underlyingType;
            }

            if (genericDef != typeof(ILossFunction<>))
            {
                return;
            }
        }

        var typeArg = propertyType.GetGenericArguments()[0];
        var lossFunctionType = typeof(MeanSquaredErrorLoss<>).MakeGenericType(typeArg);
        var lossFunction = Activator.CreateInstance(lossFunctionType);
        property.SetValue(options, lossFunction);
    }
}
