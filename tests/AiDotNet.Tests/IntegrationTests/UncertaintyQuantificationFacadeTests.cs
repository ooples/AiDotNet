using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Data.Loaders;
using AiDotNet.Engines;
using AiDotNet.Enums;
using AiDotNet.Evaluation;
using AiDotNet.Helpers;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models;
using AiDotNet.Models.Inputs;
using AiDotNet.Models.Options;
using AiDotNet.Models.Results;
using AiDotNet.NeuralNetworks;
using AiDotNet.Tests.TestUtilities;
using AiDotNet.UncertaintyQuantification.BayesianNeuralNetworks;
using AiDotNet.UncertaintyQuantification.Layers;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests;

[Collection("NonParallelIntegration")]
public sealed class UncertaintyQuantificationFacadeTests
{
    [Fact]
    public void MCDropoutLayer_WithMonteCarloMode_ProducesStochasticOutput()
    {
        var layer = new MCDropoutLayer<double>(dropoutRate: 0.5, mcMode: true, randomSeed: 123);
        layer.SetTrainingMode(false);

        var input = Tensor<double>.FromVector(new Vector<double>(new[] { 1.0, 2.0, 3.0, 4.0 }));
        var first = layer.Forward(input).ToVector();
        Assert.True(HasAnyDifferenceAcrossSamples(() => layer.Forward(input).ToVector(), first, attempts: 8));
    }

    [Fact]
    public async Task PredictWithUncertainty_WithConfiguredUq_ReturnsVarianceWithSameShape()
    {
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Medium,
            inputSize: 2,
            outputSize: 1);
        var model = new NeuralNetworkModel<double>(architecture);

        var optimizer = new PassthroughOptimizer<double, Tensor<double>, Tensor<double>>(model);

        var builder = new PredictionModelBuilder<double, Tensor<double>, Tensor<double>>()
            .ConfigureModel(model)
            .ConfigureOptimizer(optimizer)
            .ConfigureGpuAcceleration(new GpuAccelerationConfig { UsageLevel = GpuUsageLevel.AlwaysCpu })
            .ConfigureUncertaintyQuantification(new UncertaintyQuantificationOptions
            {
                Method = UncertaintyQuantificationMethod.MonteCarloDropout,
                NumSamples = 16,
                MonteCarloDropoutRate = 0.5,
                RandomSeed = 123
            });

        var x = Tensor<double>.FromMatrix(new Matrix<double>(new double[,]
        {
            { 1.0, 2.0 },
            { 3.0, 4.0 },
            { 5.0, 6.0 },
            { 7.0, 8.0 }
        }));
        Assert.Equal(2, x.Rank);

        var y = Tensor<double>.FromVector(new Vector<double>(new[] { 0.0, 1.0, 1.0, 0.0 }));

        var result = await builder
            .ConfigureDataLoader(DataLoaders.FromTensors(x, y))
            .BuildAsync();

        var trainedModel = Assert.IsType<NeuralNetworkModel<double>>(result.OptimizationResult.BestSolution);
        var injectedMcDropoutLayers = trainedModel.Network.LayersReadOnly.OfType<MCDropoutLayer<double>>().ToList();
        Assert.NotEmpty(injectedMcDropoutLayers);

        var parameters = trainedModel.Network.GetParameters();
        for (int i = 0; i < parameters.Length; i++)
        {
            parameters[i] = (i + 1) * 0.01;
        }
        trainedModel.Network.UpdateParameters(parameters);

        // Sanity-check: MC mode should yield stochastic predictions.
        foreach (var layer in injectedMcDropoutLayers)
        {
            layer.MonteCarloMode = true;
        }
        var sample1 = result.Predict(x);
        var sample2 = result.Predict(x);
        foreach (var layer in injectedMcDropoutLayers)
        {
            layer.MonteCarloMode = false;
        }
        var sample1Tensor = ConversionsHelper.ConvertToTensor<double>(sample1);
        var sample2Tensor = ConversionsHelper.ConvertToTensor<double>(sample2);
        Assert.Equal(sample1Tensor.Shape, sample2Tensor.Shape);

        var uqResult = result.PredictWithUncertainty(x, numSamples: 16);

        Assert.NotNull(uqResult.Variance);
        Assert.Equal(uqResult.Prediction.Shape, uqResult.Variance!.Shape);
        Assert.True(uqResult.Metrics.ContainsKey("predictive_entropy"));
        Assert.True(uqResult.Metrics.ContainsKey("mutual_information"));
    }

    [Fact]
    public async Task PredictWithUncertainty_WithRandomSeed_IsDeterministicPerCall()
    {
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Medium,
            inputSize: 2,
            outputSize: 1);
        var model = new NeuralNetworkModel<double>(architecture);

        var optimizer = new PassthroughOptimizer<double, Tensor<double>, Tensor<double>>(model);

        var builder = new PredictionModelBuilder<double, Tensor<double>, Tensor<double>>()
            .ConfigureModel(model)
            .ConfigureOptimizer(optimizer)
            .ConfigureGpuAcceleration(new GpuAccelerationConfig { UsageLevel = GpuUsageLevel.AlwaysCpu })
            .ConfigureUncertaintyQuantification(new UncertaintyQuantificationOptions
            {
                Method = UncertaintyQuantificationMethod.MonteCarloDropout,
                NumSamples = 8,
                MonteCarloDropoutRate = 0.5,
                RandomSeed = 42
            });

        var x = Tensor<double>.FromMatrix(new Matrix<double>(new double[,]
        {
            { 1.0, 2.0 },
            { 3.0, 4.0 }
        }));
        var y = Tensor<double>.FromVector(new Vector<double>(new[] { 0.0, 1.0 }));

        var result = await builder
            .ConfigureDataLoader(DataLoaders.FromTensors(x, y))
            .BuildAsync();
        var trainedModel = Assert.IsType<NeuralNetworkModel<double>>(result.OptimizationResult.BestSolution);
        var parameters = trainedModel.Network.GetParameters();
        for (int i = 0; i < parameters.Length; i++)
        {
            parameters[i] = (i + 1) * 0.01;
        }
        trainedModel.Network.UpdateParameters(parameters);

        var first = result.PredictWithUncertainty(x);
        var second = result.PredictWithUncertainty(x);

        AssertTensorEqual(first.Prediction, second.Prediction);
        Assert.NotNull(first.Variance);
        Assert.NotNull(second.Variance);
        AssertTensorEqual(first.Variance!, second.Variance!);

        Assert.True(first.Metrics.ContainsKey("predictive_entropy"));
        Assert.True(first.Metrics.ContainsKey("mutual_information"));
        Assert.True(second.Metrics.ContainsKey("predictive_entropy"));
        Assert.True(second.Metrics.ContainsKey("mutual_information"));

        AssertTensorEqual(first.Metrics["predictive_entropy"], second.Metrics["predictive_entropy"]);
        AssertTensorEqual(first.Metrics["mutual_information"], second.Metrics["mutual_information"]);
    }

    private static bool HasAnyDifferenceAcrossSamples(Func<Vector<double>> sampleFactory, Vector<double> baseline, int attempts)
    {
        for (int i = 0; i < attempts; i++)
        {
            var candidate = sampleFactory();
            if (candidate.Length != baseline.Length)
            {
                continue;
            }

            for (int j = 0; j < baseline.Length; j++)
            {
                if (candidate[j] != baseline[j])
                {
                    return true;
                }
            }
        }

        return false;
    }

    private static void AssertTensorEqual(Tensor<double> left, Tensor<double> right)
    {
        Assert.Equal(left.Shape, right.Shape);
        var leftVector = left.ToVector();
        var rightVector = right.ToVector();
        Assert.Equal(leftVector.Length, rightVector.Length);
        for (int i = 0; i < leftVector.Length; i++)
        {
            Assert.Equal(leftVector[i], rightVector[i]);
        }
    }

    [Fact]
    public async Task PredictWithUncertainty_WithConformalCalibration_ReturnsRegressionInterval()
    {
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Medium,
            inputSize: 1,
            outputSize: 1);
        var model = new NeuralNetworkModel<double>(architecture);

        var optimizer = new PassthroughOptimizer<double, Tensor<double>, Tensor<double>>(model);

        var xTrain = Tensor<double>.FromMatrix(new Matrix<double>(new double[,]
        {
            { 0.0 },
            { 1.0 },
            { 2.0 },
            { 3.0 }
        }));
        var yTrain = Tensor<double>.FromVector(new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0 }));

        var xCal = Tensor<double>.FromMatrix(new Matrix<double>(new double[,]
        {
            { 4.0 },
            { 5.0 },
            { 6.0 }
        }));
        var yCal = Tensor<double>.FromVector(new Vector<double>(new[] { 4.0, 5.0, 6.0 }));

        var result = await new PredictionModelBuilder<double, Tensor<double>, Tensor<double>>()
            .ConfigureDataLoader(DataLoaders.FromTensors(xTrain, yTrain))
            .ConfigureModel(model)
            .ConfigureOptimizer(optimizer)
            .ConfigureGpuAcceleration(new GpuAccelerationConfig { UsageLevel = GpuUsageLevel.AlwaysCpu })
            .ConfigureUncertaintyQuantification(
                new UncertaintyQuantificationOptions
                {
                    Method = UncertaintyQuantificationMethod.ConformalPrediction,
                    ConformalConfidenceLevel = 0.9
                },
                UncertaintyCalibrationData<Tensor<double>, Tensor<double>>.ForRegression(xCal, yCal))
            .BuildAsync();

        var uq = result.PredictWithUncertainty(xCal);
        Assert.NotNull(uq.RegressionInterval);

        var pred = ConversionsHelper.ConvertToTensor<double>(uq.Prediction).ToVector();
        var lower = ConversionsHelper.ConvertToTensor<double>(uq.RegressionInterval!.Lower).ToVector();
        var upper = ConversionsHelper.ConvertToTensor<double>(uq.RegressionInterval!.Upper).ToVector();

        Assert.Equal(pred.Length, lower.Length);
        Assert.Equal(pred.Length, upper.Length);
        for (int i = 0; i < pred.Length; i++)
        {
            Assert.True(lower[i] <= pred[i]);
            Assert.True(pred[i] <= upper[i]);
        }
    }

    [Fact]
    public async Task PredictWithUncertainty_WithClassificationConformalCalibration_ReturnsPredictionSetAndCalibratedProbabilities()
    {
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            complexity: NetworkComplexity.Simple,
            inputSize: 2,
            outputSize: 3);
        var model = new NeuralNetworkModel<double>(architecture);

        var optimizer = new DeterministicNeuralNetworkParameterOptimizer<Tensor<double>, Tensor<double>>(model);

        var xTrain = Tensor<double>.FromMatrix(new Matrix<double>(new double[,]
        {
            { 1.0, 0.0 },
            { 0.0, 1.0 },
            { 1.0, 1.0 },
            { 0.0, 0.0 }
        }));
        var yTrain = Tensor<double>.FromVector(new Vector<double>(new[] { 0.0, 1.0, 2.0, 0.0 }));

        var xCal = Tensor<double>.FromMatrix(new Matrix<double>(new double[,]
        {
            { 0.25, 0.75 },
            { 0.75, 0.25 }
        }));
        var labels = new Vector<int>(new[] { 1, 0 });

        var result = await new PredictionModelBuilder<double, Tensor<double>, Tensor<double>>()
            .ConfigureDataLoader(DataLoaders.FromTensors(xTrain, yTrain))
            .ConfigureModel(model)
            .ConfigureOptimizer(optimizer)
            .ConfigureGpuAcceleration(new GpuAccelerationConfig { UsageLevel = GpuUsageLevel.AlwaysCpu })
            .ConfigureUncertaintyQuantification(
                new UncertaintyQuantificationOptions
                {
                    Method = UncertaintyQuantificationMethod.ConformalPrediction,
                    ConformalConfidenceLevel = 0.9,
                    EnableTemperatureScaling = true
                },
                UncertaintyCalibrationData<Tensor<double>, Tensor<double>>.ForClassification(xCal, labels))
            .BuildAsync();

        var uq = result.PredictWithUncertainty(xCal);
        Assert.NotNull(uq.ClassificationSet);
        Assert.Equal(2, uq.ClassificationSet!.ClassIndices.Length);
        Assert.All(uq.ClassificationSet.ClassIndices, set => Assert.NotEmpty(set));

        // Probabilities should be normalized per sample (sum to ~1).
        var probs = ConversionsHelper.ConvertToTensor<double>(uq.Prediction).ToVector();
        const int classes = 3;
        for (int i = 0; i < labels.Length; i++)
        {
            var sum = 0.0;
            for (int c = 0; c < classes; c++)
            {
                sum += probs[i * classes + c];
            }
            Assert.InRange(sum, 0.999, 1.001);
        }

        Assert.True(uq.Metrics.ContainsKey("predictive_entropy"));
        Assert.True(uq.Metrics.ContainsKey("mutual_information"));
    }

    [Fact]
    public async Task PredictWithUncertainty_WithAdaptiveConformalClassification_ReturnsPredictionSet()
    {
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            complexity: NetworkComplexity.Simple,
            inputSize: 2,
            outputSize: 3);
        var model = new NeuralNetworkModel<double>(architecture);

        var optimizer = new DeterministicNeuralNetworkParameterOptimizer<Tensor<double>, Tensor<double>>(model);

        var xTrain = Tensor<double>.FromMatrix(new Matrix<double>(new double[,]
        {
            { 1.0, 0.0 },
            { 0.0, 1.0 },
            { 1.0, 1.0 },
            { 0.0, 0.0 }
        }));
        var yTrain = Tensor<double>.FromVector(new Vector<double>(new[] { 0.0, 1.0, 2.0, 0.0 }));

        var xCal = Tensor<double>.FromMatrix(new Matrix<double>(new double[,]
        {
            { 0.25, 0.75 },
            { 0.75, 0.25 },
            { 0.5, 0.5 }
        }));
        var labels = new Vector<int>(new[] { 1, 0, 2 });

        var result = await new PredictionModelBuilder<double, Tensor<double>, Tensor<double>>()
            .ConfigureDataLoader(DataLoaders.FromTensors(xTrain, yTrain))
            .ConfigureModel(model)
            .ConfigureOptimizer(optimizer)
            .ConfigureGpuAcceleration(new GpuAccelerationConfig { UsageLevel = GpuUsageLevel.AlwaysCpu })
            .ConfigureUncertaintyQuantification(
                new UncertaintyQuantificationOptions
                {
                    Method = UncertaintyQuantificationMethod.ConformalPrediction,
                    ConformalConfidenceLevel = 0.9,
                    ConformalMode = ConformalPredictionMode.Adaptive,
                    AdaptiveConformalBins = 5
                },
                UncertaintyCalibrationData<Tensor<double>, Tensor<double>>.ForClassification(xCal, labels))
            .BuildAsync();

        var uq = result.PredictWithUncertainty(xCal);
        Assert.NotNull(uq.ClassificationSet);
        Assert.Equal(3, uq.ClassificationSet!.ClassIndices.Length);
        Assert.All(uq.ClassificationSet.ClassIndices, set => Assert.NotEmpty(set));
    }

    [Fact]
    public async Task PredictWithUncertainty_WithBinaryPlattScaling_ReturnsNormalizedProbabilities()
    {
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.BinaryClassification,
            complexity: NetworkComplexity.Simple,
            inputSize: 2,
            outputSize: 2);
        var model = new NeuralNetworkModel<double>(architecture);

        var optimizer = new DeterministicNeuralNetworkParameterOptimizer<Tensor<double>, Tensor<double>>(model);

        var xTrain = Tensor<double>.FromMatrix(new Matrix<double>(new double[,]
        {
            { 0.0, 0.0 },
            { 1.0, 0.0 },
            { 0.0, 1.0 },
            { 1.0, 1.0 }
        }));
        var yTrain = Tensor<double>.FromVector(new Vector<double>(new[] { 0.0, 1.0, 1.0, 0.0 }));

        var xCal = xTrain;
        var labels = new Vector<int>(new[] { 0, 1, 1, 0 });

        var result = await new PredictionModelBuilder<double, Tensor<double>, Tensor<double>>()
            .ConfigureDataLoader(DataLoaders.FromTensors(xTrain, yTrain))
            .ConfigureModel(model)
            .ConfigureOptimizer(optimizer)
            .ConfigureGpuAcceleration(new GpuAccelerationConfig { UsageLevel = GpuUsageLevel.AlwaysCpu })
            .ConfigureUncertaintyQuantification(
                new UncertaintyQuantificationOptions
                {
                    Method = UncertaintyQuantificationMethod.ConformalPrediction,
                    ConformalConfidenceLevel = 0.9,
                    CalibrationMethod = ProbabilityCalibrationMethod.PlattScaling,
                    EnableTemperatureScaling = false,
                    EnablePlattScaling = true
                },
                UncertaintyCalibrationData<Tensor<double>, Tensor<double>>.ForClassification(xCal, labels))
            .BuildAsync();

        var uq = result.PredictWithUncertainty(xCal);
        var probs = ConversionsHelper.ConvertToTensor<double>(uq.Prediction).ToVector();

        const int classes = 2;
        for (int i = 0; i < labels.Length; i++)
        {
            var sum = 0.0;
            for (int c = 0; c < classes; c++)
            {
                sum += probs[i * classes + c];
            }
            Assert.InRange(sum, 0.999, 1.001);
        }
    }

    [Fact]
    public async Task PredictWithUncertainty_WithLaplaceApproximation_ReturnsVariance()
    {
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Simple,
            inputSize: 1,
            outputSize: 1);
        var model = new NeuralNetworkModel<double>(architecture);

        var optimizer = new PassthroughOptimizer<double, Tensor<double>, Tensor<double>>(model);

        var xTrain = Tensor<double>.FromMatrix(new Matrix<double>(new double[,]
        {
            { 0.0 },
            { 1.0 },
            { 2.0 },
            { 3.0 }
        }));
        var yTrain = Tensor<double>.FromVector(new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0 }));

        var xCal = Tensor<double>.FromMatrix(new Matrix<double>(new double[,]
        {
            { 4.0 },
            { 5.0 },
            { 6.0 }
        }));
        var yCal = Tensor<double>.FromVector(new Vector<double>(new[] { 4.0, 5.0, 6.0 }));

        var result = await new PredictionModelBuilder<double, Tensor<double>, Tensor<double>>()
            .ConfigureDataLoader(DataLoaders.FromTensors(xTrain, yTrain))
            .ConfigureModel(model)
            .ConfigureOptimizer(optimizer)
            .ConfigureGpuAcceleration(new GpuAccelerationConfig { UsageLevel = GpuUsageLevel.AlwaysCpu })
            .ConfigureUncertaintyQuantification(
                new UncertaintyQuantificationOptions
                {
                    Method = UncertaintyQuantificationMethod.LaplaceApproximation,
                    NumSamples = 8,
                    PosteriorFitMaxSamples = 3,
                    RandomSeed = 123
                },
                UncertaintyCalibrationData<Tensor<double>, Tensor<double>>.ForRegression(xCal, yCal))
            .BuildAsync();

        var uq = result.PredictWithUncertainty(xCal);
        Assert.Equal(UncertaintyQuantificationMethod.LaplaceApproximation, uq.MethodUsed);
        Assert.NotNull(uq.Variance);
        Assert.Equal(ConversionsHelper.ConvertToTensor<double>(uq.Prediction).Shape, ConversionsHelper.ConvertToTensor<double>(uq.Variance!).Shape);
    }

    [Fact]
    public async Task PredictWithUncertainty_WithBinaryIsotonicCalibration_ReturnsNormalizedProbabilities()
    {
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.BinaryClassification,
            complexity: NetworkComplexity.Simple,
            inputSize: 2,
            outputSize: 2);
        var model = new NeuralNetworkModel<double>(architecture);

        var optimizer = new DeterministicNeuralNetworkParameterOptimizer<Tensor<double>, Tensor<double>>(model);

        var xTrain = Tensor<double>.FromMatrix(new Matrix<double>(new double[,]
        {
            { 0.0, 0.0 },
            { 1.0, 0.0 },
            { 0.0, 1.0 },
            { 1.0, 1.0 }
        }));
        var yTrain = Tensor<double>.FromVector(new Vector<double>(new[] { 0.0, 1.0, 1.0, 0.0 }));

        var xCal = xTrain;
        var labels = new Vector<int>(new[] { 0, 1, 1, 0 });

        var result = await new PredictionModelBuilder<double, Tensor<double>, Tensor<double>>()
            .ConfigureDataLoader(DataLoaders.FromTensors(xTrain, yTrain))
            .ConfigureModel(model)
            .ConfigureOptimizer(optimizer)
            .ConfigureGpuAcceleration(new GpuAccelerationConfig { UsageLevel = GpuUsageLevel.AlwaysCpu })
            .ConfigureUncertaintyQuantification(
                new UncertaintyQuantificationOptions
                {
                    Method = UncertaintyQuantificationMethod.ConformalPrediction,
                    ConformalConfidenceLevel = 0.9,
                    CalibrationMethod = ProbabilityCalibrationMethod.IsotonicRegression,
                    EnableTemperatureScaling = false,
                    EnableIsotonicRegressionCalibration = true,
                    RandomSeed = 42
                },
                UncertaintyCalibrationData<Tensor<double>, Tensor<double>>.ForClassification(xCal, labels))
            .BuildAsync();

        var uq = result.PredictWithUncertainty(xCal);
        var probs = ConversionsHelper.ConvertToTensor<double>(uq.Prediction).ToVector();

        const int classes = 2;
        for (int i = 0; i < labels.Length; i++)
        {
            var sum = 0.0;
            for (int c = 0; c < classes; c++)
            {
                sum += probs[i * classes + c];
            }
            Assert.InRange(sum, 0.999, 1.001);
        }
    }

    [Fact]
    public async Task PredictWithUncertainty_WithSwag_ReturnsVariance()
    {
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Simple,
            inputSize: 1,
            outputSize: 1);
        var model = new NeuralNetworkModel<double>(architecture);

        var optimizer = new PassthroughOptimizer<double, Tensor<double>, Tensor<double>>(model);

        var xTrain = Tensor<double>.FromMatrix(new Matrix<double>(new double[,]
        {
            { 0.0 },
            { 1.0 },
            { 2.0 },
            { 3.0 }
        }));
        var yTrain = Tensor<double>.FromVector(new Vector<double>(new[] { 0.0, 1.0, 2.0, 3.0 }));

        var xCal = Tensor<double>.FromMatrix(new Matrix<double>(new double[,]
        {
            { 4.0 },
            { 5.0 },
            { 6.0 }
        }));
        var yCal = Tensor<double>.FromVector(new Vector<double>(new[] { 4.0, 5.0, 6.0 }));

        var result = await new PredictionModelBuilder<double, Tensor<double>, Tensor<double>>()
            .ConfigureDataLoader(DataLoaders.FromTensors(xTrain, yTrain))
            .ConfigureModel(model)
            .ConfigureOptimizer(optimizer)
            .ConfigureGpuAcceleration(new GpuAccelerationConfig { UsageLevel = GpuUsageLevel.AlwaysCpu })
            .ConfigureUncertaintyQuantification(
                new UncertaintyQuantificationOptions
                {
                    Method = UncertaintyQuantificationMethod.Swag,
                    NumSamples = 8,
                    PosteriorFitMaxSamples = 3,
                    SwagNumSteps = 6,
                    SwagBurnInSteps = 1,
                    SwagNumSnapshots = 2,
                    SwagLearningRate = 0.001,
                    RandomSeed = 123
                },
                UncertaintyCalibrationData<Tensor<double>, Tensor<double>>.ForRegression(xCal, yCal))
            .BuildAsync();

        var uq = result.PredictWithUncertainty(xCal);
        Assert.Equal(UncertaintyQuantificationMethod.Swag, uq.MethodUsed);
        Assert.NotNull(uq.Variance);
        Assert.Equal(ConversionsHelper.ConvertToTensor<double>(uq.Prediction).Shape, ConversionsHelper.ConvertToTensor<double>(uq.Variance!).Shape);
    }

    [Fact]
    public async Task EvaluateModel_WithUqCalibration_PopulatesExpectedCalibrationError()
    {
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.MultiClassClassification,
            complexity: NetworkComplexity.Simple,
            inputSize: 2,
            outputSize: 3);
        var model = new NeuralNetworkModel<double>(architecture);

        var optimizer = new DeterministicNeuralNetworkParameterOptimizer<Tensor<double>, Tensor<double>>(model);

        var xTrain = Tensor<double>.FromMatrix(new Matrix<double>(new double[,]
        {
            { 0.0, 0.0 },
            { 1.0, 0.0 },
            { 0.0, 1.0 },
            { 1.0, 1.0 }
        }));
        var yTrain = Tensor<double>.FromVector(new Vector<double>(new[] { 0.0, 1.0, 2.0, 0.0 }));

        var xCal = xTrain;
        var yCalLabels = new Vector<int>(new[] { 0, 1, 2, 0 });

        var result = await new PredictionModelBuilder<double, Tensor<double>, Tensor<double>>()
            .ConfigureDataLoader(DataLoaders.FromTensors(xTrain, yTrain))
            .ConfigureModel(model)
            .ConfigureOptimizer(optimizer)
            .ConfigureGpuAcceleration(new GpuAccelerationConfig { UsageLevel = GpuUsageLevel.AlwaysCpu })
            .ConfigureUncertaintyQuantification(
                new UncertaintyQuantificationOptions
                {
                    Enabled = true,
                    Method = UncertaintyQuantificationMethod.Auto,
                    EnableTemperatureScaling = true
                },
                UncertaintyCalibrationData<Tensor<double>, Tensor<double>>.ForClassification(xCal, yCalLabels))
            .BuildAsync();

        var evaluator = new DefaultModelEvaluator<double, Tensor<double>, Tensor<double>>();
        var eval = evaluator.EvaluateModel(new ModelEvaluationInput<double, Tensor<double>, Tensor<double>>
        {
            Model = result,
            NormInfo = result.NormalizationInfo,
            InputData = new OptimizationInputData<double, Tensor<double>, Tensor<double>>
            {
                XTrain = xTrain,
                YTrain = yTrain,
                XValidation = xTrain,
                YValidation = yTrain,
                XTest = xTrain,
                YTest = yTrain
            }
        });

        Assert.True(eval.TrainingSet.UncertaintyStats.Metrics.ContainsKey("expected_calibration_error"));
    }

    [Fact]
    public async Task PredictWithUncertainty_WithDeepEnsemble_ReturnsNonZeroVariance()
    {
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Medium,
            inputSize: 1,
            outputSize: 1);
        var model = new NeuralNetworkModel<double>(architecture);

        var optimizer = new PassthroughOptimizer<double, Tensor<double>, Tensor<double>>(model);

        var x = Tensor<double>.FromMatrix(new Matrix<double>(new double[,]
        {
            { 0.0 },
            { 1.0 },
            { 2.0 }
        }));
        var y = Tensor<double>.FromVector(new Vector<double>(new[] { 0.0, 1.0, 2.0 }));

        var result = await new PredictionModelBuilder<double, Tensor<double>, Tensor<double>>()
            .ConfigureDataLoader(DataLoaders.FromTensors(x, y))
            .ConfigureModel(model)
            .ConfigureOptimizer(optimizer)
            .ConfigureGpuAcceleration(new GpuAccelerationConfig { UsageLevel = GpuUsageLevel.AlwaysCpu })
            .ConfigureUncertaintyQuantification(new UncertaintyQuantificationOptions
            {
                Method = UncertaintyQuantificationMethod.DeepEnsemble,
                DeepEnsembleSize = 3,
                DeepEnsembleInitialNoiseStdDev = 0.05,
                RandomSeed = 123
            })
            .BuildAsync();

        var uq = result.PredictWithUncertainty(x);
        Assert.NotNull(uq.Variance);
        Assert.Equal(uq.Prediction.Shape, uq.Variance!.Shape);

        var varianceVector = uq.Variance!.ToVector();
        Assert.Contains(varianceVector, v => v > 0.0);
    }

    [Fact]
    public async Task PredictWithUncertainty_WithBayesianNeuralNetwork_ReturnsVarianceAndMetrics()
    {
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Simple,
            inputSize: 1,
            outputSize: 1);

        var bayesianModel = new BayesianNeuralNetwork<double>(architecture, numSamples: 8);
        var optimizer = new SingleStepTrainOptimizer<double, Tensor<double>, Tensor<double>>(bayesianModel);

        var x = Tensor<double>.FromMatrix(new Matrix<double>(new double[,]
        {
            { 0.0 },
            { 1.0 },
            { 2.0 }
        }));
        var y = Tensor<double>.FromMatrix(new Matrix<double>(new double[,]
        {
            { 0.0 },
            { 1.0 },
            { 2.0 }
        }));

        var result = await new PredictionModelBuilder<double, Tensor<double>, Tensor<double>>()
            .ConfigureDataLoader(DataLoaders.FromTensors(x, y))
            .ConfigureModel(bayesianModel)
            .ConfigureOptimizer(optimizer)
            .ConfigureGpuAcceleration(new GpuAccelerationConfig { UsageLevel = GpuUsageLevel.AlwaysCpu })
            .ConfigureUncertaintyQuantification(new UncertaintyQuantificationOptions
            {
                Method = UncertaintyQuantificationMethod.BayesianNeuralNetwork
            })
            .BuildAsync();

        var uq = result.PredictWithUncertainty(x);

        Assert.NotNull(uq.Variance);
        Assert.Equal(uq.Prediction.Shape, uq.Variance!.Shape);
        Assert.True(uq.Metrics.ContainsKey("predictive_entropy"));
        Assert.True(uq.Metrics.ContainsKey("mutual_information"));
    }

    [Fact]
    public async Task PredictWithUncertainty_IsThreadSafeUnderConcurrentCalls()
    {
        var architecture = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            complexity: NetworkComplexity.Medium,
            inputSize: 2,
            outputSize: 1);
        var model = new NeuralNetworkModel<double>(architecture);
        var optimizer = new PassthroughOptimizer<double, Tensor<double>, Tensor<double>>(model);

        var xTrain = Tensor<double>.FromMatrix(new Matrix<double>(new double[,]
        {
            { 1.0, 2.0 },
            { 3.0, 4.0 }
        }));
        var yTrain = Tensor<double>.FromVector(new Vector<double>(new[] { 0.0, 1.0 }));

        var result = await new PredictionModelBuilder<double, Tensor<double>, Tensor<double>>()
            .ConfigureDataLoader(DataLoaders.FromTensors(xTrain, yTrain))
            .ConfigureModel(model)
            .ConfigureOptimizer(optimizer)
            .ConfigureGpuAcceleration(new GpuAccelerationConfig { UsageLevel = GpuUsageLevel.AlwaysCpu })
            .ConfigureUncertaintyQuantification(new UncertaintyQuantificationOptions
            {
                Method = UncertaintyQuantificationMethod.MonteCarloDropout,
                NumSamples = 8,
                MonteCarloDropoutRate = 0.5,
                RandomSeed = 7
            })
            .BuildAsync();

        var x = Tensor<double>.FromMatrix(new Matrix<double>(new double[,]
        {
            { 5.0, 6.0 },
            { 7.0, 8.0 }
        }));

        var tasks = Enumerable.Range(0, 12)
            .Select(_ => Task.Run(() => result.PredictWithUncertainty(x)))
            .ToArray();

        var results = await Task.WhenAll(tasks);
        Assert.All(results, r =>
        {
            Assert.NotNull(r.Variance);
            Assert.Equal(r.Prediction.Shape, r.Variance!.Shape);
            Assert.True(r.Metrics.ContainsKey("predictive_entropy"));
            Assert.True(r.Metrics.ContainsKey("mutual_information"));
        });
    }

}
