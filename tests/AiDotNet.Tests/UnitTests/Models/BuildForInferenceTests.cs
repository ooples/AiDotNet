using System;
using AiDotNet.Enums;
using AiDotNet.Models;
using AiDotNet.NeuralNetworks;
using AiDotNet.Regression;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Models;

/// <summary>
/// Covers <c>AiModelBuilder.BuildForInference()</c>, the terminal for a model that arrives already
/// trained.
/// </summary>
/// <remarks>
/// <para>
/// <c>Build</c> trains, and <c>AiModelResult</c>'s constructors are internal, so a model that was
/// loaded from weights, returned by another algorithm, or built from known parameters had no way into
/// the facade except by being retrained — which for an adapted or loaded model destroys the thing that
/// made it worth having.
/// </para>
/// </remarks>
public class BuildForInferenceTests
{
    private static Matrix<double> Features()
    {
        double[,] rows = { { 3, 1500, 2 }, { 4, 2100, 3 }, { 2, 900, 1 }, { 5, 3000, 3 } };
        var m = new Matrix<double>(rows.GetLength(0), rows.GetLength(1));
        for (int i = 0; i < rows.GetLength(0); i++)
        {
            for (int j = 0; j < rows.GetLength(1); j++)
            {
                m[i, j] = rows[i, j];
            }
        }

        return m;
    }

    /// <summary>A model whose coefficients are known up front — nothing here needs learning.</summary>
    private static VectorModel<double> AlreadyTrained() =>
        new(new Vector<double>(new double[] { 50000, 100, 20000 }));

    /// <summary>
    /// The point of the terminal: the wrapped model's own predictions, unchanged. Anything that
    /// retrained it would move these.
    /// </summary>
    [Fact]
    public void ItPredictsWhatTheWrappedModelPredicts()
    {
        var features = Features();
        var model = AlreadyTrained();

        var result = new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureModel(model)
            .BuildForInference();

        var viaFacade = result.Predict(features);
        var direct = model.Predict(features);

        Assert.Equal(direct.Length, viaFacade.Length);
        for (int i = 0; i < direct.Length; i++)
        {
            Assert.Equal(direct[i], viaFacade[i]);
        }
    }

    /// <summary>
    /// The defect this closes, stated as a test: building the same model the training way changes its
    /// predictions, because training is what <c>Build</c> does.
    /// </summary>
    [Fact]
    public void TrainingWouldHaveChangedThem()
    {
        var features = Features();
        var prices = new Vector<double>(new double[] { 320000, 415000, 210000, 560000 });

        var inference = new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureModel(AlreadyTrained())
            .BuildForInference()
            .Predict(features);

        var trained = new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureModel(AlreadyTrained())
            .Build(features, prices)
            .Predict(features);

        bool differs = false;
        for (int i = 0; i < inference.Length; i++)
        {
            if (inference[i] != trained[i])
            {
                differs = true;
                break;
            }
        }

        Assert.True(
            differs,
            "Build and BuildForInference produced the same predictions, so either training did nothing " +
            "or BuildForInference trained after all.");
    }

    /// <summary>
    /// The result is a real one, not a stub: the model-specific accessors dispatch through it the same
    /// way they do on a built result.
    /// </summary>
    [Fact]
    public void TheResultIsAFullOne()
    {
        var result = new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureModel(AlreadyTrained())
            .BuildForInference();

        // A VectorModel is not a decomposition, so the accessor's guard should fire — which proves the
        // accessor reached the wrapped model rather than finding nothing there.
        var ex = Assert.Throws<NotSupportedException>(() => result.GetTrend());
        Assert.Contains(nameof(VectorModel<double>), ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ItWorksForANeuralNetworkOverTensors()
    {
        var architecture = new NeuralNetworkArchitecture<float>(
            InputType.OneDimensional, NeuralNetworkTaskType.Regression, inputSize: 8, outputSize: 1);
        var pretrained = new NeuralNetwork<float>(architecture);

        var result = new AiModelBuilder<float, Tensor<float>, Tensor<float>>()
            .ConfigureModel(pretrained)
            .BuildForInference();

        var input = Tensor<float>.CreateRandom(1, 8);

        var viaFacade = result.Predict(input);
        var direct = pretrained.Predict(input);

        Assert.Equal(direct.Shape[0], viaFacade.Shape[0]);
    }

    [Fact]
    public void NoModelConfigured_SaysSo()
    {
        var ex = Assert.Throws<InvalidOperationException>(() =>
            new AiModelBuilder<double, Matrix<double>, Vector<double>>().BuildForInference());

        Assert.Contains("needs a model", ex.Message, StringComparison.Ordinal);
    }

    /// <summary>
    /// There is no training pass to fit a preprocessing pipeline, so an unfitted one has to be refused
    /// rather than left to fail confusingly on the first prediction.
    /// </summary>
    [Fact]
    public void AnUnfittedPreprocessingPipeline_IsRefusedUpFront()
    {
        // Configured but never fitted, because nothing here trains.
        var builder = new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureModel(AlreadyTrained())
            .ConfigurePreprocessing();

        var ex = Assert.Throws<InvalidOperationException>(() => builder.BuildForInference());

        Assert.Contains("has not been fitted", ex.Message, StringComparison.Ordinal);
    }

    /// <summary>
    /// Repeated calls must not accumulate state on the builder or the model.
    /// </summary>
    [Fact]
    public void ItCanBeCalledTwice()
    {
        var features = Features();
        var builder = new AiModelBuilder<double, Matrix<double>, Vector<double>>()
            .ConfigureModel(AlreadyTrained());

        var first = builder.BuildForInference().Predict(features);
        var second = builder.BuildForInference().Predict(features);

        for (int i = 0; i < first.Length; i++)
        {
            Assert.Equal(first[i], second[i]);
        }
    }
}
