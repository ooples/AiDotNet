using System;
using System.Linq;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tests.ModelFamilyTests.Base;

/// <summary>
/// The contract of a model a meta-learner's <c>Adapt</c> returns (MetaLearningModelBase: AdaptedMetaModel,
/// MbPAAdaptedModel, NeuralProcessModel, LFTAdaptedModel), shared by the scalar and classifier adapted-model families
/// (#2139).
/// </summary>
/// <remarks>
/// An adapted model is theta'_i for one task (Finn et al. 2017): it is built by the algorithm, not trained directly, so
/// <c>Train</c> refuses; it predicts the same query the same way every time; its parameters are a complete description
/// of it, so writing back what it reports changes nothing; and a deep copy shares no state with it.
/// </remarks>
internal static class AdaptedMetaModelChecks
{
    public static void ReturnsTheModelUnderTest(object adapted, Type expected)
    {
        var actual = adapted.GetType();
        Assert.Equal(expected, actual.IsGenericType ? actual.GetGenericTypeDefinition() : actual);
    }

    public static void PredictsDeterministically<TOutput>(IFullModel<double, Matrix<double>, TOutput> adapted,
        Matrix<double> query, Func<TOutput, double[]> values)
    {
        var first = values(adapted.Predict(query));
        var second = values(adapted.Predict(query));
        Assert.Equal(first, second);
        for (int i = 0; i < first.Length; i++)
            Assert.False(double.IsNaN(first[i]) || double.IsInfinity(first[i]), $"Adapted prediction {i} is {first[i]}.");
    }

    public static void ParametersDescribeTheModel<TOutput>(IFullModel<double, Matrix<double>, TOutput> adapted,
        Matrix<double> query, Func<TOutput, double[]> values)
    {
        var parameters = adapted.GetParameters();
        Assert.Equal(adapted.ParameterCount, parameters.Length);
        Assert.True(parameters.Length > 0, "An adapted model reports no parameters, so nothing it learned is visible.");
        var before = values(adapted.Predict(query));

        var rebuilt = adapted.WithParameters(parameters);

        Assert.Equal(before, values(rebuilt.Predict(query)));
    }

    public static void DeepCopyIsIndependent<TOutput>(IFullModel<double, Matrix<double>, TOutput> adapted,
        Matrix<double> query, Func<TOutput, double[]> values)
    {
        var before = values(adapted.Predict(query));
        var copy = adapted.DeepCopy();
        Assert.NotSame(adapted, copy);
        Assert.Equal(before, values(copy.Predict(query)));

        var moved = copy.GetParameters().ToArray().Select(p => p + 0.5).ToArray();
        copy.SetParameters(new Vector<double>(moved));

        Assert.Equal(before, values(adapted.Predict(query)));
    }

    public static void RefusesDirectTraining<TOutput>(IFullModel<double, Matrix<double>, TOutput> adapted,
        Matrix<double> query, TOutput target)
        => Assert.Throws<NotSupportedException>(() => adapted.Train(query, target));
}