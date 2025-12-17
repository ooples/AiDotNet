using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.Regression;

namespace AiDotNet.AutoML.Registry;

/// <summary>
/// Creates built-in tabular candidate models for AutoML trials.
/// </summary>
/// <remarks>
/// <para>
/// This factory is used by built-in AutoML strategies to keep model construction logic centralized and consistent.
/// </para>
/// <para>
/// <b>For Beginners:</b> AutoML needs a way to create many models with different settings. This class builds those
/// models for each trial.
/// </para>
/// </remarks>
internal static class AutoMLTabularModelFactory<T>
{
    public static IFullModel<T, Matrix<T>, Vector<T>> Create(ModelType modelType, IReadOnlyDictionary<string, object> parameters)
    {
        return modelType switch
        {
            ModelType.SimpleRegression => CreateWithOptions(
                (RegressionOptions<T> options) => new SimpleRegression<T>(options),
                new RegressionOptions<T>(),
                parameters),

            ModelType.MultipleRegression => CreateWithOptions(
                (RegressionOptions<T> options) => new MultipleRegression<T>(options),
                new RegressionOptions<T>(),
                parameters),

            ModelType.PolynomialRegression => CreateWithOptions(
                (PolynomialRegressionOptions<T> options) => new PolynomialRegression<T>(options),
                new PolynomialRegressionOptions<T>(),
                parameters),

            ModelType.LogisticRegression => CreateWithOptions(
                (LogisticRegressionOptions<T> options) => new LogisticRegression<T>(options),
                new LogisticRegressionOptions<T>(),
                parameters),

            ModelType.MultinomialLogisticRegression => CreateWithOptions(
                (MultinomialLogisticRegressionOptions<T> options) => new MultinomialLogisticRegression<T>(options),
                new MultinomialLogisticRegressionOptions<T>(),
                parameters),

            ModelType.RandomForest => CreateWithOptions(
                (RandomForestRegressionOptions options) => new RandomForestRegression<T>(options),
                new RandomForestRegressionOptions(),
                parameters),

            ModelType.GradientBoosting => CreateWithOptions(
                (GradientBoostingRegressionOptions options) => new GradientBoostingRegression<T>(options),
                new GradientBoostingRegressionOptions(),
                parameters),

            ModelType.KNearestNeighbors => CreateWithOptions(
                (KNearestNeighborsOptions options) => new KNearestNeighborsRegression<T>(options),
                new KNearestNeighborsOptions(),
                parameters),

            ModelType.SupportVectorRegression => CreateWithOptions(
                (SupportVectorRegressionOptions options) => new SupportVectorRegression<T>(options),
                new SupportVectorRegressionOptions(),
                parameters),

            ModelType.TimeSeriesRegression => CreateWithOptions(
                (ARIMAOptions<T> options) => new TimeSeriesRegression<T>(options),
                new ARIMAOptions<T>(),
                parameters),

            ModelType.NeuralNetworkRegression => new NeuralNetworkRegression<T>(),

            _ => throw new NotSupportedException(
                $"AutoML model type '{modelType}' is not currently supported by the built-in tabular AutoML factory.")
        };
    }

    private static TModel CreateWithOptions<TOptions, TModel>(
        Func<TOptions, TModel> factory,
        TOptions options,
        IReadOnlyDictionary<string, object> parameters)
    {
        if (factory is null)
        {
            throw new ArgumentNullException(nameof(factory));
        }

        if (options is null)
        {
            throw new ArgumentNullException(nameof(options));
        }

        AutoMLHyperparameterApplicator.ApplyToOptions(options, parameters);
        return factory(options);
    }
}
