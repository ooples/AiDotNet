using AiDotNet.Interfaces;
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
    private static readonly Dictionary<Type, Func<IReadOnlyDictionary<string, object>, IFullModel<T, Matrix<T>, Vector<T>>>> _factories = new()
    {
        [typeof(SimpleRegression<>)] = p => CreateWithOptions(
            (RegressionOptions<T> o) => new SimpleRegression<T>(o), new RegressionOptions<T>(), p),

        [typeof(MultipleRegression<>)] = p => CreateWithOptions(
            (RegressionOptions<T> o) => new MultipleRegression<T>(o), new RegressionOptions<T>(), p),

        [typeof(BayesianRegression<>)] = p => CreateWithOptions(
            (BayesianRegressionOptions<T> o) => new BayesianRegression<T>(o), new BayesianRegressionOptions<T>(), p),

        [typeof(PolynomialRegression<>)] = p => CreateWithOptions(
            (PolynomialRegressionOptions<T> o) => new PolynomialRegression<T>(o), new PolynomialRegressionOptions<T>(), p),

        [typeof(QuantileRegression<>)] = p => CreateWithOptions(
            (QuantileRegressionOptions<T> o) => new QuantileRegression<T>(o), new QuantileRegressionOptions<T>(), p),

        [typeof(RobustRegression<>)] = p => CreateWithOptions(
            (RobustRegressionOptions<T> o) => new RobustRegression<T>(o), new RobustRegressionOptions<T>(), p),

        [typeof(LogisticRegression<>)] = p => CreateWithOptions(
            (LogisticRegressionOptions<T> o) => new LogisticRegression<T>(o), new LogisticRegressionOptions<T>(), p),

        [typeof(MultinomialLogisticRegression<>)] = p => CreateWithOptions(
            (MultinomialLogisticRegressionOptions<T> o) => new MultinomialLogisticRegression<T>(o), new MultinomialLogisticRegressionOptions<T>(), p),

        [typeof(GaussianProcessRegression<>)] = p => CreateWithOptions(
            (GaussianProcessRegressionOptions o) => new GaussianProcessRegression<T>(o), new GaussianProcessRegressionOptions(), p),

        [typeof(RandomForestRegression<>)] = p => CreateWithOptions(
            (RandomForestRegressionOptions o) => new RandomForestRegression<T>(o), new RandomForestRegressionOptions(), p),

        [typeof(GradientBoostingRegression<>)] = p => CreateWithOptions(
            (GradientBoostingRegressionOptions o) => new GradientBoostingRegression<T>(o), new GradientBoostingRegressionOptions(), p),

        [typeof(DecisionTreeRegression<>)] = p => CreateWithOptions(
            (DecisionTreeOptions o) => new DecisionTreeRegression<T>(o), new DecisionTreeOptions(), p),

        [typeof(ExtremelyRandomizedTreesRegression<>)] = p => CreateWithOptions(
            (ExtremelyRandomizedTreesRegressionOptions o) => new ExtremelyRandomizedTreesRegression<T>(o), new ExtremelyRandomizedTreesRegressionOptions(), p),

        [typeof(AdaBoostR2Regression<>)] = p => CreateWithOptions(
            (AdaBoostR2RegressionOptions o) => new AdaBoostR2Regression<T>(o), new AdaBoostR2RegressionOptions(), p),

        [typeof(QuantileRegressionForests<>)] = p => CreateWithOptions(
            (QuantileRegressionForestsOptions o) => new QuantileRegressionForests<T>(o), new QuantileRegressionForestsOptions(), p),

        [typeof(ConditionalInferenceTreeRegression<>)] = p => CreateWithOptions(
            (ConditionalInferenceTreeOptions o) => new ConditionalInferenceTreeRegression<T>(o), new ConditionalInferenceTreeOptions(), p),

        [typeof(M5ModelTree<>)] = p => CreateWithOptions(
            (M5ModelTreeOptions o) => new M5ModelTree<T>(o), new M5ModelTreeOptions(), p),

        [typeof(KNearestNeighborsRegression<>)] = p => CreateWithOptions(
            (KNearestNeighborsOptions o) => new KNearestNeighborsRegression<T>(o), new KNearestNeighborsOptions(), p),

        [typeof(SupportVectorRegression<>)] = p => CreateWithOptions(
            (SupportVectorRegressionOptions o) => new SupportVectorRegression<T>(o), new SupportVectorRegressionOptions(), p),

        [typeof(KernelRidgeRegression<>)] = p => CreateWithOptions(
            (KernelRidgeRegressionOptions o) => new KernelRidgeRegression<T>(o), new KernelRidgeRegressionOptions(), p),

        [typeof(GeneralizedAdditiveModel<>)] = p => CreateWithOptions(
            (GeneralizedAdditiveModelOptions<T> o) => new GeneralizedAdditiveModel<T>(o), new GeneralizedAdditiveModelOptions<T>(), p),

        [typeof(RadialBasisFunctionRegression<>)] = p => CreateWithOptions(
            (RadialBasisFunctionOptions o) => new RadialBasisFunctionRegression<T>(o), new RadialBasisFunctionOptions(), p),

        [typeof(MultilayerPerceptronRegression<>)] = p => CreateWithOptions(
            (MultilayerPerceptronOptions<T, Matrix<T>, Vector<T>> o) => new MultilayerPerceptronRegression<T>(o),
            new MultilayerPerceptronOptions<T, Matrix<T>, Vector<T>>(), p),

        [typeof(TimeSeriesRegression<>)] = p => CreateWithOptions(
            (ARIMAOptions<T> o) => new TimeSeriesRegression<T>(o), new ARIMAOptions<T>(), p),

        [typeof(NeuralNetworkRegression<>)] = p => CreateWithOptions(
            (NeuralNetworkRegressionOptions<T, Matrix<T>, Vector<T>> o) => new NeuralNetworkRegression<T>(o),
            new NeuralNetworkRegressionOptions<T, Matrix<T>, Vector<T>>(), p),
    };

    public static IFullModel<T, Matrix<T>, Vector<T>> Create(Type modelType, IReadOnlyDictionary<string, object> parameters)
    {
        if (modelType is null)
            throw new ArgumentNullException(nameof(modelType));

        var lookupType = modelType.IsGenericType ? modelType.GetGenericTypeDefinition() : modelType;

        if (_factories.TryGetValue(lookupType, out var factory))
        {
            return factory(parameters);
        }

        throw new NotSupportedException(
            $"AutoML model type '{modelType.Name}' is not currently supported by the built-in tabular AutoML factory.");
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
