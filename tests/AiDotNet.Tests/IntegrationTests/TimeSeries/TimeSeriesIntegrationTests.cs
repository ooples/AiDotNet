using AiDotNet.Models.Options;
using AiDotNet.TimeSeries;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.TimeSeries;

/// <summary>
/// Integration tests for TimeSeries models.
/// </summary>
public class TimeSeriesIntegrationTests
{
    #region ARModel Tests

    [Fact]
    public void ARModel_Construction_WithDefaultOptions_Succeeds()
    {
        var options = new ARModelOptions<double>();
        var model = new ARModel<double>(options);

        Assert.NotNull(model);
    }

    [Fact]
    public void ARModel_Construction_WithCustomAROrder_Succeeds()
    {
        var options = new ARModelOptions<double> { AROrder = 5 };
        var model = new ARModel<double>(options);

        Assert.NotNull(model);
    }

    [Fact]
    public void ARModel_Construction_WithCustomLearningRate_Succeeds()
    {
        var options = new ARModelOptions<double> { LearningRate = 0.001 };
        var model = new ARModel<double>(options);

        Assert.NotNull(model);
    }

    [Fact]
    public void ARModel_Float_Construction_Succeeds()
    {
        var options = new ARModelOptions<float>();
        var model = new ARModel<float>(options);

        Assert.NotNull(model);
    }

    [Fact]
    public void ARModel_Construction_WithDifferentAROrders_Succeeds()
    {
        var model1 = new ARModel<double>(new ARModelOptions<double> { AROrder = 1 });
        var model2 = new ARModel<double>(new ARModelOptions<double> { AROrder = 3 });
        var model3 = new ARModel<double>(new ARModelOptions<double> { AROrder = 7 });

        Assert.NotNull(model1);
        Assert.NotNull(model2);
        Assert.NotNull(model3);
    }

    #endregion

    #region MAModel Tests

    [Fact]
    public void MAModel_Construction_WithDefaultOptions_Succeeds()
    {
        var options = new MAModelOptions<double>();
        var model = new MAModel<double>(options);

        Assert.NotNull(model);
    }

    [Fact]
    public void MAModel_Construction_WithCustomMAOrder_Succeeds()
    {
        var options = new MAModelOptions<double> { MAOrder = 3 };
        var model = new MAModel<double>(options);

        Assert.NotNull(model);
    }

    [Fact]
    public void MAModel_Float_Construction_Succeeds()
    {
        var options = new MAModelOptions<float>();
        var model = new MAModel<float>(options);

        Assert.NotNull(model);
    }

    #endregion

    #region ARMAModel Tests

    [Fact]
    public void ARMAModel_Construction_WithDefaultOptions_Succeeds()
    {
        var options = new ARMAOptions<double>();
        var model = new ARMAModel<double>(options);

        Assert.NotNull(model);
    }

    [Fact]
    public void ARMAModel_Float_Construction_Succeeds()
    {
        var options = new ARMAOptions<float>();
        var model = new ARMAModel<float>(options);

        Assert.NotNull(model);
    }

    #endregion

    #region ARIMAModel Tests

    [Fact]
    public void ARIMAModel_Construction_WithDefaultOptions_Succeeds()
    {
        var options = new ARIMAOptions<double>();
        var model = new ARIMAModel<double>(options);

        Assert.NotNull(model);
    }

    [Fact]
    public void ARIMAModel_Float_Construction_Succeeds()
    {
        var options = new ARIMAOptions<float>();
        var model = new ARIMAModel<float>(options);

        Assert.NotNull(model);
    }

    #endregion

    #region SARIMAModel Tests

    [Fact]
    public void SARIMAModel_Construction_WithDefaultOptions_Succeeds()
    {
        var options = new SARIMAOptions<double>();
        var model = new SARIMAModel<double>(options);

        Assert.NotNull(model);
    }

    [Fact]
    public void SARIMAModel_Float_Construction_Succeeds()
    {
        var options = new SARIMAOptions<float>();
        var model = new SARIMAModel<float>(options);

        Assert.NotNull(model);
    }

    #endregion

    #region ExponentialSmoothingModel Tests

    [Fact]
    public void ExponentialSmoothingModel_Construction_WithDefaultOptions_Succeeds()
    {
        var options = new ExponentialSmoothingOptions<double>();
        var model = new ExponentialSmoothingModel<double>(options);

        Assert.NotNull(model);
    }

    [Fact]
    public void ExponentialSmoothingModel_Float_Construction_Succeeds()
    {
        var options = new ExponentialSmoothingOptions<float>();
        var model = new ExponentialSmoothingModel<float>(options);

        Assert.NotNull(model);
    }

    #endregion

    #region GARCHModel Tests

    [Fact]
    public void GARCHModel_Construction_WithDefaultOptions_Succeeds()
    {
        var options = new GARCHModelOptions<double>();
        var model = new GARCHModel<double>(options);

        Assert.NotNull(model);
    }

    [Fact]
    public void GARCHModel_Float_Construction_Succeeds()
    {
        var options = new GARCHModelOptions<float>();
        var model = new GARCHModel<float>(options);

        Assert.NotNull(model);
    }

    #endregion

    #region StateSpaceModel Tests

    [Fact]
    public void StateSpaceModel_Construction_WithDefaultOptions_Succeeds()
    {
        var options = new StateSpaceModelOptions<double>();
        var model = new StateSpaceModel<double>(options);

        Assert.NotNull(model);
    }

    [Fact]
    public void StateSpaceModel_Float_Construction_Succeeds()
    {
        var options = new StateSpaceModelOptions<float>();
        var model = new StateSpaceModel<float>(options);

        Assert.NotNull(model);
    }

    #endregion

    #region VectorAutoRegressionModel Tests

    [Fact]
    public void VectorAutoRegressionModel_Construction_WithDefaultOptions_Succeeds()
    {
        var options = new VARModelOptions<double>();
        var model = new VectorAutoRegressionModel<double>(options);

        Assert.NotNull(model);
    }

    [Fact]
    public void VectorAutoRegressionModel_Float_Construction_Succeeds()
    {
        var options = new VARModelOptions<float>();
        var model = new VectorAutoRegressionModel<float>(options);

        Assert.NotNull(model);
    }

    #endregion

    #region Cross-Model Tests

    [Fact]
    public void AllTimeSeriesModels_ImplementITimeSeriesModel()
    {
        var arModel = new ARModel<double>(new ARModelOptions<double>());
        var maModel = new MAModel<double>(new MAModelOptions<double>());
        var armaModel = new ARMAModel<double>(new ARMAOptions<double>());
        var arimaModel = new ARIMAModel<double>(new ARIMAOptions<double>());

        // All models should be non-null
        Assert.NotNull(arModel);
        Assert.NotNull(maModel);
        Assert.NotNull(armaModel);
        Assert.NotNull(arimaModel);
    }

    #endregion
}
