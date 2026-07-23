using System.Threading.Tasks;
using AiDotNet.Configuration;
using AiDotNet.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;
using Moq;
using Xunit;

namespace AiDotNet.Tests.UnitTests.Configuration;

/// <summary>Audit-2026-05 phase-2a slice 3 — cross-validation component isolation tests.</summary>
public class AiModelCrossValidationTests
{
    [Fact(Timeout = 30000)]
    public async Task InitialState_IsNull()
    {
        await Task.Yield();
        var cv = new AiModelCrossValidation<double, Matrix<double>, Vector<double>>();
        Assert.Null(cv.CrossValidator);
    }

    [Fact(Timeout = 30000)]
    public async Task Configure_Stores()
    {
        await Task.Yield();
        var cv = new AiModelCrossValidation<double, Matrix<double>, Vector<double>>();
        var validator = Mock.Of<ICrossValidator<double, Matrix<double>, Vector<double>>>();

        cv.ConfigureCrossValidation(validator);

        Assert.Same(validator, cv.CrossValidator);
    }

    [Fact(Timeout = 30000)]
    public async Task Interface_IsImplemented()
    {
        await Task.Yield();
        IAiModelCrossValidation<double, Matrix<double>, Vector<double>> cv
            = new AiModelCrossValidation<double, Matrix<double>, Vector<double>>();
        var validator = Mock.Of<ICrossValidator<double, Matrix<double>, Vector<double>>>();
        cv.ConfigureCrossValidation(validator);
        Assert.Same(validator, cv.CrossValidator);
    }
}
