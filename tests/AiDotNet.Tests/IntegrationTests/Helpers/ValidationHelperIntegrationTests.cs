using AiDotNet.Helpers;
using Xunit;
using System.Threading.Tasks;

namespace AiDotNet.Tests.IntegrationTests.Helpers;

/// <summary>
/// Integration tests for ValidationHelper:
/// ValidateInputData (Matrix/Vector and Tensor pairs), ValidatePoissonData,
/// GetCallerInfo, ResolveCallerInfo.
/// </summary>
public class ValidationHelperIntegrationTests
{
    #region ValidateInputData - Matrix/Vector

    [Fact(Timeout = 120000)]
    public async Task ValidateInputData_ValidMatrixVector_DoesNotThrow()
    {
        var x = new Matrix<double>(3, 2);
        x[0, 0] = 1.0; x[0, 1] = 2.0;
        x[1, 0] = 3.0; x[1, 1] = 4.0;
        x[2, 0] = 5.0; x[2, 1] = 6.0;
        var y = new Vector<double>(new double[] { 1.0, 2.0, 3.0 });

        var ex = Record.Exception(() => ValidationHelper<double>.ValidateInputData<Matrix<double>, Vector<double>>(x, y));
        Assert.Null(ex);
    }

    [Fact(Timeout = 120000)]
    public async Task ValidateInputData_MismatchedRows_Throws()
    {
        var x = new Matrix<double>(3, 2);
        var y = new Vector<double>(new double[] { 1.0, 2.0 }); // 2 != 3

        Assert.Throws<ArgumentException>(() =>
            ValidationHelper<double>.ValidateInputData<Matrix<double>, Vector<double>>(x, y));
    }

    [Fact(Timeout = 120000)]
    public async Task ValidateInputData_EmptyMatrix_Throws()
    {
        var x = new Matrix<double>(0, 0);
        var y = new Vector<double>(0);

        Assert.Throws<ArgumentException>(() =>
            ValidationHelper<double>.ValidateInputData<Matrix<double>, Vector<double>>(x, y));
    }

    #endregion

    #region ValidateInputData - Tensor pairs

    [Fact(Timeout = 120000)]
    public async Task ValidateInputData_ValidTensorPair_DoesNotThrow()
    {
        var x = new Tensor<double>(new[] { 3, 4 });
        var y = new Tensor<double>(new[] { 3, 1 });

        var ex = Record.Exception(() => ValidationHelper<double>.ValidateInputData<Tensor<double>, Tensor<double>>(x, y));
        Assert.Null(ex);
    }

    [Fact(Timeout = 120000)]
    public async Task ValidateInputData_TensorMismatchedFirstDim_Throws()
    {
        var x = new Tensor<double>(new[] { 3, 4 });
        var y = new Tensor<double>(new[] { 5, 1 }); // 5 != 3

        Assert.Throws<ArgumentException>(() =>
            ValidationHelper<double>.ValidateInputData<Tensor<double>, Tensor<double>>(x, y));
    }

    [Fact(Timeout = 120000)]
    public async Task ValidateInputData_TensorZeroDimension_Throws()
    {
        var x = new Tensor<double>(new[] { 0, 4 });
        var y = new Tensor<double>(new[] { 0, 1 });

        Assert.Throws<ArgumentException>(() =>
            ValidationHelper<double>.ValidateInputData<Tensor<double>, Tensor<double>>(x, y));
    }

    #endregion

    #region ValidatePoissonData

    [Fact(Timeout = 120000)]
    public async Task ValidatePoissonData_ValidData_DoesNotThrow()
    {
        var y = new Vector<double>(new double[] { 0.0, 1.0, 2.0, 5.0, 10.0 });
        var ex = Record.Exception(() => ValidationHelper<double>.ValidatePoissonData(y));
        Assert.Null(ex);
    }

    [Fact(Timeout = 120000)]
    public async Task ValidatePoissonData_NegativeValue_Throws()
    {
        var y = new Vector<double>(new double[] { 1.0, -1.0, 3.0 });
        Assert.Throws<ArgumentException>(() => ValidationHelper<double>.ValidatePoissonData(y));
    }

    [Fact(Timeout = 120000)]
    public async Task ValidatePoissonData_NonInteger_Throws()
    {
        var y = new Vector<double>(new double[] { 1.0, 2.5, 3.0 });
        Assert.Throws<ArgumentException>(() => ValidationHelper<double>.ValidatePoissonData(y));
    }

    #endregion

    #region GetCallerInfo

    [Fact(Timeout = 120000)]
    public async Task GetCallerInfo_ReturnsNonEmptyValues()
    {
        var (component, operation) = ValidationHelper<double>.GetCallerInfo(1);
        Assert.False(string.IsNullOrEmpty(component));
        Assert.False(string.IsNullOrEmpty(operation));
    }

    #endregion

    #region ResolveCallerInfo

    [Fact(Timeout = 120000)]
    public async Task ResolveCallerInfo_WithExplicitValues_ReturnsProvided()
    {
        var (component, operation) = ValidationHelper<double>.ResolveCallerInfo("MyComponent", "MyOperation");
        Assert.Equal("MyComponent", component);
        Assert.Equal("MyOperation", operation);
    }

    [Fact(Timeout = 120000)]
    public async Task ResolveCallerInfo_EmptyComponent_ResolvesFromCaller()
    {
        var (component, operation) = ValidationHelper<double>.ResolveCallerInfo("", "MyOperation");
        Assert.False(string.IsNullOrEmpty(component));
        Assert.Equal("MyOperation", operation);
    }

    [Fact(Timeout = 120000)]
    public async Task ResolveCallerInfo_EmptyOperation_ResolvesFromCaller()
    {
        var (component, operation) = ValidationHelper<double>.ResolveCallerInfo("MyComponent", "");
        Assert.Equal("MyComponent", component);
        Assert.False(string.IsNullOrEmpty(operation));
    }

    #endregion
}
