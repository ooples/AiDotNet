using AiDotNet.Enums;
using AiDotNet.Factories;
using AiDotNet.Interfaces;
using AiDotNet.Models;
using AiDotNet.PromptEngineering.Templates;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.Factories;

/// <summary>
/// Integration tests for factory classes:
/// ActivationFunctionFactory, WindowFunctionFactory, RegularizationFactory,
/// FitnessCalculatorFactory, MatrixDecompositionFactory, PromptTemplateFactory.
/// </summary>
public class FactoriesIntegrationTests
{
    #region ActivationFunctionFactory - Single Value

    [Theory]
    [InlineData(ActivationFunction.ReLU)]
    [InlineData(ActivationFunction.Sigmoid)]
    [InlineData(ActivationFunction.Tanh)]
    [InlineData(ActivationFunction.Linear)]
    [InlineData(ActivationFunction.Identity)]
    [InlineData(ActivationFunction.LeakyReLU)]
    [InlineData(ActivationFunction.ELU)]
    [InlineData(ActivationFunction.SELU)]
    [InlineData(ActivationFunction.Softplus)]
    [InlineData(ActivationFunction.SoftSign)]
    [InlineData(ActivationFunction.Swish)]
    [InlineData(ActivationFunction.GELU)]
    [InlineData(ActivationFunction.LiSHT)]
    public void ActivationFunctionFactory_CreateActivation_ReturnsInstance(ActivationFunction type)
    {
        var activation = ActivationFunctionFactory<double>.CreateActivationFunction(type);
        Assert.NotNull(activation);
        Assert.IsAssignableFrom<IActivationFunction<double>>(activation);
    }

    [Fact]
    public void ActivationFunctionFactory_Softmax_ThrowsForSingleValue()
    {
        Assert.Throws<NotSupportedException>(() =>
            ActivationFunctionFactory<double>.CreateActivationFunction(ActivationFunction.Softmax));
    }

    [Fact]
    public void ActivationFunctionFactory_InvalidType_Throws()
    {
        Assert.Throws<ArgumentException>(() =>
            ActivationFunctionFactory<double>.CreateActivationFunction((ActivationFunction)999));
    }

    #endregion

    #region ActivationFunctionFactory - Vector

    [Theory]
    [InlineData(ActivationFunction.Softmax)]
    [InlineData(ActivationFunction.ReLU)]
    [InlineData(ActivationFunction.Sigmoid)]
    [InlineData(ActivationFunction.Tanh)]
    [InlineData(ActivationFunction.Linear)]
    [InlineData(ActivationFunction.Identity)]
    [InlineData(ActivationFunction.LeakyReLU)]
    [InlineData(ActivationFunction.ELU)]
    [InlineData(ActivationFunction.SELU)]
    [InlineData(ActivationFunction.Softplus)]
    [InlineData(ActivationFunction.SoftSign)]
    [InlineData(ActivationFunction.Swish)]
    [InlineData(ActivationFunction.GELU)]
    [InlineData(ActivationFunction.LiSHT)]
    public void ActivationFunctionFactory_CreateVector_ReturnsInstance(ActivationFunction type)
    {
        var activation = ActivationFunctionFactory<double>.CreateVectorActivationFunction(type);
        Assert.NotNull(activation);
        Assert.IsAssignableFrom<IVectorActivationFunction<double>>(activation);
    }

    [Fact]
    public void ActivationFunctionFactory_CreateVector_InvalidType_Throws()
    {
        Assert.Throws<ArgumentException>(() =>
            ActivationFunctionFactory<double>.CreateVectorActivationFunction((ActivationFunction)999));
    }

    #endregion

    #region WindowFunctionFactory

    [Theory]
    [InlineData(WindowFunctionType.Rectangular)]
    [InlineData(WindowFunctionType.Hanning)]
    [InlineData(WindowFunctionType.Hamming)]
    [InlineData(WindowFunctionType.Blackman)]
    [InlineData(WindowFunctionType.Kaiser)]
    [InlineData(WindowFunctionType.Bartlett)]
    [InlineData(WindowFunctionType.Gaussian)]
    [InlineData(WindowFunctionType.BartlettHann)]
    [InlineData(WindowFunctionType.Bohman)]
    [InlineData(WindowFunctionType.Lanczos)]
    [InlineData(WindowFunctionType.Parzen)]
    [InlineData(WindowFunctionType.Poisson)]
    [InlineData(WindowFunctionType.Nuttall)]
    [InlineData(WindowFunctionType.Triangular)]
    [InlineData(WindowFunctionType.BlackmanHarris)]
    [InlineData(WindowFunctionType.FlatTop)]
    [InlineData(WindowFunctionType.Welch)]
    [InlineData(WindowFunctionType.BlackmanNuttall)]
    [InlineData(WindowFunctionType.Cosine)]
    [InlineData(WindowFunctionType.Tukey)]
    public void WindowFunctionFactory_Create_ReturnsInstance(WindowFunctionType type)
    {
        var window = WindowFunctionFactory.CreateWindowFunction<double>(type);
        Assert.NotNull(window);
        Assert.IsAssignableFrom<IWindowFunction<double>>(window);
    }

    [Fact]
    public void WindowFunctionFactory_InvalidType_Throws()
    {
        Assert.Throws<ArgumentException>(() =>
            WindowFunctionFactory.CreateWindowFunction<double>((WindowFunctionType)999));
    }

    #endregion

    #region RegularizationFactory

    [Fact]
    public void RegularizationFactory_NoRegularization()
    {
        var options = new RegularizationOptions { Type = RegularizationType.None };
        var reg = RegularizationFactory.CreateRegularization<double, Matrix<double>, Vector<double>>(options);
        Assert.NotNull(reg);
        Assert.IsAssignableFrom<IRegularization<double, Matrix<double>, Vector<double>>>(reg);
    }

    [Fact]
    public void RegularizationFactory_L1()
    {
        var options = new RegularizationOptions { Type = RegularizationType.L1, Strength = 0.01 };
        var reg = RegularizationFactory.CreateRegularization<double, Matrix<double>, Vector<double>>(options);
        Assert.NotNull(reg);
    }

    [Fact]
    public void RegularizationFactory_L2()
    {
        var options = new RegularizationOptions { Type = RegularizationType.L2, Strength = 0.01 };
        var reg = RegularizationFactory.CreateRegularization<double, Matrix<double>, Vector<double>>(options);
        Assert.NotNull(reg);
    }

    [Fact]
    public void RegularizationFactory_ElasticNet()
    {
        var options = new RegularizationOptions { Type = RegularizationType.ElasticNet, Strength = 0.01 };
        var reg = RegularizationFactory.CreateRegularization<double, Matrix<double>, Vector<double>>(options);
        Assert.NotNull(reg);
    }

    [Fact]
    public void RegularizationFactory_GetType_RoundTrip()
    {
        var options = new RegularizationOptions { Type = RegularizationType.L1 };
        var reg = RegularizationFactory.CreateRegularization<double, Matrix<double>, Vector<double>>(options);
        var detectedType = RegularizationFactory.GetRegularizationType(reg);
        Assert.Equal(RegularizationType.L1, detectedType);
    }

    [Fact]
    public void RegularizationFactory_GetType_AllTypes()
    {
        var types = new[] { RegularizationType.None, RegularizationType.L1, RegularizationType.L2, RegularizationType.ElasticNet };
        foreach (var type in types)
        {
            var options = new RegularizationOptions { Type = type };
            var reg = RegularizationFactory.CreateRegularization<double, Matrix<double>, Vector<double>>(options);
            var detected = RegularizationFactory.GetRegularizationType(reg);
            Assert.Equal(type, detected);
        }
    }

    #endregion

    #region FitnessCalculatorFactory

    [Theory]
    [InlineData(FitnessCalculatorType.MeanSquaredError)]
    [InlineData(FitnessCalculatorType.MeanAbsoluteError)]
    [InlineData(FitnessCalculatorType.RSquared)]
    [InlineData(FitnessCalculatorType.AdjustedRSquared)]
    [InlineData(FitnessCalculatorType.OrdinalRegressionLoss)]
    [InlineData(FitnessCalculatorType.HuberLoss)]
    [InlineData(FitnessCalculatorType.RootMeanSquaredError)]
    [InlineData(FitnessCalculatorType.ExponentialLoss)]
    public void FitnessCalculatorFactory_Create_ReturnsInstance(FitnessCalculatorType type)
    {
        var calc = FitnessCalculatorFactory.CreateFitnessCalculator<double, Matrix<double>, Vector<double>>(type);
        Assert.NotNull(calc);
        Assert.IsAssignableFrom<IFitnessCalculator<double, Matrix<double>, Vector<double>>>(calc);
    }

    [Fact]
    public void FitnessCalculatorFactory_Custom_Throws()
    {
        Assert.Throws<ArgumentException>(() =>
            FitnessCalculatorFactory.CreateFitnessCalculator<double, Matrix<double>, Vector<double>>(FitnessCalculatorType.Custom));
    }

    [Fact]
    public void FitnessCalculatorFactory_InvalidType_Throws()
    {
        Assert.Throws<ArgumentException>(() =>
            FitnessCalculatorFactory.CreateFitnessCalculator<double, Matrix<double>, Vector<double>>((FitnessCalculatorType)999));
    }

    #endregion

    #region MatrixDecompositionFactory

    [Theory]
    [InlineData(MatrixDecompositionType.Lu)]
    [InlineData(MatrixDecompositionType.Qr)]
    [InlineData(MatrixDecompositionType.Svd)]
    [InlineData(MatrixDecompositionType.Eigen)]
    [InlineData(MatrixDecompositionType.Schur)]
    [InlineData(MatrixDecompositionType.Hessenberg)]
    [InlineData(MatrixDecompositionType.Bidiagonal)]
    public void MatrixDecompositionFactory_Create_ReturnsInstance(MatrixDecompositionType type)
    {
        // Use a simple 3x3 identity matrix for decomposition
        var matrix = Matrix<double>.CreateIdentity(3);
        var decomp = MatrixDecompositionFactory.CreateDecomposition(matrix, type);
        Assert.NotNull(decomp);
        Assert.IsAssignableFrom<IMatrixDecomposition<double>>(decomp);
    }

    [Fact]
    public void MatrixDecompositionFactory_Cholesky_SPDMatrix()
    {
        // Cholesky requires SPD matrix
        var matrix = Matrix<double>.CreateIdentity(3);
        var decomp = MatrixDecompositionFactory.CreateDecomposition(matrix, MatrixDecompositionType.Cholesky);
        Assert.NotNull(decomp);
    }

    [Fact]
    public void MatrixDecompositionFactory_GetType_RoundTrip()
    {
        var matrix = Matrix<double>.CreateIdentity(3);
        var decomp = MatrixDecompositionFactory.CreateDecomposition(matrix, MatrixDecompositionType.Qr);
        var detectedType = MatrixDecompositionFactory.GetDecompositionType(decomp);
        Assert.Equal(MatrixDecompositionType.Qr, detectedType);
    }

    [Fact]
    public void MatrixDecompositionFactory_InvalidType_Throws()
    {
        var matrix = Matrix<double>.CreateIdentity(3);
        Assert.Throws<ArgumentException>(() =>
            MatrixDecompositionFactory.CreateDecomposition(matrix, (MatrixDecompositionType)999));
    }

    [Fact]
    public void MatrixDecompositionFactory_GetType_Null_Throws()
    {
        Assert.Throws<ArgumentException>(() =>
            MatrixDecompositionFactory.GetDecompositionType<double>(null));
    }

    #endregion

    #region PromptTemplateFactory

    [Fact]
    public void PromptTemplateFactory_Simple_ReturnsInstance()
    {
        var template = PromptTemplateFactory.Create(PromptTemplateType.Simple, "Hello {name}");
        Assert.NotNull(template);
        Assert.IsAssignableFrom<IPromptTemplate>(template);
    }

    [Fact]
    public void PromptTemplateFactory_Chat_ReturnsInstance()
    {
        var template = PromptTemplateFactory.Create(PromptTemplateType.Chat);
        Assert.NotNull(template);
        Assert.IsAssignableFrom<ChatPromptTemplate>(template);
    }

    [Fact]
    public void PromptTemplateFactory_ChainOfThought_ReturnsInstance()
    {
        var template = PromptTemplateFactory.Create(PromptTemplateType.ChainOfThought, "Solve {problem}");
        Assert.NotNull(template);
    }

    [Fact]
    public void PromptTemplateFactory_ReAct_ReturnsInstance()
    {
        var template = PromptTemplateFactory.Create(PromptTemplateType.ReAct, "Answer this question.");
        Assert.NotNull(template);
    }

    [Fact]
    public void PromptTemplateFactory_Simple_NullTemplate_Throws()
    {
        Assert.Throws<ArgumentException>(() =>
            PromptTemplateFactory.Create(PromptTemplateType.Simple, null));
    }

    [Fact]
    public void PromptTemplateFactory_Simple_EmptyTemplate_Throws()
    {
        Assert.Throws<ArgumentException>(() =>
            PromptTemplateFactory.Create(PromptTemplateType.Simple, ""));
    }

    [Fact]
    public void PromptTemplateFactory_FewShot_WithoutSelector_Throws()
    {
        Assert.Throws<ArgumentException>(() =>
            PromptTemplateFactory.Create(PromptTemplateType.FewShot, "template {input}"));
    }

    [Fact]
    public void PromptTemplateFactory_InvalidType_Throws()
    {
        Assert.Throws<ArgumentException>(() =>
            PromptTemplateFactory.Create((PromptTemplateType)999));
    }

    [Fact]
    public void PromptTemplateFactory_ChainOfThought_DefaultTemplate()
    {
        // When no template is provided, should use default
        var template = PromptTemplateFactory.Create(PromptTemplateType.ChainOfThought);
        Assert.NotNull(template);
    }

    [Fact]
    public void PromptTemplateFactory_ReAct_DefaultTemplate()
    {
        var template = PromptTemplateFactory.Create(PromptTemplateType.ReAct);
        Assert.NotNull(template);
    }

    #endregion

    #region Cross-Factory - Activation Functions Are Callable

    [Fact]
    public void ActivationFunction_ReLU_Activate_Positive()
    {
        var relu = ActivationFunctionFactory<double>.CreateActivationFunction(ActivationFunction.ReLU);
        var result = relu.Activate(5.0);
        Assert.Equal(5.0, result, 1e-10);
    }

    [Fact]
    public void ActivationFunction_ReLU_Activate_Negative()
    {
        var relu = ActivationFunctionFactory<double>.CreateActivationFunction(ActivationFunction.ReLU);
        var result = relu.Activate(-5.0);
        Assert.Equal(0.0, result, 1e-10);
    }

    [Fact]
    public void ActivationFunction_Sigmoid_Activate_Zero()
    {
        var sigmoid = ActivationFunctionFactory<double>.CreateActivationFunction(ActivationFunction.Sigmoid);
        var result = sigmoid.Activate(0.0);
        Assert.Equal(0.5, result, 1e-10);
    }

    [Fact]
    public void ActivationFunction_Identity_Activate_Passthrough()
    {
        var identity = ActivationFunctionFactory<double>.CreateActivationFunction(ActivationFunction.Identity);
        var result = identity.Activate(42.0);
        Assert.Equal(42.0, result, 1e-10);
    }

    #endregion

    #region Cross-Factory - Window Functions Generate Correct Size

    [Fact]
    public void WindowFunction_Rectangular_GeneratesCorrectSize()
    {
        var window = WindowFunctionFactory.CreateWindowFunction<double>(WindowFunctionType.Rectangular);
        var coefficients = window.Create(16);
        Assert.Equal(16, coefficients.Length);
    }

    [Fact]
    public void WindowFunction_Hamming_GeneratesCorrectSize()
    {
        var window = WindowFunctionFactory.CreateWindowFunction<double>(WindowFunctionType.Hamming);
        var coefficients = window.Create(32);
        Assert.Equal(32, coefficients.Length);
    }

    #endregion
}
