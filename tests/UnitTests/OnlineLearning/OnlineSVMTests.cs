using Microsoft.VisualStudio.TestTools.UnitTesting;
using AiDotNet.Kernels;
using AiDotNet.LinearAlgebra;
using AiDotNet.Models.Options;
using AiDotNet.OnlineLearning.Algorithms;

namespace AiDotNetTests.UnitTests.OnlineLearning;

[TestClass]
public class OnlineSVMTests
{
    [TestMethod]
    public void Constructor_Linear_ShouldInitializeCorrectly()
    {
        // Arrange & Act
        var svm = new OnlineSVM<double>(3);

        // Assert
        Assert.IsNotNull(svm);
        Assert.AreEqual(3, svm.GetInputFeatureCount());
    }

    [TestMethod]
    public void Constructor_WithKernel_ShouldInitializeCorrectly()
    {
        // Arrange
        var kernel = new GaussianKernel<double>(1.0);

        // Act
        var svm = new OnlineSVM<double>(3, null, kernel);

        // Assert
        Assert.IsNotNull(svm);
        Assert.AreEqual(3, svm.GetInputFeatureCount());
    }

    [TestMethod]
    public void PartialFit_LinearSVM_ShouldClassifyCorrectly()
    {
        // Arrange
        var svm = new OnlineSVM<double>(2);
        var inputs = new[]
        {
            new Vector<double>(new[] { 2.0, 3.0 }),
            new Vector<double>(new[] { 3.0, 3.0 }),
            new Vector<double>(new[] { 1.0, 1.0 }),
            new Vector<double>(new[] { 2.0, 1.0 })
        };
        var labels = new[] { 1.0, 1.0, -1.0, -1.0 };

        // Act - train multiple epochs
        for (int epoch = 0; epoch < 50; epoch++)
        {
            for (int i = 0; i < inputs.Length; i++)
            {
                svm.PartialFit(inputs[i], labels[i]);
            }
        }

        // Assert
        Assert.IsTrue(svm.Predict(new Vector<double>(new[] { 2.5, 3.0 })) > 0); // Should be positive
        Assert.IsTrue(svm.Predict(new Vector<double>(new[] { 1.5, 1.0 })) < 0); // Should be negative
    }

    [TestMethod]
    public void PartialFit_KernelSVM_ShouldHandleNonLinearData()
    {
        // Arrange
        var kernel = new GaussianKernel<double>(0.5);
        var svm = new OnlineSVM<double>(2, null, kernel);
        
        // XOR-like problem
        var inputs = new[]
        {
            new Vector<double>(new[] { -1.0, -1.0 }),
            new Vector<double>(new[] { -1.0, 1.0 }),
            new Vector<double>(new[] { 1.0, -1.0 }),
            new Vector<double>(new[] { 1.0, 1.0 })
        };
        var labels = new[] { 1.0, -1.0, -1.0, 1.0 }; // XOR pattern

        // Act - train multiple epochs
        for (int epoch = 0; epoch < 100; epoch++)
        {
            for (int i = 0; i < inputs.Length; i++)
            {
                svm.PartialFit(inputs[i], labels[i]);
            }
        }

        // Assert - check if it learned the XOR pattern
        Assert.IsTrue(svm.Predict(inputs[0]) > 0); // (−1,−1) → +1
        Assert.IsTrue(svm.Predict(inputs[1]) < 0); // (−1,+1) → -1
        Assert.IsTrue(svm.Predict(inputs[2]) < 0); // (+1,−1) → -1
        Assert.IsTrue(svm.Predict(inputs[3]) > 0); // (+1,+1) → +1
    }

    [TestMethod]
    public void PartialFitBatch_ShouldUpdateModel()
    {
        // Arrange
        var svm = new OnlineSVM<double>(2);
        var inputs = new Matrix<double>(new double[,]
        {
            { 1.0, 2.0 },
            { 2.0, 3.0 },
            { -1.0, -1.0 },
            { -2.0, -1.0 }
        });
        var labels = new Vector<double>(new[] { 1.0, 1.0, -1.0, -1.0 });

        // Act
        for (int i = 0; i < 20; i++)
        {
            // Convert Matrix to array of Vectors
            var inputVectors = new Vector<double>[inputs.Rows];
            for (int j = 0; j < inputs.Rows; j++)
            {
                inputVectors[j] = new Vector<double>(new[] { inputs[j, 0], inputs[j, 1] });
            }
            svm.PartialFitBatch(inputVectors, labels.ToArray());
        }

        // Assert
        var testPositive = new Vector<double>(new[] { 1.5, 2.5 });
        var testNegative = new Vector<double>(new[] { -1.5, -1.0 });
        Assert.IsTrue(svm.Predict(testPositive) > 0);
        Assert.IsTrue(svm.Predict(testNegative) < 0);
    }

    [TestMethod]
    public void WithDifferentC_ShouldAffectMargin()
    {
        // Arrange
        var optionsLowC = new OnlineModelOptions<double> { RegularizationParameter = 0.01 };
        var optionsHighC = new OnlineModelOptions<double> { RegularizationParameter = 100.0 };
        var svmLowC = new OnlineSVM<double>(2, optionsLowC);
        var svmHighC = new OnlineSVM<double>(2, optionsHighC);

        var inputs = new[]
        {
            new Vector<double>(new[] { 1.0, 1.0 }),
            new Vector<double>(new[] { -1.0, -1.0 })
        };
        var labels = new[] { 1.0, -1.0 };

        // Act - train both models
        for (int epoch = 0; epoch < 50; epoch++)
        {
            for (int i = 0; i < inputs.Length; i++)
            {
                svmLowC.PartialFit(inputs[i], labels[i]);
                svmHighC.PartialFit(inputs[i], labels[i]);
            }
        }

        // Assert - models should behave differently
        var testPoint = new Vector<double>(new[] { 0.1, 0.1 });
        var predLowC = svmLowC.Predict(testPoint);
        var predHighC = svmHighC.Predict(testPoint);
        
        // With different C values, the decision boundaries should be different
        Assert.AreNotEqual(Math.Sign(predLowC), Math.Sign(predHighC), 0.1);
    }

    [TestMethod]
    public void Clone_ShouldCreateIndependentCopy()
    {
        // Arrange
        var svm = new OnlineSVM<double>(2);
        var input = new Vector<double>(new[] { 1.0, 1.0 });
        svm.PartialFit(input, 1.0);

        // Act
        var clone = svm.Clone() as OnlineSVM<double>;
        svm.PartialFit(new Vector<double>(new[] { -1.0, -1.0 }), -1.0);

        // Assert
        Assert.IsNotNull(clone);
        var originalPred = svm.Predict(input);
        var clonePred = clone.Predict(input);
        Assert.AreNotEqual(originalPred, clonePred);
    }

    [TestMethod]
    public void GetSetParameters_Linear_ShouldWorkCorrectly()
    {
        // Arrange
        var svm = new OnlineSVM<double>(2);
        // Parameters are weights + bias
        var parameters = new Vector<double>(new[] { 0.5, -0.5, 1.0 });

        // Act
        svm.SetParameters(parameters);
        var retrieved = svm.GetParameters();

        // Assert
        Assert.IsNotNull(retrieved);
        Assert.AreEqual(3, retrieved.Length); // 2 weights + 1 bias
        Assert.AreEqual(0.5, retrieved[0]);
        Assert.AreEqual(-0.5, retrieved[1]);
        Assert.AreEqual(1.0, retrieved[2]); // bias
    }

    [TestMethod]
    public void KernelSVM_ShouldStoreSupportVectors()
    {
        // Arrange
        var kernel = new LinearKernel<double>();
        var svm = new OnlineSVM<double>(2, null, kernel);
        var inputs = new[]
        {
            new Vector<double>(new[] { 1.0, 1.0 }),
            new Vector<double>(new[] { -1.0, -1.0 })
        };
        var labels = new[] { 1.0, -1.0 };

        // Act
        for (int i = 0; i < inputs.Length; i++)
        {
            svm.PartialFit(inputs[i], labels[i]);
        }

        // Assert
        var parameters = svm.GetParameters();
        Assert.IsNotNull(parameters);
        // For kernelized SVM, parameters contain alphas + bias
        Assert.IsTrue(parameters.Length > 0);
        Assert.IsTrue(svm.IsKernelized);
        Assert.IsTrue(svm.SupportVectorCount > 0);
    }
}