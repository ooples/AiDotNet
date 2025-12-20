using AiDotNet.Autodiff;
using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.KnowledgeDistillation;
using AiDotNet.KnowledgeDistillation.Teachers;
using Xunit;
using JitCompilerClass = AiDotNet.JitCompiler.JitCompiler;

namespace AiDotNet.Tests.UnitTests.JitCompiler;

/// <summary>
/// Tests for JIT compilation support in Knowledge Distillation teacher models.
/// Verifies conditional JIT support based on underlying model capabilities.
/// </summary>
/// <remarks>
/// These tests are quarantined because they trigger GPU initialization which can fail
/// on machines without proper GPU support or drivers.
/// </remarks>
[Trait("Category", "GPU")]
public class KnowledgeDistillationJitCompilationTests
{
    // ========== EnsembleTeacherModel Tests ==========

    [Fact]
    public void EnsembleTeacherModel_SupportsJit_WhenAllTeachersSupportJit()
    {
        // Arrange - Create JIT-compatible mock teachers
        var teacher1 = CreateJitCompatibleTeacher();
        var teacher2 = CreateJitCompatibleTeacher();

        var ensemble = new EnsembleTeacherModel<double>(
            new[] { teacher1, teacher2 },
            new double[] { 0.5, 0.5 },
            EnsembleAggregationMode.WeightedAverage);

        // Assert
        Assert.True(ensemble.SupportsJitCompilation,
            "EnsembleTeacherModel should support JIT when all teachers support JIT");
    }

    [Fact]
    public void EnsembleTeacherModel_DoesNotSupportJit_WhenAnyTeacherDoesNotSupportJit()
    {
        // Arrange - Create one JIT-compatible and one non-JIT-compatible teacher
        var jitTeacher = CreateJitCompatibleTeacher();
        var nonJitTeacher = CreateNonJitTeacher();

        var ensemble = new EnsembleTeacherModel<double>(
            new ITeacherModel<Vector<double>, Vector<double>>[] { jitTeacher, nonJitTeacher },
            new double[] { 0.5, 0.5 },
            EnsembleAggregationMode.WeightedAverage);

        // Assert
        Assert.False(ensemble.SupportsJitCompilation,
            "EnsembleTeacherModel should not support JIT when any teacher doesn't support JIT");
    }

    [Fact]
    public void EnsembleTeacherModel_DoesNotSupportJit_WhenAggregationIsNotWeightedAverage()
    {
        // Arrange
        var teacher1 = CreateJitCompatibleTeacher();
        var teacher2 = CreateJitCompatibleTeacher();

        var ensemble = new EnsembleTeacherModel<double>(
            new[] { teacher1, teacher2 },
            new double[] { 0.5, 0.5 },
            EnsembleAggregationMode.GeometricMean); // Not WeightedAverage

        // Assert
        Assert.False(ensemble.SupportsJitCompilation,
            "EnsembleTeacherModel should not support JIT when aggregation mode is not WeightedAverage");
    }

    [Fact]
    public void EnsembleTeacherModel_ExportGraph_Succeeds_WhenSupported()
    {
        // Arrange
        var teacher1 = CreateJitCompatibleTeacher();
        var teacher2 = CreateJitCompatibleTeacher();

        var ensemble = new EnsembleTeacherModel<double>(
            new[] { teacher1, teacher2 },
            new double[] { 0.5, 0.5 },
            EnsembleAggregationMode.WeightedAverage);

        if (!ensemble.SupportsJitCompilation) return;

        // Act
        var inputNodes = new List<ComputationNode<double>>();
        var outputNode = ensemble.ExportComputationGraph(inputNodes);

        // Assert
        Assert.NotNull(outputNode);
    }

    // ========== DistributedTeacherModel Tests ==========

    [Fact]
    public void DistributedTeacherModel_SupportsJit_WhenAllWorkersSupportJit()
    {
        // Arrange
        var worker1 = CreateJitCompatibleTeacher();
        var worker2 = CreateJitCompatibleTeacher();

        var distributed = new DistributedTeacherModel<double>(
            new[] { worker1, worker2 },
            AggregationMode.Average);

        // Assert
        Assert.True(distributed.SupportsJitCompilation,
            "DistributedTeacherModel should support JIT when all workers support JIT and using Average aggregation");
    }

    [Fact]
    public void DistributedTeacherModel_DoesNotSupportJit_WhenAnyWorkerDoesNotSupportJit()
    {
        // Arrange
        var jitWorker = CreateJitCompatibleTeacher();
        var nonJitWorker = CreateNonJitTeacher();

        var distributed = new DistributedTeacherModel<double>(
            new ITeacherModel<Vector<double>, Vector<double>>[] { jitWorker, nonJitWorker },
            AggregationMode.Average);

        // Assert
        Assert.False(distributed.SupportsJitCompilation,
            "DistributedTeacherModel should not support JIT when any worker doesn't support JIT");
    }

    // ========== MultiModalTeacherModel Tests ==========

    [Fact]
    public void MultiModalTeacherModel_SupportsJit_WhenAllModalitiesSupportJit()
    {
        // Arrange
        var modality1 = CreateJitCompatibleTeacher();
        var modality2 = CreateJitCompatibleTeacher();

        var multiModal = new MultiModalTeacherModel<double>(
            new[] { modality1, modality2 },
            new double[] { 0.6, 0.4 });

        // Assert
        Assert.True(multiModal.SupportsJitCompilation,
            "MultiModalTeacherModel should support JIT when all modality teachers support JIT");
    }

    [Fact]
    public void MultiModalTeacherModel_DoesNotSupportJit_WhenAnyModalityDoesNotSupportJit()
    {
        // Arrange
        var jitModality = CreateJitCompatibleTeacher();
        var nonJitModality = CreateNonJitTeacher();

        var multiModal = new MultiModalTeacherModel<double>(
            new ITeacherModel<Vector<double>, Vector<double>>[] { jitModality, nonJitModality },
            new double[] { 0.6, 0.4 });

        // Assert
        Assert.False(multiModal.SupportsJitCompilation,
            "MultiModalTeacherModel should not support JIT when any modality doesn't support JIT");
    }

    // ========== AdaptiveTeacherModel Tests ==========

    [Fact]
    public void AdaptiveTeacherModel_SupportsJit_WhenBaseTeacherSupportsJit()
    {
        // Arrange
        var baseTeacher = CreateJitCompatibleTeacher();
        var adaptive = new AdaptiveTeacherModel<double>(baseTeacher);

        // Assert
        Assert.True(adaptive.SupportsJitCompilation,
            "AdaptiveTeacherModel should support JIT when base teacher supports JIT");
    }

    [Fact]
    public void AdaptiveTeacherModel_DoesNotSupportJit_WhenBaseTeacherDoesNotSupportJit()
    {
        // Arrange
        var baseTeacher = CreateNonJitTeacher();
        var adaptive = new AdaptiveTeacherModel<double>(baseTeacher);

        // Assert
        Assert.False(adaptive.SupportsJitCompilation,
            "AdaptiveTeacherModel should not support JIT when base teacher doesn't support JIT");
    }

    // ========== CurriculumTeacherModel Tests ==========

    [Fact]
    public void CurriculumTeacherModel_SupportsJit_WhenBaseTeacherSupportsJit()
    {
        // Arrange
        var baseTeacher = CreateJitCompatibleTeacher();
        var curriculum = new CurriculumTeacherModel<double>(baseTeacher);

        // Assert
        Assert.True(curriculum.SupportsJitCompilation,
            "CurriculumTeacherModel should support JIT when base teacher supports JIT");
    }

    [Fact]
    public void CurriculumTeacherModel_DoesNotSupportJit_WhenBaseTeacherDoesNotSupportJit()
    {
        // Arrange
        var baseTeacher = CreateNonJitTeacher();
        var curriculum = new CurriculumTeacherModel<double>(baseTeacher);

        // Assert
        Assert.False(curriculum.SupportsJitCompilation,
            "CurriculumTeacherModel should not support JIT when base teacher doesn't support JIT");
    }

    // ========== Non-JIT-Supported Teachers Tests ==========

    [Fact]
    public void SelfTeacherModel_DoesNotSupportJit()
    {
        // Arrange - SelfTeacherModel uses cached predictions (no underlying model)
        var selfTeacher = new SelfTeacherModel<double>(10);

        // Assert
        Assert.False(selfTeacher.SupportsJitCompilation,
            "SelfTeacherModel should not support JIT due to cached predictions");
    }

    [Fact]
    public void QuantizedTeacherModel_DoesNotSupportJit()
    {
        // Arrange - QuantizedTeacherModel requires runtime quantization
        var baseTeacher = CreateJitCompatibleTeacher();
        var quantized = new QuantizedTeacherModel<double>(baseTeacher, 8);

        // Assert
        Assert.False(quantized.SupportsJitCompilation,
            "QuantizedTeacherModel should not support JIT due to runtime quantization");
    }

    [Fact]
    public void TransformerTeacherModel_DoesNotSupportJit()
    {
        // Arrange - TransformerTeacherModel uses Func<> delegate
        Func<Vector<double>, Vector<double>> transformerFunc = input => input;
        var transformer = new TransformerTeacherModel<double>(transformerFunc, 10, 10);

        // Assert
        Assert.False(transformer.SupportsJitCompilation,
            "TransformerTeacherModel should not support JIT due to Func<> delegate");
    }

    [Fact]
    public void PretrainedTeacherModel_DoesNotSupportJit()
    {
        // Arrange - PretrainedTeacherModel uses Func<> delegate
        Func<Vector<double>, Vector<double>> predictionFunc = input => input;
        var pretrained = new PretrainedTeacherModel<double>(predictionFunc, 10, 10);

        // Assert
        Assert.False(pretrained.SupportsJitCompilation,
            "PretrainedTeacherModel should not support JIT due to Func<> delegate");
    }

    [Fact]
    public void OnlineTeacherModel_DoesNotSupportJit()
    {
        // Arrange - OnlineTeacherModel uses Func<> delegate and streaming updates
        Func<Vector<double>, Vector<double>> predictionFunc = input => input;
        var online = new OnlineTeacherModel<double>(predictionFunc, 10, 10);

        // Assert
        Assert.False(online.SupportsJitCompilation,
            "OnlineTeacherModel should not support JIT due to streaming nature");
    }

    // ========== JIT Compatibility Analysis Tests ==========

    [Fact]
    public void JitCompatible_EnsembleTeacher_AnalysisSucceeds()
    {
        // Arrange
        var teacher1 = CreateJitCompatibleTeacher();
        var teacher2 = CreateJitCompatibleTeacher();

        var ensemble = new EnsembleTeacherModel<double>(
            new[] { teacher1, teacher2 },
            new double[] { 0.5, 0.5 },
            EnsembleAggregationMode.WeightedAverage);

        if (!ensemble.SupportsJitCompilation) return;

        // Act
        var inputNodes = new List<ComputationNode<double>>();
        var outputNode = ensemble.ExportComputationGraph(inputNodes);

        var jit = new JitCompilerClass();
        var compatibility = jit.AnalyzeCompatibility(outputNode, inputNodes);

        // Assert
        Assert.NotNull(compatibility);
        Assert.True(compatibility.IsFullySupported || compatibility.CanUseHybridMode,
            "JIT-compatible ensemble should pass compatibility analysis");
    }

    // ========== Helper Methods ==========

    private static MockJitTeacher CreateJitCompatibleTeacher()
    {
        return new MockJitTeacher(true, 10, 10);
    }

    private static MockNonJitTeacher CreateNonJitTeacher()
    {
        return new MockNonJitTeacher(10, 10);
    }

    /// <summary>
    /// Mock teacher that supports JIT compilation.
    /// </summary>
    private class MockJitTeacher : TeacherModelBase<Vector<double>, Vector<double>, double>
    {
        private readonly int _inputDim;
        private readonly int _outputDim;
        private readonly bool _supportsJit;

        public MockJitTeacher(bool supportsJit, int inputDim, int outputDim)
        {
            _supportsJit = supportsJit;
            _inputDim = inputDim;
            _outputDim = outputDim;
        }

        public override int OutputDimension => _outputDim;

        public override Vector<double> GetLogits(Vector<double> input)
        {
            return new Vector<double>(new double[_outputDim]);
        }

        public override bool SupportsJitCompilation => _supportsJit;

        public override ComputationNode<double> ExportComputationGraph(List<ComputationNode<double>> inputNodes)
        {
            // Create a simple passthrough computation graph
            var inputTensor = new Tensor<double>(new[] { _inputDim });
            var inputNode = TensorOperations<double>.Variable(inputTensor, "mock_input");
            inputNodes.Add(inputNode);

            // Simple identity transform
            return inputNode;
        }
    }

    /// <summary>
    /// Mock teacher that does not support JIT compilation.
    /// </summary>
    private class MockNonJitTeacher : ITeacherModel<Vector<double>, Vector<double>>
    {
        private readonly int _outputDim;

        public MockNonJitTeacher(int inputDim, int outputDim)
        {
            _outputDim = outputDim;
        }

        public int OutputDimension => _outputDim;

        public Vector<double> GetLogits(Vector<double> input)
        {
            return new Vector<double>(new double[_outputDim]);
        }
    }

}
