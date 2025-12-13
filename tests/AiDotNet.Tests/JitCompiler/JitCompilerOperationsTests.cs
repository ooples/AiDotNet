using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Enums;
using AiDotNet.JitCompiler;
using AiDotNet.JitCompiler.IR.Operations;
using Xunit;

namespace AiDotNet.Tests.JitCompiler
{
    /// <summary>
    /// Tests for JIT compiler operations, especially the newly added extended activation functions.
    /// </summary>
    /// <remarks>
    /// These tests are quarantined because they trigger GPU initialization which can fail
    /// on machines without proper GPU support or drivers.
    /// </remarks>
    [Trait("Category", "GPU")]
    public class JitCompilerOperationsTests
    {
        [Fact]
        public void GetSupportedOperationTypes_Contains_Basic_Activations()
        {
            var supportedOps = AiDotNet.JitCompiler.JitCompiler.GetSupportedOperationTypes();

            Assert.Contains(OperationType.ReLU, supportedOps);
            Assert.Contains(OperationType.Sigmoid, supportedOps);
            Assert.Contains(OperationType.Tanh, supportedOps);
            Assert.Contains(OperationType.Softmax, supportedOps);
        }

        [Fact]
        public void GetSupportedOperationTypes_Contains_Extended_Activations()
        {
            var supportedOps = AiDotNet.JitCompiler.JitCompiler.GetSupportedOperationTypes();

            // Extended activation functions
            Assert.Contains(OperationType.ELU, supportedOps);
            Assert.Contains(OperationType.LeakyReLU, supportedOps);
            Assert.Contains(OperationType.GELU, supportedOps);
            Assert.Contains(OperationType.Swish, supportedOps);
            Assert.Contains(OperationType.Mish, supportedOps);
            Assert.Contains(OperationType.SoftPlus, supportedOps);
            Assert.Contains(OperationType.SELU, supportedOps);
            Assert.Contains(OperationType.HardSigmoid, supportedOps);
            Assert.Contains(OperationType.HardTanh, supportedOps);
            Assert.Contains(OperationType.SoftSign, supportedOps);
            Assert.Contains(OperationType.CELU, supportedOps);
            Assert.Contains(OperationType.LogSoftmax, supportedOps);
            Assert.Contains(OperationType.PReLU, supportedOps);
            Assert.Contains(OperationType.ThresholdedReLU, supportedOps);
        }

        [Fact]
        public void GetSupportedOperationTypes_Contains_Additional_Extended_Activations()
        {
            var supportedOps = AiDotNet.JitCompiler.JitCompiler.GetSupportedOperationTypes();

            // Additional extended set
            Assert.Contains(OperationType.LiSHT, supportedOps);
            Assert.Contains(OperationType.BentIdentity, supportedOps);
            Assert.Contains(OperationType.Gaussian, supportedOps);
            Assert.Contains(OperationType.ScaledTanh, supportedOps);
            Assert.Contains(OperationType.Squash, supportedOps);
            Assert.Contains(OperationType.ISRU, supportedOps);
            Assert.Contains(OperationType.Sign, supportedOps);
            Assert.Contains(OperationType.Softmin, supportedOps);
            Assert.Contains(OperationType.LogSoftmin, supportedOps);
            Assert.Contains(OperationType.SQRBF, supportedOps);
            Assert.Contains(OperationType.Maxout, supportedOps);
            Assert.Contains(OperationType.RReLU, supportedOps);
            Assert.Contains(OperationType.SphericalSoftmax, supportedOps);
            Assert.Contains(OperationType.TaylorSoftmax, supportedOps);
            Assert.Contains(OperationType.Sparsemax, supportedOps);
            Assert.Contains(OperationType.HierarchicalSoftmax, supportedOps);
        }

        [Fact]
        public void GetSupportedOperationTypes_Contains_Matrix_Operations()
        {
            var supportedOps = AiDotNet.JitCompiler.JitCompiler.GetSupportedOperationTypes();

            Assert.Contains(OperationType.MatMul, supportedOps);
            Assert.Contains(OperationType.Transpose, supportedOps);
        }

        [Fact]
        public void GetSupportedOperationTypes_Contains_Embedding_And_Attention()
        {
            var supportedOps = AiDotNet.JitCompiler.JitCompiler.GetSupportedOperationTypes();

            Assert.Contains(OperationType.Embedding, supportedOps);
            Assert.Contains(OperationType.ScaledDotProductAttention, supportedOps);
            Assert.Contains(OperationType.MultiHeadAttention, supportedOps);
        }

        [Fact]
        public void GetSupportedOperationTypes_Contains_Fused_Operations()
        {
            var supportedOps = AiDotNet.JitCompiler.JitCompiler.GetSupportedOperationTypes();

            Assert.Contains(OperationType.FusedMatMulAdd, supportedOps);
            Assert.Contains(OperationType.FusedLinearReLU, supportedOps);
            Assert.Contains(OperationType.FusedConvBatchNorm, supportedOps);
            Assert.Contains(OperationType.FusedAddReLU, supportedOps);
        }

        [Fact]
        public void GetSupportedOperationTypes_Contains_Recurrent_Operations()
        {
            var supportedOps = AiDotNet.JitCompiler.JitCompiler.GetSupportedOperationTypes();

            Assert.Contains(OperationType.GRUCell, supportedOps);
            Assert.Contains(OperationType.LSTMCell, supportedOps);
        }

        [Fact]
        public void GetSupportedOperationTypes_Contains_Dropout()
        {
            var supportedOps = AiDotNet.JitCompiler.JitCompiler.GetSupportedOperationTypes();

            Assert.Contains(OperationType.Dropout, supportedOps);
        }

        [Fact]
        public void GetSupportedOperationTypes_Contains_Tensor_Operations()
        {
            var supportedOps = AiDotNet.JitCompiler.JitCompiler.GetSupportedOperationTypes();

            Assert.Contains(OperationType.Gather, supportedOps);
            Assert.Contains(OperationType.Broadcast, supportedOps);
        }

        // ============================================================================
        // IR Operation Validation Tests
        // ============================================================================

        [Fact]
        public void ELUOp_Validates_With_Correct_InputCount()
        {
            var op = new ELUOp
            {
                InputIds = new[] { 0 },
                OutputId = 1,
                Alpha = 1.0
            };

            Assert.True(op.Validate());
        }

        [Fact]
        public void ELUOp_Fails_Validation_With_Wrong_InputCount()
        {
            var op = new ELUOp
            {
                InputIds = new[] { 0, 1 }, // Wrong - should be 1 input
                OutputId = 2,
                Alpha = 1.0
            };

            Assert.False(op.Validate());
        }

        [Fact]
        public void LeakyReLUOp_Validates_Correctly()
        {
            var op = new LeakyReLUOp
            {
                InputIds = new[] { 0 },
                OutputId = 1,
                Alpha = 0.01
            };

            Assert.True(op.Validate());
        }

        [Fact]
        public void GELUOp_Validates_Correctly()
        {
            var op = new GELUOp
            {
                InputIds = new[] { 0 },
                OutputId = 1,
                Approximate = true
            };

            Assert.True(op.Validate());
        }

        [Fact]
        public void SoftmaxOp_Validates_Correctly()
        {
            var op = new SoftmaxOp
            {
                InputIds = new[] { 0 },
                OutputId = 1,
                Axis = -1
            };

            Assert.True(op.Validate());
        }

        [Fact]
        public void LogSoftmaxOp_Validates_Correctly()
        {
            var op = new LogSoftmaxOp
            {
                InputIds = new[] { 0 },
                OutputId = 1,
                Axis = -1
            };

            Assert.True(op.Validate());
        }

        [Fact]
        public void HardTanhOp_Validates_Correctly()
        {
            var op = new HardTanhOp
            {
                InputIds = new[] { 0 },
                OutputId = 1,
                MinVal = -1.0,
                MaxVal = 1.0
            };

            Assert.True(op.Validate());
        }

        [Fact]
        public void SoftPlusOp_Validates_Correctly()
        {
            var op = new SoftPlusOp
            {
                InputIds = new[] { 0 },
                OutputId = 1,
                Beta = 1.0,
                Threshold = 20.0
            };

            Assert.True(op.Validate());
        }

        [Fact]
        public void PReLUOp_Validates_With_Two_Inputs()
        {
            var op = new PReLUOp
            {
                InputIds = new[] { 0, 1 }, // Input + alpha parameter
                OutputId = 2
            };

            Assert.True(op.Validate());
        }

        [Fact]
        public void PReLUOp_Fails_With_One_Input()
        {
            var op = new PReLUOp
            {
                InputIds = new[] { 0 }, // Missing alpha parameter
                OutputId = 1
            };

            Assert.False(op.Validate());
        }

        [Fact]
        public void MaxoutOp_Validates_With_Valid_NumPieces()
        {
            var op = new MaxoutOp
            {
                InputIds = new[] { 0 },
                OutputId = 1,
                NumPieces = 2
            };

            Assert.True(op.Validate());
        }

        [Fact]
        public void MaxoutOp_Fails_With_InvalidNumPieces()
        {
            var op = new MaxoutOp
            {
                InputIds = new[] { 0 },
                OutputId = 1,
                NumPieces = 1 // Must be at least 2
            };

            Assert.False(op.Validate());
        }

        [Fact]
        public void RReLUOp_Validates_With_Valid_Bounds()
        {
            var op = new RReLUOp
            {
                InputIds = new[] { 0 },
                OutputId = 1,
                Lower = 0.125,
                Upper = 0.333
            };

            Assert.True(op.Validate());
        }

        [Fact]
        public void RReLUOp_Fails_With_Invalid_Bounds()
        {
            var op = new RReLUOp
            {
                InputIds = new[] { 0 },
                OutputId = 1,
                Lower = 0.5,
                Upper = 0.1 // Upper must be >= Lower
            };

            Assert.False(op.Validate());
        }

        [Fact]
        public void TaylorSoftmaxOp_Validates_With_Valid_Order()
        {
            var op = new TaylorSoftmaxOp
            {
                InputIds = new[] { 0 },
                OutputId = 1,
                Axis = -1,
                Order = 2
            };

            Assert.True(op.Validate());
        }

        [Fact]
        public void TaylorSoftmaxOp_Fails_With_Invalid_Order()
        {
            var op = new TaylorSoftmaxOp
            {
                InputIds = new[] { 0 },
                OutputId = 1,
                Axis = -1,
                Order = 0 // Must be at least 1
            };

            Assert.False(op.Validate());
        }

        // ============================================================================
        // Extended Activation Operations - Additional Tests
        // ============================================================================

        [Fact]
        public void LiSHTOp_Validates_Correctly()
        {
            var op = new LiSHTOp
            {
                InputIds = new[] { 0 },
                OutputId = 1
            };

            Assert.True(op.Validate());
        }

        [Fact]
        public void BentIdentityOp_Validates_Correctly()
        {
            var op = new BentIdentityOp
            {
                InputIds = new[] { 0 },
                OutputId = 1
            };

            Assert.True(op.Validate());
        }

        [Fact]
        public void GaussianOp_Validates_Correctly()
        {
            var op = new GaussianOp
            {
                InputIds = new[] { 0 },
                OutputId = 1
            };

            Assert.True(op.Validate());
        }

        [Fact]
        public void ScaledTanhOp_Validates_Correctly()
        {
            var op = new ScaledTanhOp
            {
                InputIds = new[] { 0 },
                OutputId = 1,
                Beta = 2.0
            };

            Assert.True(op.Validate());
        }

        [Fact]
        public void SquashOp_Validates_Correctly()
        {
            var op = new SquashOp
            {
                InputIds = new[] { 0 },
                OutputId = 1
            };

            Assert.True(op.Validate());
        }

        [Fact]
        public void ISRUOp_Validates_Correctly()
        {
            var op = new ISRUOp
            {
                InputIds = new[] { 0 },
                OutputId = 1,
                Alpha = 1.0
            };

            Assert.True(op.Validate());
        }

        [Fact]
        public void SignOp_Validates_Correctly()
        {
            var op = new SignOp
            {
                InputIds = new[] { 0 },
                OutputId = 1
            };

            Assert.True(op.Validate());
        }

        [Fact]
        public void SoftminOp_Validates_Correctly()
        {
            var op = new SoftminOp
            {
                InputIds = new[] { 0 },
                OutputId = 1,
                Axis = -1
            };

            Assert.True(op.Validate());
        }

        [Fact]
        public void SQRBFOp_Validates_Correctly()
        {
            var op = new SQRBFOp
            {
                InputIds = new[] { 0 },
                OutputId = 1
            };

            Assert.True(op.Validate());
        }

        [Fact]
        public void SphericalSoftmaxOp_Validates_Correctly()
        {
            var op = new SphericalSoftmaxOp
            {
                InputIds = new[] { 0 },
                OutputId = 1,
                Axis = -1
            };

            Assert.True(op.Validate());
        }

        [Fact]
        public void SparsemaxOp_Validates_Correctly()
        {
            var op = new SparsemaxOp
            {
                InputIds = new[] { 0 },
                OutputId = 1,
                Axis = -1
            };

            Assert.True(op.Validate());
        }

        [Fact]
        public void HierarchicalSoftmaxOp_Validates_Correctly()
        {
            var op = new HierarchicalSoftmaxOp
            {
                InputIds = new[] { 0 },
                OutputId = 1,
                TreeStructure = new[] { 0, 1, 2 }
            };

            Assert.True(op.Validate());
        }

        // ============================================================================
        // ToString() Tests for IR Operations
        // ============================================================================

        [Fact]
        public void SoftmaxOp_ToString_ReturnsCorrectFormat()
        {
            var op = new SoftmaxOp
            {
                InputIds = new[] { 0 },
                OutputId = 1,
                Axis = -1,
                OutputShape = new[] { 10 }
            };

            var str = op.ToString();
            Assert.Contains("Softmax", str);
            Assert.Contains("axis=-1", str);
        }

        [Fact]
        public void ELUOp_ToString_ReturnsCorrectFormat()
        {
            var op = new ELUOp
            {
                InputIds = new[] { 0 },
                OutputId = 1,
                Alpha = 0.5,
                OutputShape = new[] { 10 }
            };

            var str = op.ToString();
            Assert.Contains("ELU", str);
            Assert.Contains("alpha=0.5", str);
        }

        [Fact]
        public void LeakyReLUOp_ToString_ReturnsCorrectFormat()
        {
            var op = new LeakyReLUOp
            {
                InputIds = new[] { 0 },
                OutputId = 1,
                Alpha = 0.02,
                OutputShape = new[] { 10 }
            };

            var str = op.ToString();
            Assert.Contains("LeakyReLU", str);
            Assert.Contains("alpha=0.02", str);
        }

        [Fact]
        public void MaxoutOp_ToString_ReturnsCorrectFormat()
        {
            var op = new MaxoutOp
            {
                InputIds = new[] { 0 },
                OutputId = 1,
                NumPieces = 4,
                OutputShape = new[] { 10 }
            };

            var str = op.ToString();
            Assert.Contains("Maxout", str);
            Assert.Contains("pieces=4", str);
        }

        [Fact]
        public void RReLUOp_ToString_ReturnsCorrectFormat()
        {
            var op = new RReLUOp
            {
                InputIds = new[] { 0 },
                OutputId = 1,
                Lower = 0.1,
                Upper = 0.3,
                OutputShape = new[] { 10 }
            };

            var str = op.ToString();
            Assert.Contains("RReLU", str);
            Assert.Contains("lower=0.1", str);
            Assert.Contains("upper=0.3", str);
        }
    }
}
