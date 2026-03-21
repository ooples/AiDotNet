global using AiDotNet.Enums;
global using AiDotNet.Helpers;
global using AiDotNet.Interfaces;
global using AiDotNet.LinearAlgebra;
global using AiDotNet.Models;
global using AiDotNet.Statistics;

// Resolve type ambiguity between AiDotNet and AiDotNet.Tensors.Helpers (0.13.0+)
global using QuantizationMode = AiDotNet.Enums.QuantizationMode;
global using MemoryLayout = AiDotNet.InferenceOptimization.IR.Common.MemoryLayout;
