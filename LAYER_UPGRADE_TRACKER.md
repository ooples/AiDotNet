# Neural Network Layer Production-Grade Upgrade Tracker

## Production-Grade Pattern Requirements
- Locality caches (`_lastInput`, `_lastOutput`) from forward pass
- `Tensor.Transform((x, _) => activation.Derivative(x))` for vectorized activation derivative
- `Engine.TensorMultiply` for GPU/CPU accelerated element-wise ops
- `Tensor<T>.FromMatrix()` / `Tensor<T>.FromVector()` for conversions
- `.ToMatrix()` / `.ToVector()` for reverse conversions
- **Inline topological sort** (NOT helper method call)
- Remove unused helper methods (GetTopologicalOrder, MatrixToTensor, etc.)

---

## Completed Layers

### 1. FullyConnectedLayer.cs - DONE (commit 5f849af0)
- Replaced MatrixToTensor with Tensor<T>.FromMatrix
- Replaced VectorToTensor with Tensor<T>.FromVector
- Inlined topological sort
- Replaced TensorToMatrix/TensorToVector with .ToMatrix()/.ToVector()
- Removed ~180 lines of unused helpers

### 2. DenseLayer.cs - DONE (commit 0c72e8cf)
- Same pattern applied
- Removed ~122 lines of helpers

---

## In Progress

### 3. AttentionLayer.cs - IN PROGRESS
**Status**: Has GetTopologicalOrder calls on lines 585, 596
**Good**: Already uses Engine.TensorMultiply, Tensor.Transform, locality caches
**Needs**:
- [ ] Inline topological sort at line 585 (attentionScores)
- [ ] Inline topological sort at line 596 (VNode)
- [ ] Remove GetTopologicalOrder helper (lines 626-663)
- [ ] Remove ApplyActivationAutodiff helper (lines 668-692) - unused

---

## Pending Layers (26 remaining with GetTopologicalOrder)

### 4. ReshapeLayer.cs - PENDING
**Notes**: Simple layer, likely minimal changes

### 5. LayerNormalizationLayer.cs - PENDING
**Notes**: Important for transformers

### 6. DropoutLayer.cs - PENDING
**Notes**: Should be simple

### 7. BatchNormalizationLayer.cs - PENDING
**Notes**: Complex gradients, careful review needed

### 8. AvgPoolingLayer.cs - PENDING
**Notes**: Pooling operation

### 9. TransformerDecoderLayer.cs - PENDING
**Notes**: Complex layer with multiple attention mechanisms

### 10. TransformerEncoderLayer.cs - PENDING
**Notes**: Complex layer

### 11. HighwayLayer.cs - PENDING
**Notes**: Gated layer

### 12. LogVarianceLayer.cs - PENDING
**Notes**: VAE component

### 13. MeanLayer.cs - PENDING
**Notes**: Simple aggregation

### 14. FlattenLayer.cs - PENDING
**Notes**: Should be trivial

### 15. PatchEmbeddingLayer.cs - PENDING
**Notes**: Vision transformer component

### 16. UpsamplingLayer.cs - PENDING
**Notes**: For decoders

### 17. SqueezeAndExcitationLayer.cs - PENDING
**Notes**: Channel attention

### 18. SplitLayer.cs - PENDING
**Notes**: Tensor splitting

### 19. SpatialTransformerLayer.cs - PENDING
**Notes**: Complex spatial attention

### 20. SelfAttentionLayer.cs - PENDING
**Notes**: May be similar to AttentionLayer

### 21. RBFLayer.cs - PENDING
**Notes**: Radial basis function

### 22. PositionalEncodingLayer.cs - PENDING
**Notes**: Transformer component

### 23. PoolingLayer.cs - PENDING
**Notes**: Base pooling

### 24. MultiHeadAttentionLayer.cs - PENDING
**Notes**: Complex, multiple heads

### 25. MemoryWriteLayer.cs - PENDING
**Notes**: Memory network component

### 26. MaskingLayer.cs - PENDING
**Notes**: Masking operations

### 27. LSTMLayer.cs - PENDING
**Notes**: Complex RNN, careful review

### 28. GlobalPoolingLayer.cs - PENDING
**Notes**: Global aggregation

### 29. GRULayer.cs - PENDING
**Notes**: RNN variant

### 30. DeconvolutionalLayer.cs - PENDING
**Notes**: Transposed convolution

---

## Stats
- Total layers with BackwardViaAutodiff: 76
- Layers still using GetTopologicalOrder helper: 28
- Completed: 2
- In Progress: 1
- Remaining: 27
