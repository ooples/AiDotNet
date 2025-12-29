# Any-Rank Tensor Support Checklist

## Pattern to Implement
1. Add `private int[] _originalInputShape = [];` field
2. In Forward: Store original shape, flatten to required rank, process, reshape back
3. In Backward: Flatten gradient, process, reshape back to original

## Recurrent Layers (Require 3D: [batch, timeSteps, features])

- [x] LSTMLayer - Completed
- [x] GRULayer - Completed
- [x] BidirectionalLayer - Completed (covers BiLSTM and BiGRU)

## Attention Layers (Require 3D: [batch, seq, features])

- [x] SelfAttentionLayer - Completed
- [x] MultiHeadAttentionLayer - Already has any-rank support
- [x] CrossAttentionLayer - Completed

## Normalization Layers

- [x] BatchNormalizationLayer - Completed
- [x] LayerNormalizationLayer - Completed
- [x] GroupNormalizationLayer - Completed
- [x] InstanceNormalizationLayer - Completed (newly implemented with any-rank support)

## Already Completed (Before This Session)

- [x] DenseLayer - Has any-rank support
- [x] EmbeddingLayer - Has any-rank support

## Notes

- The key pattern is: Store shape -> Flatten -> Process -> Reshape back
- For output, last dimension(s) may change (e.g., hiddenSize replaces features)
- Both Forward and Backward methods need updating
