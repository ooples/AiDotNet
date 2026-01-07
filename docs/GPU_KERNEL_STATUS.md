# GPU Kernel Implementation Status

This document tracks all GPU kernels needed for full GPU-resident training across CUDA, HIP (AMD), and OpenCL backends.

## Kernel Categories

### Legend
| Symbol | Meaning |
|--------|---------|
| ✅ | Implemented in all backends (CUDA, HIP, OpenCL) |
| ⚠️ | Implemented in some backends (see notes) |
| ❌ | Not implemented in any backend |
| 🔧 | Exists but needs fixes/improvements |

---

## 1. Activation Forward Kernels

| Kernel | CUDA | HIP | OpenCL | Notes |
|--------|------|-----|--------|-------|
| relu | ✅ | ✅ | ✅ | |
| leaky_relu | ✅ | ✅ | ✅ | |
| sigmoid | ✅ | ✅ | ✅ | |
| tanh | ✅ | ✅ | ✅ | |
| gelu | ✅ | ✅ | ✅ | |
| swish/silu | ✅ | ✅ | ✅ | |
| softmax | ✅ | ✅ | ✅ | |
| elu | ✅ | ✅ | ✅ | |
| mish | ✅ | ✅ | ✅ | |
| softplus | ✅ | ✅ | ✅ | |
| hardswish | ✅ | ✅ | ✅ | |

## 2. Activation Backward Kernels

| Kernel | CUDA | HIP | OpenCL | Unblocks |
|--------|------|-----|--------|----------|
| relu_backward | ✅ | ✅ | ✅ | ActivationLayer, all ReLU layers |
| leaky_relu_backward | ✅ | ✅ | ✅ | LeakyReLU activations |
| sigmoid_backward | ✅ | ✅ | ✅ | Sigmoid activations, gates |
| tanh_backward | ✅ | ✅ | ✅ | Tanh activations, LSTM/GRU |
| gelu_backward | ✅ | ✅ | ✅ | Transformers, BERT |
| softmax_backward | ✅ | ✅ | ✅ | Attention, classification |
| elu_backward | ✅ | ✅ | ✅ | ELU activations |
| swish_backward | ✅ | ✅ | ✅ | Swish/SiLU activations |
| mish_backward | ❌ | ❌ | ❌ | Mish activation |
| softplus_backward | ❌ | ❌ | ❌ | Softplus activation |
| hardswish_backward | ❌ | ❌ | ❌ | HardSwish activation |

## 3. Convolution Kernels

| Kernel | CUDA | HIP | OpenCL | Unblocks |
|--------|------|-----|--------|----------|
| **Forward** |
| im2col | ✅ | ✅ | ✅ | All conv layers |
| conv2d_direct | ✅ | ✅ | ✅ | ConvolutionalLayer |
| depthwise_conv2d | ✅ | ✅ | ✅ | DepthwiseSeparable |
| conv_transpose2d | ✅ | ✅ | ✅ | DeconvolutionalLayer |
| conv3d_direct | ✅ | ✅ | ✅ | Conv3DLayer |
| **Backward** |
| col2im | ✅ | ✅ | ✅ | All conv backward |
| conv2d_backward_input | ✅ | ✅ | ✅ | ConvolutionalLayer backward |
| conv2d_backward_weights | ✅ | ✅ | ✅ | ConvolutionalLayer backward |
| conv3d_backward_input | ❌ | ❌ | ❌ | Conv3DLayer backward |
| conv3d_backward_weights | ❌ | ❌ | ❌ | Conv3DLayer backward |
| deconv_backward_input | ❌ | ❌ | ❌ | DeconvolutionalLayer backward |
| deconv_backward_weights | ❌ | ❌ | ❌ | DeconvolutionalLayer backward |
| depthwise_conv2d_backward | ❌ | ❌ | ❌ | DepthwiseSeparable backward |
| dilated_conv2d_backward | ❌ | ❌ | ❌ | DilatedConvolutionalLayer |

## 4. Normalization Kernels

| Kernel | CUDA | HIP | OpenCL | Unblocks |
|--------|------|-----|--------|----------|
| **Forward** |
| batchnorm_forward | ✅ | ✅ | ✅ | BatchNormalizationLayer |
| layernorm_forward | ✅ | ✅ | ✅ | LayerNormalizationLayer |
| groupnorm_forward | ✅ | ✅ | ✅ | GroupNormalizationLayer |
| instancenorm_forward | ✅ | ✅ | ✅ | InstanceNormalizationLayer |
| rmsnorm_forward | ✅ | ✅ | ✅ | RMSNorm (LLaMA style) |
| **Backward** |
| batchnorm_backward | ✅ | ✅ | ✅ | BatchNormalizationLayer |
| layernorm_backward | ✅ | ✅ | ✅ | LayerNormalizationLayer |
| layernorm_grad_params | ✅ | ✅ | ✅ | LayerNorm gamma/beta grads |
| groupnorm_backward | ❌ | ❌ | ❌ | GroupNormalizationLayer |
| instancenorm_backward | ❌ | ❌ | ❌ | InstanceNormalizationLayer |
| rmsnorm_backward | ❌ | ❌ | ✅ | RMSNorm backward |
| rmsnorm_grad_gamma | ❌ | ❌ | ✅ | RMSNorm gamma gradient |

## 5. Pooling Kernels

| Kernel | CUDA | HIP | OpenCL | Unblocks |
|--------|------|-----|--------|----------|
| **Forward** |
| maxpool2d | ✅ | ✅ | ✅ | MaxPoolingLayer |
| avgpool2d | ✅ | ✅ | ✅ | AveragePoolingLayer |
| global_avgpool2d | ✅ | ✅ | ✅ | GlobalPoolingLayer |
| global_maxpool2d | ✅ | ✅ | ✅ | GlobalPoolingLayer |
| adaptive_avgpool2d | ✅ | ✅ | ✅ | AdaptiveAveragePoolingLayer |
| **Backward** |
| maxpool2d_backward | ✅ | ✅ | ✅ | MaxPoolingLayer ✓ |
| avgpool2d_backward | ✅ | ✅ | ✅ | AveragePoolingLayer ✓ |
| global_avgpool2d_backward | ❌ | ❌ | ❌ | GlobalPoolingLayer |
| global_maxpool2d_backward | ❌ | ❌ | ❌ | GlobalPoolingLayer |
| adaptive_avgpool2d_backward | ❌ | ❌ | ❌ | AdaptiveAveragePoolingLayer |
| maxpool3d_backward | ❌ | ❌ | ❌ | MaxPool3DLayer |
| avgpool3d_backward | ❌ | ❌ | ❌ | AveragePool3DLayer |

## 6. Attention Kernels

| Kernel | CUDA | HIP | OpenCL | Unblocks |
|--------|------|-----|--------|----------|
| **Forward** |
| scaled_dot_product_attention | ✅ | ✅ | ✅ | AttentionLayer, MHA |
| flash_attention_v2 | ✅ | ✅ | ✅ | Memory-efficient attention |
| grouped_query_attention | ✅ | ✅ | ✅ | GQA (LLaMA 2 style) |
| **Backward** |
| flash_attention_backward | ✅ | ✅ | ✅ | All attention layers |
| grouped_query_attention_backward | ✅ | ✅ | ✅ | GQA backward |
| cross_attention_backward | ❌ | ❌ | ❌ | CrossAttentionLayer |
| multi_head_attention_backward | ❌ | ❌ | ❌ | QKV projection grads |

## 7. Loss Function Kernels

| Kernel | CUDA | HIP | OpenCL | Unblocks |
|--------|------|-----|--------|----------|
| **Forward** |
| mse_loss | ✅ | ✅ | ✅ | MeanSquaredErrorLoss |
| cross_entropy_loss | ✅ | ✅ | ✅ | CrossEntropyLoss |
| bce_loss | ✅ | ✅ | ✅ | BinaryCrossEntropyLoss |
| smooth_l1_loss | ✅ | ✅ | ✅ | HuberLoss |
| **Backward** |
| mse_backward | ✅ | ✅ | ✅ | MeanSquaredErrorLoss |
| cross_entropy_backward | ✅ | ✅ | ✅ | CrossEntropyLoss |
| bce_backward | ✅ | ✅ | ✅ | BinaryCrossEntropyLoss |
| smooth_l1_backward | ✅ | ✅ | ✅ | HuberLoss |
| focal_loss | ❌ | ❌ | ❌ | FocalLoss |
| focal_loss_backward | ❌ | ❌ | ❌ | FocalLoss |
| triplet_loss | ❌ | ❌ | ❌ | TripletLoss |
| triplet_loss_backward | ❌ | ❌ | ❌ | TripletLoss |
| contrastive_loss | ❌ | ❌ | ❌ | ContrastiveLoss |
| contrastive_loss_backward | ❌ | ❌ | ❌ | ContrastiveLoss |

## 8. Optimizer Kernels

| Kernel | CUDA | HIP | OpenCL | Unblocks |
|--------|------|-----|--------|----------|
| sgd_step | ✅ | ✅ | ✅ | SGDOptimizer |
| sgd_momentum_update | ❌ | ❌ | ✅ | MomentumOptimizer |
| adam_step | ✅ | ✅ | ✅ | AdamOptimizer |
| adamw_step | ✅ | ✅ | ✅ | AdamWOptimizer |
| rmsprop_step | ❌ | ❌ | ❌ | RMSpropOptimizer |
| adagrad_step | ❌ | ❌ | ❌ | AdagradOptimizer |
| nag_step | ❌ | ❌ | ❌ | NesterovOptimizer |
| lars_step | ❌ | ❌ | ❌ | LARSOptimizer |
| lamb_step | ❌ | ❌ | ❌ | LAMBOptimizer |
| gradient_clip_norm | ❌ | ❌ | ❌ | All optimizers |
| gradient_clip_value | ❌ | ❌ | ❌ | All optimizers |

## 9. Embedding Kernels

| Kernel | CUDA | HIP | OpenCL | Unblocks |
|--------|------|-----|--------|----------|
| embedding_forward | ✅ | ✅ | ✅ | EmbeddingLayer |
| embedding_backward | ✅ | ✅ | ✅ | EmbeddingLayer (sparse scatter) |
| gather_kernel | ❌ | ❌ | ✅ | General gather ops |
| scatter_add_kernel | ❌ | ❌ | ✅ | Sparse gradient accumulation |

## 10. Recurrent Kernels (LSTM/GRU)

| Kernel | CUDA | HIP | OpenCL | Unblocks |
|--------|------|-----|--------|----------|
| **LSTM** |
| lstm_forward | ❌ | ❌ | ❌ | LSTMLayer |
| lstm_backward | ❌ | ❌ | ❌ | LSTMLayer (BPTT) |
| lstm_cell_forward | ❌ | ❌ | ❌ | Single LSTM step |
| lstm_cell_backward | ❌ | ❌ | ❌ | Single LSTM backward |
| **GRU** |
| gru_forward | ❌ | ❌ | ❌ | GRULayer |
| gru_backward | ❌ | ❌ | ❌ | GRULayer (BPTT) |
| gru_cell_forward | ❌ | ❌ | ❌ | Single GRU step |
| gru_cell_backward | ❌ | ❌ | ❌ | Single GRU backward |
| **ConvLSTM** |
| convlstm_forward | ❌ | ❌ | ❌ | ConvLSTMLayer |
| convlstm_backward | ❌ | ❌ | ❌ | ConvLSTMLayer (Issue #700) |

## 11. Utility Kernels

| Kernel | CUDA | HIP | OpenCL | Notes |
|--------|------|-----|--------|-------|
| transpose_2d | ✅ | ✅ | ✅ | |
| batched_transpose | ✅ | ✅ | ✅ | |
| permute_general | ✅ | ✅ | ❌ | General axis permutation |
| copy_buffer | ✅ | ✅ | ✅ | |
| fill_buffer | ❌ | ❌ | ✅ | Zero initialization |
| dropout_forward | ✅ | ✅ | ✅ | |
| dropout_backward | ✅ | ✅ | ✅ | |
| clamp | ✅ | ✅ | ✅ | |
| where_cond | ✅ | ✅ | ✅ | |
| argmax_axis | ✅ | ✅ | ✅ | |
| argmin_axis | ✅ | ✅ | ✅ | |
| reduce_sum | ✅ | ✅ | ❌ | Needs OpenCL impl |
| reduce_max | ✅ | ✅ | ❌ | Needs OpenCL impl |

## 12. Fused Kernels

| Kernel | CUDA | HIP | OpenCL | Notes |
|--------|------|-----|--------|-------|
| gemm_bias_relu | ✅ | ✅ | ✅ | |
| gemm_bias_gelu | ✅ | ✅ | ✅ | |
| gemm_bias_sigmoid | ✅ | ✅ | ✅ | |
| gemm_bias_tanh | ✅ | ✅ | ✅ | |
| gemm_bias | ✅ | ✅ | ✅ | |
| gemm_bias_swish | ✅ | ✅ | ❌ | |
| layernorm_relu | ✅ | ✅ | ✅ | |
| layernorm_gelu | ✅ | ✅ | ✅ | |
| residual_layernorm | ✅ | ✅ | ✅ | |
| bias_dropout | ✅ | ✅ | ✅ | |

## 13. Graph Neural Network Kernels

| Kernel | CUDA | HIP | OpenCL | Unblocks |
|--------|------|-----|--------|----------|
| sparse_mm_forward | ❌ | ❌ | ⚠️ | GCN, GAT, GraphSAGE |
| sparse_mm_backward | ❌ | ❌ | ❌ | All GNN backward |
| message_passing_forward | ❌ | ❌ | ❌ | MessagePassingLayer |
| message_passing_backward | ❌ | ❌ | ❌ | MessagePassingLayer |
| scatter_add | ❌ | ❌ | ✅ | Graph aggregation |
| scatter_max | ❌ | ❌ | ❌ | Graph aggregation |
| scatter_mean | ❌ | ❌ | ❌ | Graph aggregation |
| edge_softmax | ❌ | ❌ | ❌ | GAT attention |
| diffusion_conv_forward | ❌ | ❌ | ❌ | DiffusionConvLayer |
| diffusion_conv_backward | ❌ | ❌ | ❌ | DiffusionConvLayer (Issue #700) |

## 14. 3D/Mesh Kernels

| Kernel | CUDA | HIP | OpenCL | Unblocks |
|--------|------|-----|--------|----------|
| upsample3d_nearest | ❌ | ❌ | ❌ | Upsample3DLayer |
| upsample3d_nearest_backward | ❌ | ❌ | ❌ | Upsample3DLayer |
| mesh_conv_forward | ❌ | ❌ | ❌ | MeshEdgeConvLayer |
| mesh_conv_backward | ❌ | ❌ | ❌ | MeshEdgeConvLayer |
| spiral_conv_forward | ❌ | ❌ | ❌ | SpiralConvLayer |
| spiral_conv_backward | ❌ | ❌ | ❌ | SpiralConvLayer |

---

## Priority Kernel Implementation Order

### Tier 0: Blockers for Basic Training (CRITICAL)
These must be implemented first to enable any GPU training:

1. **GEMM backward (for DenseLayer)** - Already available via transpose + GEMM
2. **Activation backward** - ✅ Already implemented (relu, sigmoid, tanh, gelu, softmax)
3. **Loss backward** - ✅ Already implemented (mse, cross_entropy, bce)
4. **SGD/Adam update** - ✅ Already implemented

**Status: UNBLOCKED** - Basic training infrastructure kernels exist!

### Tier 1: CNN Training
1. conv2d_backward_input - ✅ Exists
2. conv2d_backward_weights - ✅ Exists
3. batchnorm_backward - ✅ Exists
4. pooling backward - ✅ Exists (max, avg)

**Status: UNBLOCKED** - CNN training kernels exist!

### Tier 2: Transformer Training
1. layernorm_backward - ✅ Exists
2. attention backward - ✅ flash_attention_backward exists
3. embedding_backward - ✅ Exists

**Status: UNBLOCKED** - Transformer training kernels exist!

### Tier 3: Recurrent Network Training (BLOCKERS)
1. ❌ lstm_cell_forward
2. ❌ lstm_cell_backward  
3. ❌ gru_cell_forward
4. ❌ gru_cell_backward

**Status: BLOCKED** - Need LSTM/GRU kernels

### Tier 4: Graph Neural Network Training (BLOCKERS)
1. ❌ sparse_mm_backward
2. ❌ scatter_add (CUDA/HIP)
3. ❌ message_passing_backward

**Status: BLOCKED** - Need sparse/scatter kernels

### Tier 5: Missing Backward Kernels (Low Priority)
1. ❌ conv3d_backward
2. ❌ groupnorm_backward
3. ❌ instancenorm_backward
4. ❌ global_pool_backward
5. ❌ mish_backward, softplus_backward, hardswish_backward

---

## Backend Parity Gaps

### OpenCL Missing (compared to CUDA/HIP)
- reduce_sum, reduce_max
- permute_general
- gemm_bias_swish
- sgd_momentum_update (uses different name)

### CUDA/HIP Missing (compared to OpenCL)
- rmsnorm_backward, rmsnorm_grad_gamma
- scatter_add_kernel, gather_kernel
- fill_buffer

---

## Summary Statistics

| Category | Total Kernels | Implemented | Missing | % Complete |
|----------|--------------|-------------|---------|------------|
| Activation Forward | 11 | 11 | 0 | 100% |
| Activation Backward | 11 | 8 | 3 | 73% |
| Convolution | 14 | 8 | 6 | 57% |
| Normalization | 12 | 9 | 3 | 75% |
| Pooling | 12 | 6 | 6 | 50% |
| Attention | 7 | 6 | 1 | 86% |
| Loss Functions | 10 | 8 | 2 | 80% |
| Optimizer | 11 | 4 | 7 | 36% |
| Embedding | 4 | 3 | 1 | 75% |
| Recurrent (LSTM/GRU) | 10 | 0 | 10 | 0% |
| Graph Neural Networks | 10 | 1 | 9 | 10% |
| 3D/Mesh | 6 | 0 | 6 | 0% |
| **TOTAL** | **118** | **64** | **54** | **54%** |

## Key Findings

### Good News
1. **Basic training is UNBLOCKED**: Dense, Conv2D, BatchNorm, Attention, Loss functions all have backward kernels
2. **Optimizer kernels exist**: SGD and Adam are implemented
3. **Good backend parity**: CUDA, HIP, OpenCL have similar coverage

### Blockers to Address
1. **LSTM/GRU kernels**: 0% complete - blocks all recurrent layer training
2. **GNN kernels**: 10% complete - blocks graph neural network training  
3. **Conv3D backward**: Missing - blocks 3D CNN training
4. **Some optimizers**: RMSprop, Adagrad, LARS, LAMB missing

### Recommended Implementation Order
1. LSTM cell forward/backward (unblocks LSTMLayer, ConvLSTMLayer)
2. GRU cell forward/backward (unblocks GRULayer)
3. scatter_add for CUDA/HIP (unblocks GNN layers)
4. sparse_mm_backward (unblocks GNN training)
5. Conv3D backward (unblocks 3D CNNs)
6. Remaining optimizer kernels
