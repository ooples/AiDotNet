### Overview

This issue is the central, living roadmap for AiDotNet's development in 2025. It organizes the high-impact backlog into thematic areas, tracks their status, and links to the relevant, detailed issues. The original priority list has been superseded by this more structured format.

**Quick Links:**
-   **Project Board:** https://github.com/users/ooples/projects/7
-   **Milestones:** https://github.com/ooples/AiDotNet/milestones

---

### Theme 1: Retrieval-Augmented Generation (RAG) & Search

**Status:** `Foundational Work Complete`

This theme focuses on building a state-of-the-art, in-house RAG framework and the persistent storage backends required to support it at scale.

#### Completed Planning & Analysis

-   **[#303: RAG Framework Implementation]**: The core components of the RAG framework have been implemented. This issue has been updated to serve as a **verified status report** and tracks the remaining integration and testing work.
-   **[#306: In-House Graph Database]**: A detailed, phased implementation plan has been created to build a persistent, scalable graph database from the existing in-memory version.
-   **[#305: In-House Document Store]**: A detailed, phased implementation plan has been created to build a self-contained, file-based document store as a dependency-free alternative to external databases.

#### Next Steps

-   Execute the phased implementation plans for the **Graph Database (#306)** and **Document Store (#305)**.
-   Complete the final integration, testing, and documentation tasks outlined in the **RAG Framework tracker (#303)**.

---

### Theme 2: Advanced Generative AI (Diffusion Models)

**Status:** `Core Architecture Planned`

This theme focuses on building a comprehensive suite of tools for generative AI, centered around diffusion models for image generation.

#### Completed Planning & Analysis

-   **[#298: Diffusion Models Part 2]**: A comprehensive, phased implementation plan has been created. It details the construction of critical components like the **U-Net**, **Variational Autoencoder (VAE)**, and advanced **Schedulers**, all building upon the foundational interfaces defined in Part 1 (#261, #263).

#### Next Steps

-   Execute the detailed, multi-phase plan for the **Diffusion Models suite (#298)**.

---

### Theme 3: Meta-Learning

**Status:** `Ready for Development`

This theme focuses on implementing several state-of-the-art meta-learning algorithms to enable models that can learn new tasks rapidly from a small number of examples.

#### Next Steps

-   **[#289: Implement SEAL]**: The core implementation of the SEAL (Self-supervised Epoch-wise Active Learning) algorithm.
-   **[#291: Implement MAML]**: Implement the Model-Agnostic Meta-Learning baseline.
-   **[#292: Implement Reptile]**: Implement the Reptile meta-learning baseline.
-   **[#290: Episodic Data Abstractions]**: Create the necessary data structures (N-way, K-shot) to support meta-learning training loops.

---

### Theme 4: Core Infrastructure & Productionization

**Status:** `Ready for Development`

This theme covers critical cross-cutting concerns required to make the library robust, efficient, and easy to use in production environments.

#### Next Steps

-   **[#282: Datasets and DataLoaders]**: Create a unified API for loading and preprocessing various data modalities (image, audio, video).
-   **[#283: Training Recipes & Config System]**: Implement a configuration system (using YAML or JSON) to define and reproduce training runs.
-   **[#277: Inference Optimizations]**: Implement key performance optimizations like KV Caching, RoPE, and FlashAttention.
-   **[#278: Quantization]**: Add support for post-training (PTQ) and quantization-aware (QAT) training to reduce model size and improve inference speed.
-   **[#280: ONNX Export & Runtime]**: Build tools to export models to the ONNX format and execute them with the ONNX Runtime on both CPU and GPU (DirectML).
