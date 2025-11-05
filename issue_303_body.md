### Overview

This issue tracks the final integration, testing, and documentation tasks required to make the Retrieval-Augmented Generation (RAG) framework production-ready. The core components have been implemented (see previous issue history), and this plan details the remaining work in a granular, checklist format suitable for a junior developer.

---

### **Task 1: Finalize Builder Integration**

**Goal:** Seamlessly integrate the `RAGConfigurationBuilder` into the main `PredictionModelBuilder` to provide a single, fluent interface for users.

#### **AC 1.1: Create Fluent RAG Configuration Method (3 points)**
**Requirement:** Add a `.WithRAG()` method to the `PredictionModelBuilder`.

-   [ ] Open the file `src/PredictionModelBuilder.cs`.
-   [ ] Add a new private field to the `PredictionModelBuilder` class: `private RAGConfiguration _ragConfiguration;`
-   [ ] Create a new public method with the following signature: `public PredictionModelBuilder WithRAG(Action<RAGConfigurationBuilder> ragConfigAction)`
-   [ ] **Implement the method logic:**
    -   [ ] Instantiate a new `RAGConfigurationBuilder`: `var ragBuilder = new RAGConfigurationBuilder();`
    -   [ ] Pass the builder to the user's action: `ragConfigAction(ragBuilder);`
    -   [ ] Build the configuration and store it: `_ragConfiguration = ragBuilder.Build();`
    -   [ ] Return `this` to allow for fluent chaining.
-   [ ] **Update the `Build()` method:**
    -   [ ] Inside the `Build()` method, add a check: `if (_ragConfiguration != null)`.
    -   [ ] If true, instantiate and return a `RAGPipeline` (this may require creating a `RAGPipeline` class if it doesn't exist), passing the `_ragConfiguration` to its constructor.

---

### **Task 2: Create Comprehensive End-to-End Example**

**Goal:** Add a clear, runnable example to the test console that demonstrates how to use the entire RAG pipeline.

#### **AC 2.1: Create Example File and Class (2 points)**
**Requirement:** Set up the file and class structure for the example.

-   [ ] In the `testconsole` project, create a new folder: `Examples`.
-   [ ] Inside the new folder, create a new file: `RAGPipelineExample.cs`.
-   [ ] Inside the file, define a new `internal static class RAGPipelineExample`.
-   [ ] Create a public static method: `public static void Run()`. All example logic will go inside this method.

#### **AC 2.2: Implement Example Logic (5 points)**
**Requirement:** Write the code demonstrating the RAG pipeline from configuration to result.

-   [ ] **Step 1: Setup Data:** Inside `Run()`, create a `List<string>` of 3-5 sample text documents. (e.g., "The cat sat on the mat.", "The dog chased the ball.").
-   [ ] **Step 2: Configure Pipeline:** Instantiate the `PredictionModelBuilder`.
    -   [ ] Call the new `.WithRAG()` method.
    -   [ ] Inside the lambda, use the `ragBuilder` to configure the pipeline:
        -   [ ] `.WithDocumentStore(new InMemoryDocumentStore<float>(vectorDimension: 4));` (The dimension must match the stub model).
        -   [ ] `.WithChunkingStrategy(new FixedSizeChunkingStrategy(chunkSize: 10, chunkOverlap: 2));`
        -   [ ] `.WithEmbeddingModel(new StubEmbeddingModel(outputDimension: 4));`
        -   [ ] `.WithRetriever(new VectorRetriever<float>());`
-   [ ] **Step 3: Build Pipeline:** Call `var ragPipeline = predictionBuilder.Build();`
-   [ ] **Step 4: Ingest Data:** Call `ragPipeline.Ingest(documents);`
-   [ ] **Step 5: Ask Question:** Define a question string related to the documents, e.g., `string question = "What did the cat do?";`
-   [ ] **Step 6: Get Answer:** Call `var result = ragPipeline.Ask(question);`
-   [ ] **Step 7: Display Results:** Use `Console.WriteLine` to display the augmented answer and the content of the retrieved source documents.

#### **AC 2.3: Integrate with Main Program (1 point)**
**Requirement:** Make the example runnable from the main console entry point.

-   [ ] Open `testconsole/Program.cs`.
-   [ ] Add a call to `RAGPipelineExample.Run();` within the `Main` method.

---

### **Task 3: Enhance Test Coverage and Benchmarking**

**Goal:** Ensure the RAG framework is robust and performant.

#### **AC 3.1: Audit and Improve Test Coverage (5 points)**
**Requirement:** Ensure all RAG components meet the project's 80% coverage standard.

-   [ ] Run your code coverage tool on the `tests` project.
-   [ ] Generate a coverage report.
-   [ ] Identify all classes in the `AiDotNet.RetrievalAugmentedGeneration` namespace with line or branch coverage below 80%.
-   [ ] For each identified class, add new unit tests to its corresponding test file in the `tests` project until the 80% threshold is met or exceeded.

#### **AC 3.2: Implement RAG Performance Benchmarks (8 points)**
**Requirement:** Create benchmarks for key RAG components using `BenchmarkDotNet`.

-   [ ] In the `AiDotNetBenchmarkTests` project, create a new file: `Benchmarks/RAGBenchmarks.cs`.
-   [ ] Create a public class `RAGBenchmarks` with the `[MemoryDiagnoser]` attribute.
-   [ ] **Setup:** In a `[GlobalSetup]` method, create and populate an `InMemoryDocumentStore` with at least 1,000 documents.
-   [ ] **Benchmark 1 (Retrievers):**
    -   [ ] Create a `[Benchmark]` method for each retriever to be tested (`VectorRetriever`, `BM25Retriever`, `HybridRetriever`).
    -   [ ] Inside each method, call the retriever's `Retrieve` method with a sample query and return the results.
-   [ ] **Benchmark 2 (Rerankers):**
    -   [ ] Create a `[Benchmark]` method for each reranker (`CrossEncoderReranker`, `LLMBasedReranker`).
    -   [ ] The method should take a pre-retrieved list of 50 documents and call the reranker's `Rerank` method.

---

### **Definition of Done**

-   [ ] All checklist items are complete.
-   [ ] The RAG pipeline is configurable and runnable via the `PredictionModelBuilder`.
-   [ ] A complete, working example exists in the `testconsole`.
-   [ ] All RAG components have at least 80% test coverage.
-   [ ] Performance benchmarks for retrievers and rerankers are implemented.