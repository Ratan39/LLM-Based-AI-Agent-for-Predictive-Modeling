# LLM-Based-AI-Agent-for-Predictive-Modeling

## Project Overview

This project was developed as part of Master's level course on **Foundations of Emerging Systems (FES)**. The objective was to create an LLM-based AI agent to assist in designing and supporting the development of predictive models using a small dataset.

The AI agent is built on **Llama-2-13B-chat-GGUF**, configured using specific parameters, and deployed within a Google Colab environment with GPU support. Prompts were designed to direct the agent in performing data analysis and modeling tasks.


## Tasks Performed by the Agent

The AI agent was prompted to:

1. **Calculate basic statistics**  
   - Mean, Max, Min for all numerical variables in the dataset.

2. **Identify high correlations**  
   - Using Pearson's correlation coefficient to identify strong linear relationships.

3. **Feature importance for classification**  
   - Identify the most significant variables for predicting "Propensity to Pay".

4. **Model recommendations**  
   - Recommend top 4 machine learning or deep learning classification models with pro's and cons

## LLM Configuration

The agent uses the `llama-cpp-python` interface to connect to the `Llama-2-13B-chat-GGUF` model with the following configuration:

```python
n_threads = 2
n_batch = 512
n_gpu_layers = 43
n_ctx = 4096
