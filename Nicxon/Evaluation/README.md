# Library Chatbot RAG System Evaluation

This folder contains a comprehensive evaluation system that compares your RAG system against other language models using cosine similarity analysis and 3D embedding visualizations.

## 🎯 Evaluation Overview

**Models Compared:**

- **RAG System**: Your ChromaDB + Gemini 2.0 Flash system
- **Raw Gemini 2.0**: Base Gemini Flash without RAG context
- **DeepSeek R1 Zero**: Free model via OpenRouter API

**Evaluation Methods:**

- **Cosine Similarity**: Between question embeddings and answer embeddings
- **Inter-Model Similarity**: How similar are answers between different models
- **3D Visualization**: PCA and t-SNE dimensionality reduction for visual analysis

## 📁 File Structure

```
Evaluation/
├── test_questions.json              # 10 diverse library questions
├── collect_answers.py               # Collect answers from all 3 models
├── generate_embeddings.py           # Generate embeddings & similarities
├── visualize_embeddings.py          # Create 3D visualizations
├── run_evaluation.py               # Master script (runs everything)
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── model_answers.json              # Generated: Raw text answers
├── model_embeddings.json           # Generated: 768D embeddings
├── cosine_similarities.json        # Generated: Similarity scores
├── evaluation_report.json          # Generated: Analysis report
│
└── visualizations/                 # Generated: 3D plots & heatmaps
    ├── embedding_visualization_pca.png
    ├── embedding_visualization_pca_interactive.html
    ├── embedding_visualization_tsne.png
    ├── embedding_visualization_tsne_interactive.html
    ├── similarity_heatmap.png
    └── README.md
```

## 🚀 Quick Start

### Prerequisites

1. **API Keys** (choose one method):

   **Option A: .env file (Recommended)**

   ```bash
   # Create .env file in Evaluation folder
   GOOGLE_API_KEY=your_google_api_key_here
   OPENROUTER_API_KEY=your_openrouter_api_key_here
   ```

   **Option B: Environment variables**

   ```bash
   export GOOGLE_API_KEY="your_google_api_key"
   export OPENROUTER_API_KEY="your_openrouter_api_key"
   ```

2. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Vector Database**: Ensure your RAG system is set up at `../vector_db/chroma_db/`

### Run Complete Evaluation

```bash
python run_evaluation.py
```

This master script runs all three phases automatically:

1. **Answer Collection** (~5-10 minutes)
2. **Embedding Analysis** (~3-5 minutes)
3. **Visualization Generation** (~2-3 minutes)

### Run Individual Phases

```bash
# Phase 1: Collect answers
python collect_answers.py

# Phase 2: Generate embeddings and similarities
python generate_embeddings.py

# Phase 3: Create visualizations
python visualize_embeddings.py
```

## 📊 Output Data Structure

### model_answers.json

```json
[
  {
    "id": 1,
    "question": "What books by Thomas Aquinas are available?",
    "category": "author_search",
    "difficulty": "easy",
    "RAG_gemini_flash_2.0": "Based on the library collection, we have...",
    "gemini_flash_2.0": "Thomas Aquinas wrote several important works...",
    "deepseek_r1_zero": "Aquinas was a medieval philosopher who..."
  }
]
```

### model_embeddings.json

```json
[
  {
    "id": 1,
    "question": [0.1234, -0.5678, ...],        # 768D embedding
    "RAG_gemini_flash_2.0": [0.2345, ...],    # 768D embedding
    "gemini_flash_2.0": [0.3456, ...],        # 768D embedding
    "deepseek_r1_zero": [0.4567, ...],        # 768D embedding
    "metadata": {...}
  }
]
```

### cosine_similarities.json

```json
[
  {
    "id": 1,
    "category": "author_search",
    "question_vs_rag": 0.847,        # How relevant is RAG answer to question
    "question_vs_gemini": 0.723,     # How relevant is Gemini answer to question
    "question_vs_deepseek": 0.681,   # How relevant is DeepSeek answer to question
    "rag_vs_gemini": 0.654,          # How similar are RAG and Gemini answers
    "rag_vs_deepseek": 0.598,        # How similar are RAG and DeepSeek answers
    "gemini_vs_deepseek": 0.742      # How similar are Gemini and DeepSeek answers
  }
]
```

## 📈 Visualization Outputs

### 1. **3D PCA Visualization**

- Shows linear relationships in embedding space
- Good for understanding major variance directions
- Files: `embedding_visualization_pca.png` + interactive HTML

### 2. **3D t-SNE Visualization**

- Reveals non-linear patterns and clusters
- Better for seeing semantic groupings
- Files: `embedding_visualization_tsne.png` + interactive HTML

### 3. **Similarity Heatmap**

- Quantitative overview of average similarities
- Quick comparison of model performance
- File: `similarity_heatmap.png`

## 🎨 Using Visualizations in Your Report

### Static Images (PNG)

Perfect for academic reports, presentations, documentation:

- High resolution (300 DPI)
- Multiple viewing angles (3D + 2D projections)
- Professional color scheme

### Interactive Plots (HTML)

Great for demonstrations, exploring data:

- Rotate, zoom, pan in 3D space
- Hover for detailed information
- Can be embedded in web presentations

### Color Coding

- 🔴 **Red**: Original Questions
- 🟦 **Teal**: RAG System Answers
- 🔵 **Blue**: Raw Gemini Answers
- 🟢 **Green**: DeepSeek R1 Zero Answers

## 📋 Interpretation Guide

### What Good Results Look Like

1. **High Question Relevance**: RAG answers should have higher cosine similarity to questions than raw models
2. **Clustering Patterns**: RAG answers should cluster closer to questions in 3D space
3. **Consistency**: Similar question types should produce consistent answer patterns
4. **Specificity**: RAG answers should be more specific due to library context

### Key Metrics to Report

- **Average Question Relevance Scores**: Compare across models
- **Category Performance**: How each model performs on different question types
- **Consistency Analysis**: Standard deviation of similarities
- **Clustering Quality**: Visual separation in 3D plots

## 🔧 Customization

### Adding More Questions

Edit `test_questions.json` to include additional questions:

```json
{
  "id": 11,
  "question": "Your new question here",
  "category": "your_category",
  "difficulty": "easy|medium|hard",
  "expected_elements": ["keyword1", "keyword2"]
}
```

### Adding More Models

Modify `collect_answers.py` to include additional models:

1. Add new API integration
2. Update data structure
3. Modify visualization scripts

### Changing Visualization Style

Edit `visualize_embeddings.py`:

- Colors: Modify `self.colors` dictionary
- Plot types: Add new visualization methods
- Dimensionality: Change PCA/t-SNE parameters

## ⚠️ Troubleshooting

### Common Issues

1. **API Key Errors**: Ensure environment variables are set correctly
2. **Vector DB Not Found**: Run vector database creation script first
3. **Memory Issues**: Reduce batch sizes in embedding generation
4. **Slow Performance**: Use fewer questions or reduce visualization complexity

### Rate Limiting

- Google API: 1 second delay between calls
- OpenRouter API: 1 second delay between calls
- Total time for 10 questions: ~10-15 minutes

## 📊 Expected Results

Your RAG system should demonstrate:

- **Higher relevance** to library-specific questions
- **More specific answers** with exact book titles, authors, dates
- **Better clustering** around question embeddings
- **Consistent performance** across question categories

The evaluation provides quantitative evidence of your RAG system's effectiveness for your project report!
