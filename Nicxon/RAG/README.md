# Library Chatbot RAG System

An intelligent library assistant powered by RAG with semantic search over 8,900+ book records.

## Quick Setup

1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

2. **Set Environment Variable**
```bash
# Create .env file
echo "GOOGLE_API_KEY=your_actual_api_key_here" > .env
```

3. **Run the Application**
```bash
chainlit run chainlit_app.py
```

4. **Access Interface**
Open http://localhost:8000 in your browser

## Features

- **Intent Classification** - Understands different query types
- **Semantic Search** - Vector-based book search
- **Thinking Process Display** - Shows AI reasoning
- **Context Memory** - Maintains conversation flow
- **Multi-language Support** - Search across 44+ languages

## Requirements

- Google AI API Key
- Vector database at `../vector_db/chroma_db/`
- Python 3.8+

## Troubleshooting

**Import Error**: Make sure all files are created properly and dependencies installed
**API Error**: Check your GOOGLE_API_KEY in .env file
**Database Error**: Verify vector database exists at correct path

Ready to explore your library collection! 🚀📚 