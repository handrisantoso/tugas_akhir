#!/usr/bin/env python3
"""
Master Evaluation Script for Library Chatbot RAG System

This script runs the complete evaluation pipeline:
1. Collect answers from all models (RAG, Gemini, DeepSeek)
2. Generate embeddings for all answers and questions
3. Calculate cosine similarities
4. Generate visualizations for project report

Usage:
    python run_evaluation.py
    
Prerequisites:
    - Set environment variables: GOOGLE_API_KEY, OPENROUTER_API_KEY
      (Can be in .env file or system environment variables)
    - Install requirements: pip install -r requirements.txt
    - Ensure vector database is set up (../vector_db/chroma_db/)
"""

import os
import sys
import subprocess
import time
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def print_banner():
    """Print evaluation banner"""
    print("=" * 70)
    print("🌟 LIBRARY CHATBOT RAG SYSTEM EVALUATION")
    print("=" * 70)
    print("📊 Comparing: RAG System vs Gemini 2.0 vs DeepSeek R1 Zero")
    print("🎯 Metrics: Cosine Similarity, Question Relevance, Model Comparison")
    print("📈 Output: JSON data + 3D visualizations for project report")
    print("=" * 70)

def check_prerequisites():
    """Check if all prerequisites are met"""
    
    print("\n🔍 Checking prerequisites...")
    
    # Check API keys
    google_api_key = os.getenv('GOOGLE_API_KEY')
    openrouter_api_key = os.getenv('OPENROUTER_API_KEY')
    
    if not google_api_key:
        print("❌ GOOGLE_API_KEY environment variable not set")
        print("   Please set it: export GOOGLE_API_KEY='your_key_here'")
        return False
    
    if not openrouter_api_key:
        print("❌ OPENROUTER_API_KEY environment variable not set")
        print("   Please set it: export OPENROUTER_API_KEY='your_key_here'")
        return False
    
    print("✅ API keys found")
    
    # Check vector database
    vector_db_path = "../vector_db/chroma_db"
    if not os.path.exists(vector_db_path):
        print(f"❌ Vector database not found at {vector_db_path}")
        print("   Please run the vector database creation script first")
        return False
    
    print("✅ Vector database found")
    
    # Check test questions
    if not os.path.exists("test_questions.json"):
        print("❌ test_questions.json not found")
        print("   This file should have been created automatically")
        return False
    
    print("✅ Test questions found")
    
    # Check if dependencies are installed
    try:
        import google.generativeai
        import requests
        import sklearn
        import matplotlib
        import plotly
        import seaborn
        print("✅ All dependencies installed")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("   Please install: pip install -r requirements.txt")
        return False
    
    return True

def get_rate_limit_config():
    """Get rate limiting configuration from user"""
    print("\n🚦 Rate Limiting Configuration:")
    print("Google API free tier typically allows 10-15 requests per minute")
    print("The evaluation will make ~60 Google API calls for 10 questions")
    
    try:
        rate_limit = input("Enter your API rate limit (requests/min) [default: 8]: ").strip()
        if not rate_limit:
            rate_limit = 8
        else:
            rate_limit = int(rate_limit)
        
        if rate_limit > 15:
            print("⚠️  Warning: That seems high for free tier. Using 10 instead.")
            rate_limit = 10
        elif rate_limit < 5:
            print("⚠️  Warning: That seems too low. Using 5 instead.")
            rate_limit = 5
            
    except (ValueError, KeyboardInterrupt):
        print("Using default rate limit: 8 requests/min")
        rate_limit = 8
    
    # Calculate estimated time
    total_api_calls = 60  # Estimate for 10 questions
    estimated_minutes = (total_api_calls * (60 / rate_limit + 5)) / 60  # +5s buffer
    
    print(f"✅ Using rate limit: {rate_limit} requests/min")
    print(f"⏱️  Estimated completion time: {estimated_minutes:.1f} minutes")
    
    return rate_limit

def run_step(step_name: str, script_name: str, description: str, **kwargs):
    """Run a single evaluation step"""
    
    print(f"\n🚀 Step: {step_name}")
    print(f"📝 {description}")
    print("─" * 50)
    
    start_time = time.time()
    
    try:
        # Prepare command with any additional arguments
        cmd = [sys.executable, script_name]
        
        # Add environment variables for configuration
        env = os.environ.copy()
        for key, value in kwargs.items():
            env[key.upper()] = str(value)
        
        # Run the script with modified environment
        result = subprocess.run(cmd, env=env, check=True)
        
        elapsed_time = time.time() - start_time
        print(f"\n✅ {step_name} completed successfully!")
        print(f"⏱️  Time taken: {elapsed_time:.1f} seconds")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {step_name} failed with error code {e.returncode}")
        print("Check the output above for error details")
        return False
    except Exception as e:
        print(f"\n❌ {step_name} failed with error: {e}")
        return False

def run_direct_import_method():
    """Alternative method: Run steps by importing modules directly"""
    
    print("\n🔄 Running evaluation using direct imports (better error handling)...")
    
    try:
        # Import and run collect_answers
        print("\n🚀 Step 1: Collecting answers...")
        import collect_answers
        import asyncio
        asyncio.run(collect_answers.main())
        
        # Import and run generate_embeddings  
        print("\n🚀 Step 2: Generating embeddings...")
        import generate_embeddings
        asyncio.run(generate_embeddings.main())
        
        # Import and run visualize_embeddings
        print("\n🚀 Step 3: Creating visualizations...")
        import visualize_embeddings
        visualize_embeddings.main()
        
        return True
        
    except Exception as e:
        print(f"❌ Direct import method failed: {e}")
        return False

def main():
    """Run the complete evaluation pipeline"""
    
    print_banner()
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n❌ Prerequisites not met. Please fix the issues above.")
        return 1
    
    # Get rate limiting configuration
    rate_limit = get_rate_limit_config()
    
    print("\n🎯 All prerequisites met! Starting evaluation...")
    
    # Record start time
    total_start_time = time.time()
    
    # Choose execution method
    print("\n📋 Choose execution method:")
    print("1. Subprocess method (runs scripts independently)")  
    print("2. Direct import method (better error handling, recommended)")
    
    try:
        method = input("Enter choice [1 or 2, default: 2]: ").strip()
        if not method:
            method = "2"
    except KeyboardInterrupt:
        method = "2"
    
    success = False
    
    if method == "1":
        # Method 1: Subprocess calls (original method)
        print("\n🔧 Using subprocess method...")
        
        # Step 1: Collect answers
        success = run_step(
            "Answer Collection",
            "collect_answers.py", 
            "Collecting answers from RAG system, Gemini 2.0, and DeepSeek R1 Zero",
            rate_limit=rate_limit
        )
        
        if not success:
            print("\n💥 Evaluation failed at answer collection step")
            return 1
        
        # Step 2: Generate embeddings and similarities
        success = run_step(
            "Embedding Analysis",
            "generate_embeddings.py",
            "Generating embeddings and calculating cosine similarities"
        )
        
        if not success:
            print("\n💥 Evaluation failed at embedding analysis step")
            return 1
        
        # Step 3: Create visualizations
        success = run_step(
            "Visualization Generation", 
            "visualize_embeddings.py",
            "Creating 3D visualizations and similarity heatmaps"
        )
        
        if not success:
            print("\n💥 Evaluation failed at visualization step")
            return 1
    
    else:
        # Method 2: Direct imports (recommended)
        success = run_direct_import_method()
        
        if not success:
            print("\n💥 Evaluation failed. Try running scripts individually.")
            return 1
    
    # Calculate total time
    total_time = time.time() - total_start_time
    
    # Success summary
    print("\n" + "=" * 70)
    print("🎉 EVALUATION COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"⏱️  Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"📅 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n📁 Generated Files:")
    files_to_check = [
        ("model_answers.json", "Raw answers from all models"),
        ("model_embeddings.json", "768D embeddings for all texts"),
        ("cosine_similarities.json", "Similarity scores between all pairs"),
        ("evaluation_report.json", "Comprehensive analysis report"),
        ("visualizations/", "3D plots and heatmaps for your report")
    ]
    
    for filename, description in files_to_check:
        if os.path.exists(filename):
            print(f"   ✅ {filename} - {description}")
        else:
            print(f"   ⚠️  {filename} - {description} (not found)")
    
    print("\n🎨 Visualization Files:")
    viz_files = [
        "visualizations/embedding_visualization_pca.png",
        "visualizations/embedding_visualization_pca_interactive.html",
        "visualizations/embedding_visualization_tsne.png", 
        "visualizations/embedding_visualization_tsne_interactive.html",
        "visualizations/similarity_heatmap.png",
        "visualizations/README.md"
    ]
    
    for viz_file in viz_files:
        if os.path.exists(viz_file):
            print(f"   ✅ {viz_file}")
        else:
            print(f"   ⚠️  {viz_file} (not found)")
    
    print("\n📋 Next Steps:")
    print("   1. Review evaluation_report.json for quantitative results")
    print("   2. Use 3D visualizations in your project report")
    print("   3. Include similarity heatmap in your analysis")
    print("   4. Compare question relevance scores between models")
    print("   5. Analyze clustering patterns in the embeddings")
    
    print("\n🏆 Your RAG system evaluation is complete!")
    
    # Offer to open results
    try:
        open_results = input("\n📊 Open evaluation report? [y/N]: ").strip().lower()
        if open_results == 'y':
            if os.path.exists("evaluation_report.json"):
                import json
                with open("evaluation_report.json", 'r') as f:
                    report = json.load(f)
                
                summary = report.get('evaluation_summary', {})
                print("\n📈 Quick Results Summary:")
                print(f"   Questions processed: {summary.get('total_questions', 'N/A')}")
                
                relevance = summary.get('average_relevance_scores', {})
                print(f"   RAG relevance score: {relevance.get('RAG_to_question', 0):.3f}")
                print(f"   Gemini relevance score: {relevance.get('Gemini_to_question', 0):.3f}")
                print(f"   DeepSeek relevance score: {relevance.get('DeepSeek_to_question', 0):.3f}")
    except KeyboardInterrupt:
        pass
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 