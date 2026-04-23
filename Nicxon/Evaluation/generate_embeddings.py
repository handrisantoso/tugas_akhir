import json
import numpy as np
import os
import sys
from typing import List, Dict, Any
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class EmbeddingAnalyzer:
    def __init__(self, google_api_key: str):
        """Initialize the embedding analyzer"""
        genai.configure(api_key=google_api_key)
        self.embedding_model = 'text-embedding-004'
        print("✅ Embedding analyzer initialized!")
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        try:
            result = genai.embed_content(
                model=self.embedding_model,
                content=text,
                task_type="retrieval_document"
            )
            return result['embedding']
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return [0.0] * 768  # Return zero vector as fallback
    
    def generate_all_embeddings(self, answers_file: str, embeddings_file: str):
        """Generate embeddings for all answers and questions"""
        
        print("📖 Loading model answers...")
        with open(answers_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        results = []
        total_items = len(data)
        
        print(f"🧠 Generating embeddings for {total_items} questions...")
        print("=" * 60)
        
        for i, item in enumerate(data, 1):
            print(f"\n[{i}/{total_items}] Processing question {item['id']}...")
            
            # Generate embeddings for question and all answers
            question_embedding = self.generate_embedding(item['question'])
            time.sleep(0.5)  # Rate limiting
            
            rag_embedding = self.generate_embedding(item['RAG_gemini_flash_2.0'])
            time.sleep(0.5)
            
            gemini_embedding = self.generate_embedding(item['gemini_flash_2.0'])
            time.sleep(0.5)
            
            deepseek_embedding = self.generate_embedding(item['deepseek_r1_zero'])
            time.sleep(0.5)
            
            # Create embedding result
            embedding_result = {
                "id": item['id'],
                "question": question_embedding,
                "RAG_gemini_flash_2.0": rag_embedding,
                "gemini_flash_2.0": gemini_embedding,
                "deepseek_r1_zero": deepseek_embedding,
                "metadata": {
                    "category": item['category'],
                    "difficulty": item['difficulty'],
                    "question_text": item['question'][:100] + "..." if len(item['question']) > 100 else item['question']
                }
            }
            
            results.append(embedding_result)
            
            # Save progress incrementally
            with open(embeddings_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
            
            print(f"  ✅ Embeddings generated for question {i}")
        
        print(f"\n🎉 All embeddings generated! Saved to {embeddings_file}")
        return results
    
    def calculate_cosine_similarities(self, embeddings_file: str, similarities_file: str):
        """Calculate cosine similarities between all embeddings"""
        
        print("📊 Loading embeddings...")
        with open(embeddings_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        results = []
        
        print("🔍 Calculating cosine similarities...")
        print("=" * 60)
        
        for item in data:
            question_vec = np.array(item['question']).reshape(1, -1)
            rag_vec = np.array(item['RAG_gemini_flash_2.0']).reshape(1, -1)
            gemini_vec = np.array(item['gemini_flash_2.0']).reshape(1, -1)
            deepseek_vec = np.array(item['deepseek_r1_zero']).reshape(1, -1)
            
            # Calculate all pairwise similarities
            similarities = {
                "id": item['id'],
                "category": item['metadata']['category'],
                "difficulty": item['metadata']['difficulty'],
                "question_text": item['metadata']['question_text'],
                
                # Question vs Answers (Relevance scores)
                "question_vs_rag": float(cosine_similarity(question_vec, rag_vec)[0][0]),
                "question_vs_gemini": float(cosine_similarity(question_vec, gemini_vec)[0][0]),
                "question_vs_deepseek": float(cosine_similarity(question_vec, deepseek_vec)[0][0]),
                
                # Answer vs Answer (Model comparison)
                "rag_vs_gemini": float(cosine_similarity(rag_vec, gemini_vec)[0][0]),
                "rag_vs_deepseek": float(cosine_similarity(rag_vec, deepseek_vec)[0][0]),
                "gemini_vs_deepseek": float(cosine_similarity(gemini_vec, deepseek_vec)[0][0])
            }
            
            results.append(similarities)
            
            print(f"Question {item['id']}: Q→RAG={similarities['question_vs_rag']:.3f}, "
                  f"Q→Gemini={similarities['question_vs_gemini']:.3f}, "
                  f"Q→DeepSeek={similarities['question_vs_deepseek']:.3f}")
        
        # Save similarities
        with open(similarities_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Similarities calculated! Saved to {similarities_file}")
        return results
    
    def generate_analysis_report(self, similarities_file: str, report_file: str):
        """Generate comprehensive analysis report"""
        
        with open(similarities_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Calculate aggregate statistics
        question_vs_rag = [item['question_vs_rag'] for item in data]
        question_vs_gemini = [item['question_vs_gemini'] for item in data]
        question_vs_deepseek = [item['question_vs_deepseek'] for item in data]
        
        rag_vs_gemini = [item['rag_vs_gemini'] for item in data]
        rag_vs_deepseek = [item['rag_vs_deepseek'] for item in data]
        gemini_vs_deepseek = [item['gemini_vs_deepseek'] for item in data]
        
        report = {
            "evaluation_summary": {
                "total_questions": len(data),
                "average_relevance_scores": {
                    "RAG_to_question": np.mean(question_vs_rag),
                    "Gemini_to_question": np.mean(question_vs_gemini),
                    "DeepSeek_to_question": np.mean(question_vs_deepseek)
                },
                "average_inter_model_similarities": {
                    "RAG_vs_Gemini": np.mean(rag_vs_gemini),
                    "RAG_vs_DeepSeek": np.mean(rag_vs_deepseek),
                    "Gemini_vs_DeepSeek": np.mean(gemini_vs_deepseek)
                }
            },
            "detailed_analysis": {
                "best_question_relevance": {
                    "RAG": max(question_vs_rag),
                    "Gemini": max(question_vs_gemini),
                    "DeepSeek": max(question_vs_deepseek)
                },
                "worst_question_relevance": {
                    "RAG": min(question_vs_rag),
                    "Gemini": min(question_vs_gemini),
                    "DeepSeek": min(question_vs_deepseek)
                },
                "consistency_analysis": {
                    "RAG_consistency": np.std(question_vs_rag),
                    "Gemini_consistency": np.std(question_vs_gemini),
                    "DeepSeek_consistency": np.std(question_vs_deepseek)
                }
            },
            "category_performance": {},
            "individual_results": data
        }
        
        # Category-wise analysis
        categories = set(item['category'] for item in data)
        for category in categories:
            cat_data = [item for item in data if item['category'] == category]
            
            cat_rag_scores = [item['question_vs_rag'] for item in cat_data]
            cat_gemini_scores = [item['question_vs_gemini'] for item in cat_data]
            cat_deepseek_scores = [item['question_vs_deepseek'] for item in cat_data]
            
            report["category_performance"][category] = {
                "count": len(cat_data),
                "RAG_avg": np.mean(cat_rag_scores),
                "Gemini_avg": np.mean(cat_gemini_scores),
                "DeepSeek_avg": np.mean(cat_deepseek_scores)
            }
        
        # Save report
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        self.print_analysis_summary(report)
        
        return report
    
    def print_analysis_summary(self, report: Dict[str, Any]):
        """Print a readable summary of the analysis"""
        
        print("\n" + "=" * 70)
        print("📊 MODEL EVALUATION ANALYSIS SUMMARY")
        print("=" * 70)
        
        summary = report['evaluation_summary']
        
        print(f"\n🎯 QUESTION RELEVANCE SCORES (Higher = More Relevant to Question)")
        print(f"  RAG System:     {summary['average_relevance_scores']['RAG_to_question']:.3f}")
        print(f"  Raw Gemini:     {summary['average_relevance_scores']['Gemini_to_question']:.3f}")
        print(f"  DeepSeek R1:    {summary['average_relevance_scores']['DeepSeek_to_question']:.3f}")
        
        # Determine winner
        relevance_scores = summary['average_relevance_scores']
        best_model = max(relevance_scores, key=relevance_scores.get)
        print(f"\n🏆 MOST RELEVANT TO QUESTIONS: {best_model.replace('_to_question', '').replace('_', ' ')}")
        
        print(f"\n🔄 INTER-MODEL SIMILARITIES (How similar are the answers?)")
        inter_model = summary['average_inter_model_similarities']
        print(f"  RAG vs Gemini:    {inter_model['RAG_vs_Gemini']:.3f}")
        print(f"  RAG vs DeepSeek:  {inter_model['RAG_vs_DeepSeek']:.3f}")
        print(f"  Gemini vs DeepSeek: {inter_model['Gemini_vs_DeepSeek']:.3f}")
        
        print(f"\n📂 CATEGORY PERFORMANCE")
        for category, data in report['category_performance'].items():
            print(f"  {category.replace('_', ' ').title()}:")
            print(f"    RAG: {data['RAG_avg']:.3f} | Gemini: {data['Gemini_avg']:.3f} | DeepSeek: {data['DeepSeek_avg']:.3f}")
        
        print("\n" + "=" * 70)

def main():
    """Main embedding analysis runner"""
    
    print("🧠 Library Chatbot Embedding Analysis")
    print("=" * 50)
    
    # Check for API key
    google_api_key = os.getenv('GOOGLE_API_KEY')
    
    if not google_api_key:
        print("❌ Error: GOOGLE_API_KEY environment variable not set")
        return
    
    # Initialize analyzer
    analyzer = EmbeddingAnalyzer(google_api_key)
    
    # File paths
    answers_file = "model_answers.json"
    embeddings_file = "model_embeddings.json"
    similarities_file = "cosine_similarities.json"
    report_file = "evaluation_report.json"
    
    # Check if answers file exists
    if not os.path.exists(answers_file):
        print(f"❌ Error: {answers_file} not found. Please run collect_answers.py first.")
        return
    
    # Generate embeddings
    print("\n🚀 Step 1: Generating embeddings...")
    analyzer.generate_all_embeddings(answers_file, embeddings_file)
    
    # Calculate similarities
    print("\n🚀 Step 2: Calculating cosine similarities...")
    analyzer.calculate_cosine_similarities(embeddings_file, similarities_file)
    
    # Generate analysis report
    print("\n🚀 Step 3: Generating analysis report...")
    analyzer.generate_analysis_report(similarities_file, report_file)
    
    print(f"\n🎉 Analysis complete! Check these files:")
    print(f"  📊 Embeddings: {embeddings_file}")
    print(f"  📈 Similarities: {similarities_file}")
    print(f"  📋 Full Report: {report_file}")

if __name__ == "__main__":
    main() 