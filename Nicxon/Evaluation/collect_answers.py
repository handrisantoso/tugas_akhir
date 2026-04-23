import json
import os
import sys
import time
from typing import List, Dict, Any
import google.generativeai as genai
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
import asyncio

# Load environment variables from .env file
load_dotenv()

# Add parent directory to path to import RAG engine
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'RAG'))
from rag_engine import LibraryRAGEngine

class RateLimiter:
    """Rate limiter to respect API limits"""
    
    def __init__(self, requests_per_minute: int = 10, buffer_seconds: int = 5):
        self.requests_per_minute = requests_per_minute
        self.buffer_seconds = buffer_seconds
        self.request_times = []
        
        # Calculate delay between requests (with buffer)
        self.min_delay = (60 / requests_per_minute) + buffer_seconds
        
        print(f"🚦 Rate limiter initialized: {requests_per_minute} requests/min")
        print(f"⏱️  Minimum delay between requests: {self.min_delay:.1f} seconds")
    
    async def wait_if_needed(self):
        """Wait if necessary to respect rate limits"""
        current_time = datetime.now()
        
        # Remove requests older than 1 minute
        cutoff_time = current_time - timedelta(minutes=1)
        self.request_times = [t for t in self.request_times if t > cutoff_time]
        
        # Check if we're at the limit
        if len(self.request_times) >= self.requests_per_minute:
            # Wait until the oldest request is more than 1 minute old
            wait_until = self.request_times[0] + timedelta(minutes=1, seconds=self.buffer_seconds)
            if current_time < wait_until:
                wait_seconds = (wait_until - current_time).total_seconds()
                print(f"⏸️  Rate limit reached. Waiting {wait_seconds:.1f} seconds...")
                await asyncio.sleep(wait_seconds)
        
        # Record this request
        self.request_times.append(datetime.now())

class ModelEvaluator:
    def __init__(self, google_api_key: str, openrouter_api_key: str, rate_limit: int = 8):
        """Initialize the evaluator with API keys and rate limiting"""
        
        # Configure Google API for RAG and raw Gemini
        genai.configure(api_key=google_api_key)
        self.google_api_key = google_api_key
        
        # OpenRouter API for DeepSeek
        self.openrouter_api_key = openrouter_api_key
        self.openrouter_headers = {
            "Authorization": f"Bearer {openrouter_api_key}",
            "Content-Type": "application/json"
        }
        
        # Initialize rate limiter (conservative: 8 requests/min instead of 10)
        self.rate_limiter = RateLimiter(requests_per_minute=rate_limit)
        
        # Initialize RAG system
        print("🔧 Initializing RAG system...")
        self.rag_system = LibraryRAGEngine()
        
        # Initialize raw Gemini model
        self.gemini_model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        print("✅ All models initialized successfully!")
    
    async def get_rag_answer(self, question: str) -> str:
        """Get answer from RAG system (complete pipeline)"""
        try:
            print("    🔄 Processing through RAG pipeline...")
            
            # The RAG system makes multiple API calls internally
            # We'll add delays within the RAG system calls
            response_data = await self.rag_system.process_query(question)
            
            # Extract the final response from the RAG engine
            if 'response' in response_data:
                return response_data['response']
            elif 'full_response' in response_data:
                # Parse THINKING/RESPONSE format if needed
                full_response = response_data['full_response']
                if "RESPONSE:" in full_response:
                    return full_response.split("RESPONSE:")[-1].strip()
                else:
                    return full_response
            else:
                return "I couldn't process your query through the RAG system."
            
        except Exception as e:
            return f"Error generating RAG response: {str(e)}"
    
    async def get_gemini_raw_answer(self, question: str) -> str:
        """Get answer from raw Gemini 2.0 (no RAG context)"""
        try:
            await self.rate_limiter.wait_if_needed()
            
            prompt = f"""Please answer this library-related question: {question}

Provide a helpful and informative response about books, authors, or literature."""
            
            response = self.gemini_model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            return f"Error generating Gemini response: {str(e)}"
    
    async def get_deepseek_answer(self, question: str) -> str:
        """Get answer from DeepSeek R1 Zero via OpenRouter"""
        try:
            # OpenRouter has different rate limits, but we'll still be conservative
            await asyncio.sleep(2)  # Small delay for OpenRouter
            
            url = "https://openrouter.ai/api/v1/chat/completions"
            
            data = {
                "model": "deepseek/deepseek-r1-zero:free",
                "messages": [
                    {
                        "role": "user", 
                        "content": f"Please answer this library-related question: {question}\n\nProvide a helpful and informative response about books, authors, or literature."
                    }
                ]
            }
            
            response = requests.post(url, json=data, headers=self.openrouter_headers, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            return result['choices'][0]['message']['content']
            
        except Exception as e:
            return f"Error generating DeepSeek response: {str(e)}"
    
    async def collect_all_answers(self, questions_file: str, output_file: str):
        """Collect answers from all models for all questions"""
        
        print("📖 Loading test questions...")
        with open(questions_file, 'r', encoding='utf-8') as f:
            questions = json.load(f)
        
        results = []
        total_questions = len(questions)
        
        # Calculate estimated time
        # RAG system: ~5 API calls per question
        # Raw Gemini: 1 API call per question  
        # DeepSeek: 1 API call (different service)
        # Total Google API calls: ~6 per question
        total_api_calls = total_questions * 6
        estimated_minutes = (total_api_calls * self.rate_limiter.min_delay) / 60
        
        print(f"🚀 Starting evaluation with {total_questions} questions...")
        print(f"📊 Estimated Google API calls: {total_api_calls}")
        print(f"⏱️  Estimated completion time: {estimated_minutes:.1f} minutes")
        print("=" * 60)
        
        start_time = datetime.now()
        
        for i, q in enumerate(questions, 1):
            question_text = q['question']
            print(f"\n[{i}/{total_questions}] Processing: {question_text[:50]}...")
            
            # Initialize result structure
            result = {
                "id": q['id'],
                "question": question_text,
                "category": q['category'],
                "difficulty": q['difficulty'],
                "RAG_gemini_flash_2.0": "",
                "gemini_flash_2.0": "",
                "deepseek_r1_zero": ""
            }
            
            # Get RAG answer (this makes multiple API calls internally)
            print("  🤖 Getting RAG answer...")
            result["RAG_gemini_flash_2.0"] = await self.get_rag_answer(question_text)
            
            # Get raw Gemini answer (rate-limited)
            print("  💎 Getting Gemini raw answer...")
            result["gemini_flash_2.0"] = await self.get_gemini_raw_answer(question_text)
            
            # Get DeepSeek answer (different API)
            print("  🧠 Getting DeepSeek answer...")
            result["deepseek_r1_zero"] = await self.get_deepseek_answer(question_text)
            
            results.append(result)
            
            # Save progress incrementally
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            # Calculate progress and ETA
            elapsed_time = datetime.now() - start_time
            progress_percent = (i / total_questions) * 100
            if i > 1:
                avg_time_per_question = elapsed_time.total_seconds() / (i - 1)
                remaining_questions = total_questions - i
                eta_seconds = remaining_questions * avg_time_per_question
                eta_time = datetime.now() + timedelta(seconds=eta_seconds)
                
                print(f"  ✅ Question {i} completed ({progress_percent:.1f}%)")
                print(f"  ⏱️  ETA: {eta_time.strftime('%H:%M:%S')} ({eta_seconds/60:.1f} min remaining)")
            else:
                print(f"  ✅ Question {i} completed")
        
        total_time = datetime.now() - start_time
        
        print("\n" + "=" * 60)
        print(f"🎉 Evaluation completed! Results saved to {output_file}")
        print(f"📊 Total questions processed: {total_questions}")
        print(f"⏱️  Total time: {total_time.total_seconds()/60:.1f} minutes")
        
        # Display sample results
        print("\n📋 Sample Results Preview:")
        for result in results[:2]:  # Show first 2 results
            print(f"\nQ: {result['question']}")
            print(f"RAG: {result['RAG_gemini_flash_2.0'][:100]}...")
            print(f"Gemini: {result['gemini_flash_2.0'][:100]}...")
            print(f"DeepSeek: {result['deepseek_r1_zero'][:100]}...")
    
    def generate_statistics(self, results_file: str):
        """Generate basic statistics about the collected answers"""
        
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        print("\n📊 Answer Collection Statistics:")
        print("=" * 50)
        print(f"Total questions: {len(results)}")
        
        # Calculate average answer lengths
        rag_lengths = [len(r['RAG_gemini_flash_2.0']) for r in results]
        gemini_lengths = [len(r['gemini_flash_2.0']) for r in results]
        deepseek_lengths = [len(r['deepseek_r1_zero']) for r in results]
        
        print(f"\nAverage Answer Lengths:")
        print(f"  RAG Gemini:     {sum(rag_lengths)/len(rag_lengths):.0f} characters")
        print(f"  Raw Gemini:     {sum(gemini_lengths)/len(gemini_lengths):.0f} characters")
        print(f"  DeepSeek R1:    {sum(deepseek_lengths)/len(deepseek_lengths):.0f} characters")
        
        # Category breakdown
        categories = {}
        for result in results:
            cat = result['category']
            categories[cat] = categories.get(cat, 0) + 1
        
        print(f"\nQuestions by Category:")
        for cat, count in categories.items():
            print(f"  {cat}: {count}")

async def main():
    """Main evaluation runner"""
    
    print("🌟 Library Chatbot Model Evaluation")
    print("=" * 50)
    
    # Check for API keys
    google_api_key = os.getenv('GOOGLE_API_KEY')
    openrouter_api_key = os.getenv('OPENROUTER_API_KEY')
    
    if not google_api_key:
        print("❌ Error: GOOGLE_API_KEY environment variable not set")
        print("Please set your Google API key: export GOOGLE_API_KEY='your_key_here'")
        return
    
    if not openrouter_api_key:
        print("❌ Error: OPENROUTER_API_KEY environment variable not set")
        print("Please set your OpenRouter API key: export OPENROUTER_API_KEY='your_key_here'")
        return
    
    # Get rate limiting configuration (from environment or user input)
    rate_limit = os.getenv('RATE_LIMIT')
    
    if rate_limit:
        # Rate limit passed from master script
        rate_limit = int(rate_limit)
        print(f"🚦 Using rate limit from configuration: {rate_limit} requests/min")
    else:
        # Ask user for rate limiting configuration
        print("\n🚦 Rate Limiting Configuration:")
        print("Google API free tier typically allows 10-15 requests per minute")
        
        try:
            rate_limit_input = input("Enter your API rate limit (requests/min) [default: 8]: ").strip()
            if not rate_limit_input:
                rate_limit = 8
            else:
                rate_limit = int(rate_limit_input)
            
            if rate_limit > 15:
                print("⚠️  Warning: That seems high for free tier. Using 10 instead.")
                rate_limit = 10
                
        except (ValueError, KeyboardInterrupt):
            print("Using default rate limit: 8 requests/min")
            rate_limit = 8
    
    # Initialize evaluator with rate limiting
    evaluator = ModelEvaluator(google_api_key, openrouter_api_key, rate_limit=rate_limit)
    
    # File paths
    questions_file = "test_questions.json"
    answers_file = "model_answers.json"
    
    # Collect answers
    await evaluator.collect_all_answers(questions_file, answers_file)
    
    # Generate statistics
    evaluator.generate_statistics(answers_file)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main()) 