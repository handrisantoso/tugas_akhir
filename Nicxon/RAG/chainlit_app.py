"""
Chainlit Application for Library Chatbot RAG System
Interactive interface with thinking process display
"""
import os
import asyncio
import logging
from typing import Dict, Any
import chainlit as cl
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import our RAG components
from rag_engine import LibraryRAGEngine
from config import RAGConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global RAG engine instance
rag_engine = None

@cl.on_chat_start
async def start():
    """Initialize the chat session"""
    global rag_engine
    
    # Display welcome message
    await cl.Message(
        content="# 📚 Welcome to the Library Assistant!\n\n"
                "I'm your AI librarian, ready to help you find books from our collection of **8,900+ titles**.\n\n"
                "**What I can help you with:**\n"
                "- 🔍 Search for books by topic, genre, or keywords\n"
                "- 📖 Get book recommendations\n"
                "- ℹ️ Find publication details and acquisition information\n"
                "- 🖼️ **Upload a book cover image** to find visually similar books\n"
                "- 🗣️ Answer questions about library services\n\n"
                "**Example searches:**\n"
                "- \"I'm looking for mystery novels from the 1990s\"\n"
                "- \"Find books about artificial intelligence\"\n"
                "- \"Recommend some Spanish literature\"\n"
                "- *Upload a book cover image* to find visually similar titles\n\n"
                "Just ask me anything about books!"
    ).send()
    
    # Initialize RAG engine
    try:
        rag_engine = LibraryRAGEngine()
        
        # Check system status
        status = rag_engine.get_system_status()
        
        if status['config_valid']:
            await cl.Message(
                content="✅ **System Ready!** Connected to multimodal library database.\n"
                       f"📚 Text collection: **{status.get('text_doc_count', '?')}** chunks | "
                       f"🖼️ Image collection: **{status.get('image_doc_count', '?')}** covers\n"
                       f"🔢 Embedding: **{RAGConfig.EMBEDDING_DISPLAY_NAME}**",
                author="System"
            ).send()
        else:
            await cl.Message(
                content="⚠️ **Configuration Issues Detected:**\n" +
                       "\n".join(f"- {issue}" for issue in status['config_issues']),
                author="System"
            ).send()
            
    except Exception as e:
        logger.error(f"Failed to initialize RAG engine: {e}")
        await cl.Message(
            content="❌ **System Error:** Failed to initialize the library system. "
                   "Please check the configuration and try again.",
            author="System"
        ).send()

@cl.on_message
async def main(message: cl.Message):
    """Handle incoming user messages"""
    global rag_engine
    
    if not rag_engine:
        await cl.Message(
            content="❌ **System Error:** Library system not initialized. Please refresh the page.",
            author="System"
        ).send()
        return
    
    user_message = message.content

    # Extract uploaded image from message elements (if any)
    image_bytes = None
    mime_type = None
    for el in (message.elements or []):
        if isinstance(el, cl.Image):
            try:
                with open(el.path, 'rb') as f:
                    image_bytes = f.read()
                mime_type = getattr(el, 'mime', 'image/jpeg') or 'image/jpeg'
                logger.info(f"Image upload detected: {el.name}, mime={mime_type}, {len(image_bytes)/1024:.1f} KB")
            except Exception as img_err:
                logger.warning(f"Could not read uploaded image: {img_err}")
            break  # Only process the first image

    # Show thinking indicator
    thinking_msg = cl.Message(
        content="🤔 **Thinking...**\n\nLet me search through the library collection and process your request.",
        author="Assistant"
    )
    await thinking_msg.send()
    
    try:
        # Process the query through RAG engine (pass image if uploaded)
        response_data = await rag_engine.process_query(user_message,
                                                        image_bytes=image_bytes,
                                                        mime_type=mime_type)
        
        # Extract components
        thinking_process = response_data.get('thinking', '')
        assistant_response = response_data.get('response', '')
        search_results = response_data.get('search_results', [])
        intent = response_data.get('intent', 'UNKNOWN')
        
        # Update the thinking message with detailed process
        thinking_content = f"🧠 **My Thinking Process:**\n\n{thinking_process}\n\n"
        
        search_mode = response_data.get('search_mode', 'TEXT')
        mode_badge = {
            'TEXT':   '🔤 TEXT search',
            'IMAGE':  '🖼️ IMAGE search',
            'HYBRID': '🔀 HYBRID search (text + image)',
        }.get(search_mode, search_mode)

        if search_results:
            thinking_content += f"📊 **Search Stats:** Found {len(search_results)} relevant books | Intent: {intent} | Mode: {mode_badge}"
        else:
            thinking_content += f"💭 **Processing:** Handled as {intent} intent | Mode: {mode_badge}"
        
        thinking_msg.content = thinking_content
        await thinking_msg.update()
        
        # Send the main response
        response_msg = cl.Message(
            content=assistant_response,
            author="Librarian"
        )
        await response_msg.send()
        
        # If there are search results, offer to show more details
        if search_results:
            await show_search_details(search_results)
        
        # Add conversation stats (minimized)
        stats = rag_engine.get_conversation_stats()
        if stats['total_turns'] > 0:
            stats_content = f"💬 **Conversation:** Turn {stats['total_turns']} | " \
                          f"Searches: {stats.get('search_turns', 0)}"
            
            await cl.Message(
                content=stats_content,
                author="System"
            ).send()
    
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        thinking_msg.content = f"❌ **Error:** {str(e)}\n\nPlease try rephrasing your question or contact support."
        await thinking_msg.update()

async def show_search_details(search_results: list):
    """Display search results in an organized format"""
    if not search_results:
        return
    
    # Create detailed results display
    details_content = "📚 **Detailed Search Results:**\n\n"
    
    # Show search mode badge
    if search_results:
        mode = search_results[0].get('search_mode', 'TEXT')
        mode_labels = {
            'TEXT':   '🔤 TEXT',
            'IMAGE':  '🖼️ IMAGE',
            'HYBRID': '🔀 HYBRID',
        }
        details_content += f"**Search mode:** {mode_labels.get(mode, mode)}\n\n"

    # Show applied filters if any
    if search_results and 'applied_filters' in search_results[0]:
        filters = search_results[0]['applied_filters']
        if filters:
            details_content += "🔍 **Applied Filters:**\n"
            for filter_key, filter_value in filters.items():
                if filter_key == 'language':
                    details_content += f"   🌐 Language: {filter_value.title()}\n"
                elif filter_key == 'publish_year':
                    if isinstance(filter_value, dict):
                        if '$gt' in filter_value:
                            details_content += f"   📅 Published after: {filter_value['$gt']}\n"
                        elif '$lt' in filter_value:
                            details_content += f"   📅 Published before: {filter_value['$lt']}\n"
                        elif '$gte' in filter_value and '$lte' in filter_value:
                            details_content += f"   📅 Published: {filter_value['$gte']}-{filter_value['$lte']}\n"
                    else:
                        details_content += f"   📅 Published: {filter_value}\n"
                elif filter_key == 'page_count':
                    if isinstance(filter_value, dict):
                        if '$lt' in filter_value:
                            details_content += f"   📄 Pages: Under {filter_value['$lt']}\n"
                        elif '$gt' in filter_value:
                            details_content += f"   📄 Pages: Over {filter_value['$gt']}\n"
                    else:
                        details_content += f"   📄 Pages: {filter_value}\n"
                elif filter_key == 'format':
                    details_content += f"   📖 Format: {filter_value.title()}\n"
                elif filter_key == 'has_description':
                    details_content += f"   📝 Content: Books with descriptions\n"
                elif filter_key == 'has_subjects':
                    details_content += f"   🏷️ Content: Books with subject information\n"
            details_content += "\n"
    
    elements = []  # Chainlit image elements to attach

    for i, result in enumerate(search_results[:5], 1):
        doc = result['document']
        metadata = result.get('metadata', {})
        similarity = result.get('similarity_score', 0)
        
        # Extract title from document (assuming it's first in the formatted text)
        lines = doc.split('\n')
        title = lines[0] if lines else "Unknown Title"
        
        details_content += f"**{i}. {title}**\n"
        details_content += f"   📈 Relevance: {similarity:.1%}\n"
        
        # Add metadata if available
        if metadata.get('language'):
            details_content += f"   🌐 Language: {metadata['language'].title()}\n"
        if metadata.get('publish_year'):
            details_content += f"   📅 Year: {metadata['publish_year']}\n"
        if metadata.get('page_count'):
            details_content += f"   📄 Pages: {metadata['page_count']}\n"
        if metadata.get('format'):
            details_content += f"   📖 Format: {metadata['format'].title()}\n"
        
        # Show if book has rich content
        if metadata.get('has_description'):
            details_content += f"   📝 Has detailed description\n"
        if metadata.get('has_toc'):
            details_content += f"   📋 Has table of contents\n"
        
        # Attach cover image if available
        cover_path = metadata.get('cover_image_path', '')
        if metadata.get('has_cover_image') and cover_path and os.path.exists(cover_path):
            try:
                elements.append(
                    cl.Image(
                        name=f"cover_{i}",
                        path=cover_path,
                        display="inline"
                    )
                )
                details_content += f"   🖼️ Cover: see image below (cover_{i})\n"
            except Exception:
                pass  # Silently skip if image can't be loaded
        
        # Show LLM explanation if available
        if result.get('llm_explanation'):
            details_content += f"   💡 Why relevant: {result['llm_explanation'][:100]}...\n"
        
        details_content += f"   📄 Preview: {doc[:200]}...\n\n"
    
    # Send as a collapsible details message with optional cover images
    await cl.Message(
        content=details_content,
        elements=elements,
        author="Search Results"
    ).send()

@cl.on_chat_end
async def end():
    """Handle chat session end"""
    global rag_engine
    
    if rag_engine:
        stats = rag_engine.get_conversation_stats()
        
        await cl.Message(
            content=f"👋 **Thanks for using the Library Assistant!**\n\n"
                   f"Session Summary:\n"
                   f"- Total questions: {stats.get('total_turns', 0)}\n"
                   f"- Book searches: {stats.get('search_turns', 0)}\n"
                   f"- Duration: {stats.get('duration_minutes', 0):.1f} minutes\n\n"
                   f"Happy reading! 📖",
            author="System"
        ).send()
        
        # Clear conversation for next session
        rag_engine.clear_conversation()

# Action handlers for interactive features
@cl.action_callback("show_more_books")
async def show_more_books(action):
    """Action to show more book results"""
    global rag_engine
    
    if rag_engine:
        last_results = rag_engine.conversation_manager.get_last_search_context()
        if last_results and len(last_results) > 5:
            await show_search_details(last_results[5:10])
        else:
            await cl.Message(
                content="No additional results available.",
                author="System"
            ).send()

@cl.action_callback("clear_conversation")
async def clear_conversation(action):
    """Action to clear conversation history"""
    global rag_engine
    
    if rag_engine:
        rag_engine.clear_conversation()
        await cl.Message(
            content="🗑️ **Conversation cleared!** Starting fresh.",
            author="System"
        ).send()

@cl.action_callback("system_status")
async def system_status(action):
    """Action to show system status"""
    global rag_engine
    
    if rag_engine:
        status = rag_engine.get_system_status()
        stats = rag_engine.get_conversation_stats()
        
        status_content = f"🔧 **System Status:**\n\n"
        status_content += f"✅ Configuration: {'Valid' if status['config_valid'] else 'Invalid'}\n"
        status_content += f"📚 Text collection: {status.get('text_collection', '?')} ({status.get('text_doc_count', '?')} docs)\n"
        status_content += f"🖼️ Image collection: {status.get('image_collection', '?')} ({status.get('image_doc_count', '?')} covers)\n"
        status_content += f"🤖 Models: {status['models_used']['llm']}\n"
        status_content += f"🔢 Embedding: {status['models_used']['embedding']} ({RAGConfig.EMBEDDING_BACKEND})\n"
        status_content += f"💬 Current Session: {stats.get('total_turns', 0)} turns\n"
        
        if status['config_issues']:
            status_content += f"\n⚠️ **Issues:**\n"
            for issue in status['config_issues']:
                status_content += f"- {issue}\n"
        
        await cl.Message(
            content=status_content,
            author="System"
        ).send()

# Add action buttons to the interface
def add_action_buttons():
    """Add interactive action buttons"""
    actions = [
        cl.Action(name="show_more_books", value="more", description="📚 Show More Books"),
        cl.Action(name="clear_conversation", value="clear", description="🗑️ Clear History"),
        cl.Action(name="system_status", value="status", description="🔧 System Status"),
    ]
    return actions

# Configuration for Chainlit
if __name__ == "__main__":
    # Set up the Chainlit configuration
    cl.run(
        host="localhost",
        port=8000,
        debug=True,
        headless=False
    ) 