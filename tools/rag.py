# rag.py
import os
import re
import json
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import uuid

# tool_configs for agent (only the function w/ @mcp.tools)
tool_configs = [
    {
        "type": "function",
        "function": {
            "name": "rag_query",
            "description": "A RAG tool to search the bpy and Infinigen documentation for information relevant to the current instruction, in order to support your use of execute_and_evaluate.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "The query to search the bpy and Infinigen documentation for information relevant to the current instruction."}},
                "required": ["query"]
            }
        }
    }
]

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None

try:
    import chromadb
    from chromadb.config import Settings, DEFAULT_TENANT, DEFAULT_DATABASE
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    chromadb = None

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None


class BlenderInfinigenRAG:
    """RAG tool: Vector-based search through knowledge.jsonl for Blender and Infinigen documentation"""
    
    def __init__(self, openai_api_key: str = None, knowledge_file: str = None):
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.knowledge_file = knowledge_file or "tools/knowledge_base/rag_kb_deduplicated.jsonl"
        
        # Initialize OpenAI client
        if self.openai_api_key and OPENAI_AVAILABLE:
            self.openai_client = OpenAI(api_key=self.openai_api_key)
        else:
            self.openai_client = None
            if not OPENAI_AVAILABLE:
                logging.warning("OpenAI library not available. Enhanced generation disabled.")
            elif not self.openai_api_key:
                logging.warning("OpenAI API key not provided. Enhanced generation disabled.")
        
        # Initialize vector database
        self.vector_db = None
        self.embedding_model = None
        self._init_vector_db()
        
        # Instruction pattern matching for code generation
        self.instruction_patterns = {
            'physics_placement': [
                r'physics.*placement|rigid.*body|physics.*simulation',
                r'gravity|collision|rigidbody|physics.*world'
            ],
            'object_creation': [
                r'create.*object|add.*object|generate.*object',
                r'primitive|basic.*shape|mesh.*add|bpy\.ops\.mesh'
            ],
            'lighting': [
                r'lighting|illumination|light|shadow',
                r'material|texture|shader|node'
            ],
            'animation': [
                r'animation|keyframe|motion|transform',
                r'timeline|fcurve|driver|constraint'
            ],
            'scene_setup': [
                r'scene.*setup|environment|camera|render',
                r'world|background|composition'
            ]
        }
    
    def _get_collection_name(self) -> str:
        """Generate collection name based on knowledge file name"""
        # Extract filename without extension and use as collection name
        filename = Path(self.knowledge_file).stem
        return filename
    
    def _init_vector_db(self):
        """Initialize vector database for knowledge search"""
        if not CHROMADB_AVAILABLE:
            logging.warning("ChromaDB not available. Falling back to simple text search.")
            return
        
        try:
            # Initialize ChromaDB client
            self.vector_db = chromadb.PersistentClient(
                path="./tools/knowledge_base/chroma",              # Local directory (will be created automatically)
                settings=Settings(),          # Optional: Settings(anonymized_telemetry=False)
                tenant=DEFAULT_TENANT,        # Optional
                database=DEFAULT_DATABASE     # Optional
            )
            
            # Initialize embedding model
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            elif self.openai_client:
                # Use OpenAI embeddings as fallback
                self.embedding_model = "openai"
            else:
                logging.warning("No embedding model available. Using simple text search.")
                return
            
            # Load knowledge from JSONL file
            self._load_knowledge_to_vector_db()
            
        except Exception as e:
            logging.error(f"Failed to initialize vector database: {e}")
            self.vector_db = None
    
    def _load_knowledge_to_vector_db(self):
        """Load knowledge from JSONL file into vector database"""
        if not self.vector_db or not os.path.exists(self.knowledge_file):
            return
        
        try:
            # Use knowledge file path as collection name to enable different collections for different files
            collection_name = self._get_collection_name()
            try:
                collection = self.vector_db.get_collection(collection_name)
                logging.info(f"Using existing knowledge collection: {collection_name}")
                return
            except:
                collection = self.vector_db.create_collection(collection_name)
                logging.info(f"Created new knowledge collection: {collection_name}")
            
            # Load and process knowledge entries
            documents = []
            metadatas = []
            ids = []
            
            with open(self.knowledge_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    if not line.strip():
                        continue
                    
                    try:
                        entry = json.loads(line.strip())
                        
                        # Create searchable text from entry
                        searchable_text = self._create_searchable_text(entry)
                        
                        documents.append(searchable_text)
                        metadatas.append({
                            'id': entry.get('id', str(uuid.uuid4())),
                            'domain': entry.get('domain', ''),
                            'title': entry.get('title', ''),
                            'url': entry.get('url', ''),
                            'version': entry.get('version', ''),
                            'tags': ','.join(entry.get('tags', [])),
                            'updated': entry.get('updated', ''),
                            'source_type': entry.get('source_type', ''),
                            'section_path': ','.join(entry.get('section_path', [])),
                            'line_number': line_num
                        })
                        ids.append(entry.get('id', str(uuid.uuid4())))
                        
                    except json.JSONDecodeError as e:
                        logging.warning(f"Failed to parse line {line_num}: {e}")
                        continue
            
            if documents:
                # Add documents to collection
                collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                logging.info(f"Loaded {len(documents)} knowledge entries into vector database")
            
        except Exception as e:
            logging.error(f"Failed to load knowledge to vector database: {e}")
    
    def _create_searchable_text(self, entry: Dict) -> str:
        """Create searchable text from knowledge entry"""
        parts = []
        
        # Add title
        if entry.get('title'):
            parts.append(f"Title: {entry['title']}")
        
        # Add content summary
        if entry.get('content_summary'):
            parts.append(f"Content: {entry['content_summary']}")
        
        # Add tags
        if entry.get('tags'):
            parts.append(f"Tags: {', '.join(entry['tags'])}")
        
        # Add section path
        if entry.get('section_path'):
            parts.append(f"Section: {' > '.join(entry['section_path'])}")
        
        # Add domain
        if entry.get('domain'):
            parts.append(f"Domain: {entry['domain']}")
        
        return "\n".join(parts)
    
    def search_vector_knowledge(self, query: str, max_results: int = 5, domain_filter: str = None) -> List[Dict]:
        """Search knowledge using vector similarity"""
        if not self.vector_db:
            return self._fallback_text_search(query, max_results, domain_filter)
        
        try:
            # Use the same collection name as used during loading
            collection_name = self._get_collection_name()
            collection = self.vector_db.get_collection(collection_name)
            
            # Apply domain filter if specified
            where_clause = None
            if domain_filter:
                where_clause = {"domain": domain_filter}
            
            # Perform vector search
            results = collection.query(
                query_texts=[query],
                n_results=max_results,
                where=where_clause
            )
            
            # Format results
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i] if results['metadatas'] and results['metadatas'][0] else {}
                    distance = results['distances'][0][i] if results['distances'] and results['distances'][0] else 0
                    
                    formatted_results.append({
                        'title': metadata.get('title', 'Unknown'),
                        'url': metadata.get('url', ''),
                        'snippet': doc,
                        'source': metadata.get('domain', 'unknown'),
                        'similarity': 1 - distance,  # Convert distance to similarity
                        'tags': metadata.get('tags', ''),
                        'section_path': metadata.get('section_path', ''),
                        'content': doc
                    })
            
            return formatted_results
            
        except Exception as e:
            logging.error(f"Vector search failed: {e}")
            return self._fallback_text_search(query, max_results, domain_filter)
    
    def _fallback_text_search(self, query: str, max_results: int = 5, domain_filter: str = None) -> List[Dict]:
        """Fallback text search when vector search is not available"""
        if not os.path.exists(self.knowledge_file):
            return []
        
        results = []
        query_lower = query.lower()
        
        try:
            with open(self.knowledge_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    if not line.strip():
                        continue
                    
                    try:
                        entry = json.loads(line.strip())
                        
                        # Apply domain filter
                        if domain_filter and entry.get('domain') != domain_filter:
                            continue
                        
                        # Calculate relevance score
                        score = 0
                        text_to_search = f"{entry.get('title', '')} {entry.get('content_summary', '')} {' '.join(entry.get('tags', []))}"
                        text_lower = text_to_search.lower()
                        
                        # Simple keyword matching
                        query_words = query_lower.split()
                        for word in query_words:
                            if word in text_lower:
                                score += 1
                        
                        if score > 0:
                            results.append({
                                'title': entry.get('title', 'Unknown'),
                                'url': entry.get('url', ''),
                                'snippet': entry.get('content_summary', ''),
                                'source': entry.get('domain', 'unknown'),
                                'similarity': score / len(query_words),  # Normalized score
                                'tags': ','.join(entry.get('tags', [])),
                                'section_path': ' > '.join(entry.get('section_path', [])),
                                'content': entry.get('content_summary', '')
                            })
                        
                        if len(results) >= max_results * 2:  # Get more than needed for sorting
                            break
                            
                    except json.JSONDecodeError:
                        continue
            
            # Sort by similarity score and return top results
            results.sort(key=lambda x: x['similarity'], reverse=True)
            return results[:max_results]
            
        except Exception as e:
            logging.error(f"Fallback text search failed: {e}")
            return []
    
    def parse_instruction(self, instruction: str) -> Dict:
        """Parse user instruction and extract key information"""
        instruction_lower = instruction.lower()
        
        # Identify instruction type
        instruction_type = None
        for category, patterns in self.instruction_patterns.items():
            for pattern in patterns:
                if re.search(pattern, instruction_lower):
                    instruction_type = category
                    break
            if instruction_type:
                break
        
        # Extract object types
        object_types = ['cube', 'sphere', 'plane', 'cylinder', 'cone', 'torus', 'monkey']
        detected_objects = []
        for obj_type in object_types:
            if obj_type.lower() in instruction_lower:
                detected_objects.append(obj_type)
        
        # Extract position information
        position_match = re.search(r'location.*?\(([^)]+)\)|position.*?\(([^)]+)\)', instruction_lower)
        position = None
        if position_match:
            pos_str = position_match.group(1) or position_match.group(2)
            try:
                # Try to parse coordinates
                coords = [float(x.strip()) for x in pos_str.split(',')]
                if len(coords) >= 3:
                    position = tuple(coords[:3])
            except:
                position = None
        
        # Extract physics-related parameters
        physics_params = {}
        if re.search(r'gravity', instruction_lower):
            gravity_match = re.search(r'gravity.*?(\d+\.?\d*)', instruction_lower)
            if gravity_match:
                physics_params['gravity'] = float(gravity_match.group(1))
        
        if re.search(r'mass', instruction_lower):
            mass_match = re.search(r'mass.*?(\d+\.?\d*)', instruction_lower)
            if mass_match:
                physics_params['mass'] = float(mass_match.group(1))
        
        return {
            'instruction_type': instruction_type,
            'objects': detected_objects,
            'position': position,
            'physics_params': physics_params,
            'original_instruction': instruction
        }
    
    def search_knowledge_base(self, parsed_instruction: Dict) -> List[Dict]:
        """Search for relevant information in knowledge base using vector search"""
        instruction = parsed_instruction.get('original_instruction', '')
        if not instruction:
            return []
        
        # Use vector search to find relevant knowledge - get more results for better deduplication
        vector_results = self.search_vector_knowledge(instruction, max_results=40)
        logging.info(f"Vector search returned {len(vector_results)} results")
        
        # Convert vector results to API-like format for compatibility
        api_results = []
        for result in vector_results:
            # Extract potential API information from the content
            content = result.get('content', '')
            title = result.get('title', '')
            
            # Try to find API names and descriptions in the content
            api_matches = re.findall(r'(bpy\.\w+(?:\.\w+)*)', content)
            if api_matches:
                for api_name in set(api_matches):  # Remove duplicates
                    api_results.append({
                        'name': api_name,
                        'description': content[:100],
                        'example': '',
                        'use_case': result.get('tags', ''),
                        'source': result.get('source', ''),
                        'similarity': result.get('similarity', 0),
                        'url': result.get('url', '')
                    })
            else:
                # If no specific APIs found, create a general entry
                api_results.append({
                    'name': title,
                    'description': content,
                    'example': '',
                    'use_case': result.get('tags', ''),
                    'source': result.get('source', ''),
                    'similarity': result.get('similarity', 0),
                    'url': result.get('url', '')
                })
        
        logging.info(f"Created {len(api_results)} API results from vector search")
        
        # Remove duplicates based on name and description similarity
        deduplicated_results = self._deduplicate_results(api_results)
        logging.info(f"After deduplication: {len(deduplicated_results)} results")
        
        # Sort by similarity and return top 5
        deduplicated_results.sort(key=lambda x: x.get('similarity', 0), reverse=True)
        return deduplicated_results[:5]
    
    def _deduplicate_results(self, results: List[Dict]) -> List[Dict]:
        """Remove duplicate results based on exact name matches only"""
        if not results:
            return results
        
        # Simple deduplication: only remove exact name matches, keep the one with highest similarity
        seen_names = {}
        
        for result in results:
            name = result.get('name', '').strip()
            
            # Only deduplicate exact matches (case-insensitive)
            if name.lower() in seen_names:
                # Keep the one with higher similarity
                if result.get('similarity', 0) > seen_names[name.lower()].get('similarity', 0):
                    seen_names[name.lower()] = result
            else:
                seen_names[name.lower()] = result
        
        # Return all unique results
        return list(seen_names.values())
    
    def rag_query(self, instruction: str) -> Dict:
        """Main RAG query function using vector search"""
        try:
            parsed_instruction = self.parse_instruction(instruction)
            relevant_apis = self.search_knowledge_base(parsed_instruction)
            return {'status': 'success', 'output': relevant_apis}
            
        except Exception as e:
            logging.error(f"RAG query failed: {e}")
            return {'status': 'error', 'output': str(e)}

# Global instance
_rag_tool = None

# MCP tool interface
from mcp.server.fastmcp import FastMCP
mcp = FastMCP("rag-tool")

@mcp.tool()
def rag_query(instruction: str) -> Dict:
    """RAG query interface"""
    global _rag_tool
    if _rag_tool is None:
        _rag_tool = BlenderInfinigenRAG()
    
    return _rag_tool.rag_query(instruction)

@mcp.tool()
def initialize(args: dict) -> dict:
    """
    Initialize RAG tool with vector database
    
    Args:
        openai_api_key: OpenAI API key (optional)
        knowledge_file: Path to knowledge.jsonl file (optional)
        
    Returns:
        dict: Initialization result
    """
    try:
        global _rag_tool
        _rag_tool = BlenderInfinigenRAG(
            openai_api_key=args.get("openai_api_key"),
            knowledge_file=args.get("knowledge_file", "tools/knowledge_base/rag_kb_deduplicated.jsonl")
        )
        return {"status": "success", "output": "RAG tool initialized successfully with vector database"}
    except Exception as e:
        return {"status": "error", "output": str(e)}


def main():
    """Run MCP server or test the RAG tool when --test is provided"""
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        # Ensure tool is initialized (optionally with OPENAI_API_KEY env)
        global _rag_tool
        if _rag_tool is None:
            _rag_tool = BlenderInfinigenRAG(os.getenv("OPENAI_API_KEY"))

        # Test 1: Initialize with vector database
        print("[test:initialize] Initializing RAG tool with vector database...")
        init_args = {
            "openai_api_key": os.getenv("OPENAI_API_KEY"),
            "knowledge_file": "tools/knowledge_base/rag_kb_deduplicated.jsonl"
        }
        init_result = initialize(init_args)
        print("[test:initialize]", json.dumps(init_result, ensure_ascii=False, indent=2))

        # Test 2: RAG query
        result = rag_query("Create a living room with table and chair")
        print("[test:rag_query]", json.dumps(result, ensure_ascii=False, indent=2))
        with open('logs/rag.log', 'w') as f:
            f.write(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        # Default: run as MCP server
        mcp.run(transport="stdio")

if __name__ == "__main__":
    main()
