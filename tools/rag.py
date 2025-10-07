# rag.py
import os
import re
import json
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import requests
from bs4 import BeautifulSoup

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None


class BlenderInfinigenRAG:
    """RAG tool: Search for bpy and Infinigen related documentation and generate code examples"""
    
    def __init__(self, openai_api_key: str = None):
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if self.openai_api_key and OPENAI_AVAILABLE:
            self.openai_client = OpenAI(api_key=self.openai_api_key)
        else:
            self.openai_client = None
            if not OPENAI_AVAILABLE:
                logging.warning("OpenAI library not available. Enhanced generation disabled.")
            elif not self.openai_api_key:
                logging.warning("OpenAI API key not provided. Enhanced generation disabled.")
        
        # Predefined documentation knowledge base
        self.knowledge_base = self._build_knowledge_base()
        
        # Instruction pattern matching
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
        # Authoritative seed URLs for stable crawling/extraction
        self.seed_urls = {
            'blender_docs': [
                'https://docs.blender.org/api/current/index.html',
                'https://docs.blender.org/api/current/info_overview.html',
                'https://docs.blender.org/api/current/info_api_reference.html',
                'https://docs.blender.org/api/current/bpy.ops.html',
                'https://docs.blender.org/api/2.82/index.html'
            ],
            'infinigen_docs': [
                'https://infinigen.org/',
                'https://infinigen.org/docs/installation/intro',
                'https://github.com/princeton-vl/infinigen',
                'https://arxiv.org/abs/2406.11824',
                'https://infinigen.org/docs-contributing/begin'
            ]
        }
    
    def _build_knowledge_base(self) -> Dict:
        """Build knowledge base for bpy and Infinigen (expanded, curated)."""
        return {
            'bpy_overview': {
                'title': 'Blender Python API Overview',
                'apis': [
                    {
                        'name': 'bpy',
                        'description': 'Top-level Blender Python module exposing data (bpy.data), context (bpy.context), and operators (bpy.ops).',
                        'example': 'import bpy\nprint(bpy.app.version)',
                        'use_case': 'General scripting entry-point'
                    },
                    {
                        'name': 'bpy.context',
                        'description': 'Access the current context: active object, scene, view layer, etc.',
                        'example': 'obj = bpy.context.active_object\nscene = bpy.context.scene',
                        'use_case': 'Read/write the current scene state'
                    },
                    {
                        'name': 'bpy.data',
                        'description': 'Access datablocks (objects, meshes, materials, images, texts).',
                        'example': "for obj in bpy.data.objects: print(obj.name)",
                        'use_case': 'Enumerate or create data blocks (assets)'
                    }
                ]
            },
            'bpy_ops_object': {
                'title': 'bpy.ops.object — Object operators',
                'apis': [
                    {
                        'name': 'bpy.ops.object.select_all',
                        'description': 'Select or deselect all objects.',
                        'example': "bpy.ops.object.select_all(action='SELECT')",
                        'use_case': 'Bulk selection control'
                    },
                    {
                        'name': 'bpy.ops.object.delete',
                        'description': 'Delete selected objects from the scene.',
                        'example': 'bpy.ops.object.delete(use_global=False)',
                        'use_case': 'Remove unwanted objects'
                    },
                    {
                        'name': 'bpy.ops.object.transform_apply',
                        'description': 'Apply location/rotation/scale transforms to selected objects.',
                        'example': "bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)",
                        'use_case': 'Freeze transforms before export/physics'
                    },
                    {
                        'name': 'bpy.ops.object.camera_add',
                        'description': 'Add a camera to the scene at a given location/rotation.',
                        'example': 'bpy.ops.object.camera_add(location=(3, -2, 2), rotation=(1.2, 0, 0.9))',
                        'use_case': 'Introduce a render camera'
                    },
                    {
                        'name': 'bpy.ops.object.light_add',
                        'description': 'Add a light object (POINT/SUN/SPOT/AREA).',
                        'example': "bpy.ops.object.light_add(type='AREA', location=(0,0,3))",
                        'use_case': 'Lighting setup'
                    }
                ]
            },
            'bpy_ops_mesh': {
                'title': 'bpy.ops.mesh — Mesh primitives',
                'apis': [
                    {
                        'name': 'bpy.ops.mesh.primitive_plane_add',
                        'description': 'Create a plane primitive at a location/size.',
                        'example': 'bpy.ops.mesh.primitive_plane_add(size=10, location=(0, 0, 0))',
                        'use_case': 'Floor or ground plane'
                    },
                    {
                        'name': 'bpy.ops.mesh.primitive_cube_add',
                        'description': 'Create a cube primitive.',
                        'example': 'bpy.ops.mesh.primitive_cube_add(size=2, location=(0, 0, 1))',
                        'use_case': 'Basic box geometry'
                    },
                    {
                        'name': 'bpy.ops.mesh.primitive_uv_sphere_add',
                        'description': 'Create a UV sphere primitive.',
                        'example': 'bpy.ops.mesh.primitive_uv_sphere_add(radius=1, location=(0, 0, 1))',
                        'use_case': 'Spherical geometry'
                    },
                    {
                        'name': 'bpy.ops.mesh.primitive_cylinder_add',
                        'description': 'Create a cylinder primitive.',
                        'example': 'bpy.ops.mesh.primitive_cylinder_add(radius=0.5, depth=2, location=(0,0,1))',
                        'use_case': 'Cylindrical geometry'
                    },
                    {
                        'name': 'bpy.ops.mesh.primitive_cone_add',
                        'description': 'Create a cone primitive.',
                        'example': 'bpy.ops.mesh.primitive_cone_add(radius1=1, depth=2, location=(0,0,1))',
                        'use_case': 'Conical geometry'
                    }
                ]
            },
            'bpy_import_export': {
                'title': 'Import/Export formats',
                'apis': [
                    {
                        'name': 'bpy.ops.import_scene.gltf',
                        'description': 'Import a GLTF/GLB file.',
                        'example': "bpy.ops.import_scene.gltf(filepath='/path/to/model.glb', import_animations=True)",
                        'use_case': 'Bring external .glb assets into the scene'
                    },
                    {
                        'name': 'bpy.ops.wm.obj_import',
                        'description': 'Import a Wavefront OBJ (may require io_scene_obj addon).',
                        'example': "bpy.ops.wm.obj_import(filepath='/path/to/model.obj')",
                        'use_case': 'OBJ asset import'
                    },
                    {
                        'name': 'bpy.ops.export_scene.gltf',
                        'description': 'Export scene/selection to GLTF/GLB.',
                        'example': "bpy.ops.export_scene.gltf(filepath='/tmp/out.glb', export_format='GLB')",
                        'use_case': 'Export for downstream engines'
                    }
                ]
            },
            'bpy_types_object_camera_light': {
                'title': 'bpy.types — Object/Camera/Light essentials',
                'apis': [
                    {
                        'name': 'bpy.types.Object',
                        'description': 'Core data-block representing an object; holds transform and data.',
                        'example': "obj = bpy.data.objects['Cube']; obj.location = (1,0,0)",
                        'use_case': 'Transform, parenting, visibility'
                    },
                    {
                        'name': 'bpy.types.Camera',
                        'description': 'Camera data settings (lens, DOF, sensor).',
                        'example': "cam = bpy.data.cameras.new('Cam'); cam.lens = 35",
                        'use_case': 'Configure camera properties'
                    },
                    {
                        'name': 'bpy.types.Light',
                        'description': 'Light data settings (type, energy, color).',
                        'example': "light = bpy.data.lights.new('Key','AREA'); light.energy = 1000",
                        'use_case': 'Lighting configuration'
                    }
                ]
            },
            'bpy_materials_render': {
                'title': 'Materials, Nodes, and Rendering',
                'apis': [
                    {
                        'name': 'bpy.data.materials.new',
                        'description': 'Create a new material and enable nodes.',
                        'example': "mat = bpy.data.materials.new('Mat'); mat.use_nodes = True",
                        'use_case': 'Assign and edit materials'
                    },
                    {
                        'name': 'Cycles/Eevee settings',
                        'description': 'Set render engine and samples.',
                        'example': "scene = bpy.context.scene; scene.render.engine='CYCLES'; scene.cycles.samples=64",
                        'use_case': 'Control render quality and engine'
                    },
                    {
                        'name': 'bpy.ops.render.render',
                        'description': 'Render the current frame to file if write_still is True.',
                        'example': "scene.render.filepath = '/tmp/r.png'\nbpy.ops.render.render(write_still=True)",
                        'use_case': 'Produce images for verification'
                    }
                ]
            },
            'bpy_physics': {
                'title': 'Rigidbody / Physics',
                'apis': [
                    {
                        'name': 'bpy.ops.rigidbody.world_add',
                        'description': 'Create rigid body world settings on the scene.',
                        'example': 'bpy.ops.rigidbody.world_add()',
                        'use_case': 'Enable physics simulation'
                    },
                    {
                        'name': 'bpy.ops.rigidbody.object_add',
                        'description': 'Add rigid body to the active object (ACTIVE/PASSIVE).',
                        'example': "bpy.ops.rigidbody.object_add(type='ACTIVE')",
                        'use_case': 'Make object simulate physics'
                    },
                    {
                        'name': 'bpy.context.scene.rigidbody_world.gravity',
                        'description': 'Adjust scene gravity vector.',
                        'example': 'bpy.context.scene.rigidbody_world.gravity = (0, 0, -9.81)',
                        'use_case': 'Tune gravity for dynamics'
                    }
                ]
            },
            'infinigen_overview': {
                'title': 'Infinigen Overview',
                'apis': [
                    {
                        'name': 'infinigen.core.scene',
                        'description': 'Scene generation and asset composition utilities.',
                        'example': 'scene.add_objects(objects, placement_strategy="physics")',
                        'use_case': 'Assemble procedural scenes'
                    },
                    {
                        'name': 'infinigen.core.lighting',
                        'description': 'Lighting utilities for natural/indoor lighting setups.',
                        'example': 'lighting.setup_natural_lighting(scene)',
                        'use_case': 'Quick realistic lighting'
                    }
                ]
            },
            'infinigen_placement_physics': {
                'title': 'Infinigen Placement and Physics',
                'apis': [
                    {
                        'name': 'infinigen.core.placement.placement',
                        'description': 'Constraint-aware placement system for objects.',
                        'example': 'placement.place_object(obj, surface, physics=True)',
                        'use_case': 'Physically plausible object placement'
                    },
                    {
                        'name': 'infinigen.core.placement.surface',
                        'description': 'Surface utilities for finding placement points.',
                        'example': 'surface.find_surface_point(location, radius)',
                        'use_case': 'Place on detected surfaces'
                    },
                    {
                        'name': 'infinigen.core.physics.rigidbody',
                        'description': 'Helpers for rigid body configuration.',
                        'example': 'rigidbody.setup_rigidbody(obj, mass=1.0)',
                        'use_case': 'Configure dynamics properties'
                    }
                ]
            },
            'infinigen_assets_layout': {
                'title': 'Infinigen Assets and Layout',
                'apis': [
                    {
                        'name': 'infinigen.core.assets',
                        'description': 'Asset library and generation helpers (varies by version).',
                        'example': 'assets.load_asset("chair_basic")',
                        'use_case': 'Fetch or generate assets to populate scenes'
                    },
                    {
                        'name': 'infinigen.core.layout',
                        'description': 'Room/indoor layout utilities for composition.',
                        'example': 'layout.build_room(width=4, depth=3, height=2.8)',
                        'use_case': 'Create indoor envelopes and furniture placement'
                    }
                ]
            }
        }
    
    def search_blender_docs(self, query: str, max_results: int = 5) -> List[Dict]:
        """Search Blender Python API documentation for relevant information"""
        try:
            # Search Blender API documentation
            blender_search_url = f"https://docs.blender.org/api/current/search.html?q={query}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(blender_search_url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            results = []
            
            # Strategy 1: parse standard result containers
            containers = []
            containers.extend(soup.find_all('div', class_='search-results'))
            containers.extend(soup.find_all('div', class_='result'))
            for result in containers:
                title_elem = result.find('a') or result.find('h3')
                snippet_elem = result.find('p') or result.find('div', class_='highlight')
                if title_elem:
                    title = (title_elem.get_text() or '').strip()
                    url = title_elem.get('href', '') or ''
                    if url and not url.startswith('http'):
                        url = f"https://docs.blender.org/api/current/{url.lstrip('./')}"
                    snippet = (snippet_elem.get_text() if snippet_elem else '') or ''
                    if title or url:
                        results.append({'title': title, 'url': url, 'snippet': snippet, 'source': 'blender_docs'})

            # Strategy 2: if empty, probe common sections by keyword mapping
            if not results:
                keyword_map = self._find_relevant_blender_sections(query)
                if keyword_map:
                    results.extend(keyword_map)

            # Strategy 3: if still empty, add baseline useful entries
            if not results:
                baseline = [
                    {'title': 'Blender Python API - Operators', 'url': 'https://docs.blender.org/api/current/bpy.ops.html', 'snippet': 'Overview of bpy.ops operator modules', 'source': 'blender_docs'},
                    {'title': 'Blender Python API - Types', 'url': 'https://docs.blender.org/api/current/bpy.types.html', 'snippet': 'API for core bpy.types classes', 'source': 'blender_docs'},
                    {'title': 'Object Operators', 'url': 'https://docs.blender.org/api/current/bpy.ops.object.html', 'snippet': 'Manipulate objects (add, transform, etc.)', 'source': 'blender_docs'},
                ]
                results.extend(baseline[:max_results])
            
            # If no specific search results, try to find relevant sections
            if len(results) < max_results:
                relevant_sections = self._find_relevant_blender_sections(query)
                # Avoid duplicates by URL
                seen = {r['url'] for r in results if r.get('url')}
                for sec in relevant_sections:
                    if sec.get('url') not in seen:
                        results.append(sec)
                        seen.add(sec.get('url'))
                    if len(results) >= max_results:
                        break
            
            return results
            
        except Exception as e:
            logging.error(f"Blender docs search failed: {e}")
            # Hard fallback to baseline links
            return [
                {'title': 'Blender Python API - Index', 'url': 'https://docs.blender.org/api/current/index.html', 'snippet': 'Main index for Blender Python API', 'source': 'blender_docs'}
            ]

    def search_infinigen_docs(self, query: str, max_results: int = 5) -> List[Dict]:
        """Search Infinigen documentation for relevant information"""
        try:
            # Search Infinigen GitHub docs
            infinigen_search_url = f"https://github.com/princeton-vl/infinigen/tree/main/docs"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(infinigen_search_url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            results = []
            
            # Strategy 1: parse directory listing links
            doc_links = soup.find_all('a', href=True)
            for link in doc_links:
                href = link.get('href', '') or ''
                title = (link.get_text() or '').strip()
                if '/docs/' in href and (href.endswith('.md') or href.endswith('/')):
                    score = 0
                    q = query.lower()
                    t = title.lower()
                    for kw in ['physics','scene','lighting','placement','rigid','object','mesh','camera','animation']:
                        if kw in q and kw in t:
                            score += 1
                    if score > 0 or not results:
                        results.append({
                            'title': f"Infinigen: {title or href.split('/')[-1]}",
                            'url': f"https://github.com{href}",
                            'snippet': f"Documentation section: {title or href}",
                            'source': 'infinigen_docs'
                        })
            
            # Strategy 2: curated fallbacks by keyword
            if not results:
                curated = [
                    {'title': 'Infinigen README', 'url': 'https://github.com/princeton-vl/infinigen/blob/main/README.md', 'snippet': 'Project overview and docs entry points', 'source': 'infinigen_docs'},
                    {'title': 'Infinigen Docs Index', 'url': 'https://github.com/princeton-vl/infinigen/tree/main/docs', 'snippet': 'Browse Infinigen documentation', 'source': 'infinigen_docs'},
                ]
                results.extend(curated)

            return results[:max_results]
            
        except Exception as e:
            logging.error(f"Infinigen docs search failed: {e}")
            return [
                {'title': 'Infinigen Docs Index', 'url': 'https://github.com/princeton-vl/infinigen/tree/main/docs', 'snippet': 'Browse Infinigen documentation', 'source': 'infinigen_docs'}
            ]

    def _find_relevant_blender_sections(self, query: str) -> List[Dict]:
        """Find relevant Blender API sections based on query keywords"""
        query_lower = query.lower()
        relevant_sections = []
        
        # Map query keywords to relevant API sections
        section_mapping = {
            'physics': [
                {'title': 'Rigidbody Operators', 'url': 'https://docs.blender.org/api/current/bpy.ops.rigidbody.html', 'snippet': 'Physics simulation operators for rigid body dynamics'},
                {'title': 'Physics World', 'url': 'https://docs.blender.org/api/current/bpy.types.RigidBodyWorld.html', 'snippet': 'Rigid body world settings and properties'}
            ],
            'mesh': [
                {'title': 'Mesh Operators', 'url': 'https://docs.blender.org/api/current/bpy.ops.mesh.html', 'snippet': 'Mesh creation and manipulation operators'},
                {'title': 'Mesh Types', 'url': 'https://docs.blender.org/api/current/bpy.types.Mesh.html', 'snippet': 'Mesh data structure and properties'}
            ],
            'object': [
                {'title': 'Object Operators', 'url': 'https://docs.blender.org/api/current/bpy.ops.object.html', 'snippet': 'Object manipulation and creation operators'},
                {'title': 'Object Types', 'url': 'https://docs.blender.org/api/current/bpy.types.Object.html', 'snippet': 'Object data structure and properties'}
            ],
            'light': [
                {'title': 'Light Operators', 'url': 'https://docs.blender.org/api/current/bpy.ops.object.html#bpy.ops.object.light_add', 'snippet': 'Light creation and manipulation'},
                {'title': 'Light Types', 'url': 'https://docs.blender.org/api/current/bpy.types.Light.html', 'snippet': 'Light data structure and properties'}
            ],
            'material': [
                {'title': 'Material Operators', 'url': 'https://docs.blender.org/api/current/bpy.ops.material.html', 'snippet': 'Material creation and manipulation operators'},
                {'title': 'Material Types', 'url': 'https://docs.blender.org/api/current/bpy.types.Material.html', 'snippet': 'Material data structure and properties'}
            ],
            'camera': [
                {'title': 'Camera Operators', 'url': 'https://docs.blender.org/api/current/bpy.ops.object.html#bpy.ops.object.camera_add', 'snippet': 'Camera creation and manipulation'},
                {'title': 'Camera Types', 'url': 'https://docs.blender.org/api/current/bpy.types.Camera.html', 'snippet': 'Camera data structure and properties'}
            ],
            'scene': [
                {'title': 'Scene Operators', 'url': 'https://docs.blender.org/api/current/bpy.ops.scene.html', 'snippet': 'Scene management operators'},
                {'title': 'Scene Types', 'url': 'https://docs.blender.org/api/current/bpy.types.Scene.html', 'snippet': 'Scene data structure and properties'}
            ]
        }
        
        for keyword, sections in section_mapping.items():
            if keyword in query_lower:
                relevant_sections.extend(sections)
        
        return relevant_sections

    def _is_relevant_to_query(self, title: str, query: str) -> bool:
        """Check if a documentation title is relevant to the query"""
        query_lower = query.lower()
        title_lower = title.lower()
        
        # Keywords that indicate relevance
        relevant_keywords = ['physics', 'mesh', 'object', 'light', 'material', 'camera', 'scene', 'placement', 'generation']
        
        for keyword in relevant_keywords:
            if keyword in query_lower and keyword in title_lower:
                return True
        
        return False

    def search_documentation(self, query: str, max_results: int = 5) -> List[Dict]:
        """Search both Blender and Infinigen documentation"""
        results = []
        
        # Search Blender documentation
        blender_results = self.search_blender_docs(query, max_results // 2 or 1)
        results.extend(blender_results)
        
        # Search Infinigen documentation
        infinigen_results = self.search_infinigen_docs(query, max_results - len(results))
        results.extend(infinigen_results)
        
        # As a last resort, ensure at least one high-signal link per source
        if not results:
            results = [
                {'title': 'Blender Python API - Index', 'url': 'https://docs.blender.org/api/current/index.html', 'snippet': 'Main index for Blender Python API', 'source': 'blender_docs'},
                {'title': 'Infinigen Docs', 'url': 'https://github.com/princeton-vl/infinigen/tree/main/docs', 'snippet': 'Infinigen documentation index', 'source': 'infinigen_docs'},
            ]
        return results[:max_results]

    # ---------------------------
    # Extraction utilities
    # ---------------------------
    def _fetch(self, url: str, timeout: int = 10) -> Optional[str]:
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36'
            }
            r = requests.get(url, headers=headers, timeout=timeout)
            if r.status_code == 200:
                return r.text
        except Exception as e:
            logging.warning(f"fetch failed for {url}: {e}")
        return None

    def _extract_blender_excerpt(self, html: str, query: str, max_len: int = 800) -> str:
        try:
            soup = BeautifulSoup(html, 'html.parser')
            # Remove script/style
            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()
            # Main content area often under role="main" or div.document
            main = soup.find(attrs={'role': 'main'}) or soup.find('div', class_='document') or soup
            text_parts = []
            # Prefer API signatures and their descriptions
            # Blender docs often use dl > dt (signature) + dd (description)
            for dl in main.find_all('dl'):
                for dt in dl.find_all('dt'):
                    sig = dt.get_text(" ", strip=True)
                    if not sig:
                        continue
                    lower = sig.lower()
                    if any(k for k in re.split(r"[^a-z0-9_]+", query.lower()) if k and k in lower):
                        # Find following dd
                        dd = dt.find_next_sibling('dd')
                        desc = dd.get_text(" ", strip=True) if dd else ""
                        text_parts.append(sig)
                        if desc:
                            text_parts.append(desc)
                        if len("\n".join(text_parts)) >= max_len:
                            break
                if len("\n".join(text_parts)) >= max_len:
                    break

            # If still not found, fall back to keyword-based paragraphs/pre/code
            keywords = [k for k in set(re.split(r"[^a-zA-Z0-9_]+", query.lower())) if k]
            # Collect matching headings and paragraphs
            for el in main.find_all(['dt','dd','p','pre','code','li','h1','h2','h3']):
                t = el.get_text(" ", strip=True)
                if not t:
                    continue
                lower = t.lower()
                if any(k in lower for k in keywords):
                    text_parts.append(t)
                if len("\n".join(text_parts)) > max_len:
                    break
            if not text_parts:
                # Fallback: take first paragraphs/code
                for el in main.find_all(['p','pre','code'])[:6]:
                    t = el.get_text(" ", strip=True)
                    if t:
                        text_parts.append(t)
                        if len("\n".join(text_parts)) > max_len:
                            break
            excerpt = "\n".join(text_parts)
            # Normalize whitespace to ensure plain text
            excerpt = re.sub(r"\s+", " ", excerpt)
            return excerpt[:max_len]
        except Exception as e:
            logging.warning(f"_extract_blender_excerpt failed: {e}")
            return ""

    def _extract_markdown_excerpt(self, md_text: str, query: str, window: int = 3, max_blocks: int = 6) -> str:
        lines = md_text.splitlines()
        q = query.lower()
        indices = []
        for i, line in enumerate(lines):
            if any(tok and tok in line.lower() for tok in re.split(r"[^a-zA-Z0-9_]+", q)):
                indices.append(i)
        blocks = []
        used = set()
        for idx in indices:
            start = max(0, idx - window)
            end = min(len(lines), idx + window + 1)
            if (start, end) in used:
                continue
            used.add((start, end))
            blocks.append("\n".join(lines[start:end]))
            if len(blocks) >= max_blocks:
                break
        if not blocks:
            # fallback to head of file
            blocks.append("\n".join(lines[:min(40, len(lines))]))
        # Clean markdown artifacts lightly (keep code blocks textually)
        text = "\n---\n".join(blocks)
        text = re.sub(r"\s+", " ", text)
        return text

    def _maybe_to_raw_github(self, url: str) -> str:
        # Convert GitHub blob URL to raw
        # e.g., https://github.com/org/repo/blob/branch/path.md -> https://raw.githubusercontent.com/org/repo/branch/path.md
        m = re.match(r"https://github\.com/([^/]+)/([^/]+)/blob/([^/]+)/(.*)", url)
        if m:
            owner, repo, branch, path = m.groups()
            return f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}"
        return url

    def _enrich_doc_results(self, doc_results: List[Dict], query: str, per_source_limit: int = 3) -> List[Dict]:
        enriched = []
        blender_count = 0
        infi_count = 0
        for item in doc_results:
            url = item.get('url')
            source = item.get('source')
            if not url:
                enriched.append(item)
                continue
            excerpt = ''
            if source == 'blender_docs' and blender_count < per_source_limit:
                html = self._fetch(url)
                if html:
                    excerpt = self._extract_blender_excerpt(html, query)
                blender_count += 1
            elif source == 'infinigen_docs' and infi_count < per_source_limit:
                fetch_url = self._maybe_to_raw_github(url)
                text = self._fetch(fetch_url)
                if text:
                    excerpt = self._extract_markdown_excerpt(text, query)
                infi_count += 1
            enriched.append({**item, 'excerpt': excerpt})
        return enriched

    def search_seeded_docs(self, query: str, max_results: int = 6) -> List[Dict]:
        """Search authoritative seeds and extract relevant plain text excerpts."""
        results: List[Dict] = []
        # Blender seeds
        for url in self.seed_urls.get('blender_docs', []):
            html = self._fetch(url)
            if not html:
                continue
            excerpt = self._extract_blender_excerpt(html, query)
            if excerpt:
                results.append({'title': 'Blender Docs', 'url': url, 'snippet': excerpt[:160], 'source': 'blender_docs', 'excerpt': excerpt})
            if len(results) >= max_results:
                break
        # Infinigen seeds
        if len(results) < max_results:
            for url in self.seed_urls.get('infinigen_docs', []):
                fetch_url = self._maybe_to_raw_github(url)
                text = self._fetch(fetch_url)
                if not text:
                    continue
                # Choose extraction based on content type
                if 'githubusercontent' in fetch_url or fetch_url.endswith('.md'):
                    excerpt = self._extract_markdown_excerpt(text, query)
                else:
                    excerpt = self._extract_blender_excerpt(text, query)  # reuse HTML extractor for generic pages
                if excerpt:
                    results.append({'title': 'Infinigen Docs', 'url': url, 'snippet': excerpt[:160], 'source': 'infinigen_docs', 'excerpt': excerpt})
                if len(results) >= max_results:
                    break
        return results[:max_results]
    
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
        """Search for relevant information in knowledge base"""
        results = []
        instruction_type = parsed_instruction.get('instruction_type')
        
        # Search relevant knowledge based on instruction type
        if instruction_type == 'physics_placement':
            results.extend(self.knowledge_base['bpy_physics']['apis'])
            results.extend(self.knowledge_base['infinigen_physics']['apis'])
        
        if instruction_type == 'object_creation':
            results.extend(self.knowledge_base['bpy_objects']['apis'])
        
        if instruction_type == 'scene_setup':
            results.extend(self.knowledge_base['infinigen_scene']['apis'])
        
        # If no specific type matches, search all relevant APIs
        if not results:
            for category in self.knowledge_base.values():
                results.extend(category['apis'])

        # Rank and filter top-5 most relevant to the original instruction
        try:
            instruction = (parsed_instruction.get('original_instruction') or '').lower()
            tokens = [t for t in re.split(r"[^a-z0-9_]+", instruction) if t]

            def score_api(api: Dict) -> float:
                name = (api.get('name') or '').lower()
                desc = (api.get('description') or '').lower()
                use_case = (api.get('use_case') or '').lower()
                example = (api.get('example') or '').lower()
                score = 0.0
                for tok in tokens:
                    # weight matches: name 3x, desc 2x, use_case 1.5x, example 1x
                    if tok in name:
                        score += 3.0
                    if tok in desc:
                        score += 2.0
                    if tok in use_case:
                        score += 1.5
                    if tok in example:
                        score += 1.0
                return score

            scored = [(score_api(api), api) for api in results]
            # If all zero, keep original order but cap 5
            if any(s > 0 for s, _ in scored):
                scored.sort(key=lambda x: x[0], reverse=True)
                filtered = [api for _, api in scored[:5]]
            else:
                filtered = results[:5]
            return filtered
        except Exception:
            # Fallback: return up to 5 without ranking on any unexpected error
            return results[:5]
    
    def generate_code_example(self, parsed_instruction: Dict, relevant_apis: List[Dict]) -> str:
        """Generate code example"""
        instruction_type = parsed_instruction.get('instruction_type')
        objects = parsed_instruction.get('objects', ['cube'])
        position = parsed_instruction.get('position', (0, 0, 0))
        physics_params = parsed_instruction.get('physics_params', {})
        
        code_lines = ["import bpy", "import bmesh", ""]
        
        # Generate corresponding code based on instruction type
        if instruction_type == 'physics_placement':
            code_lines.extend([
                "# Setup physics world",
                "bpy.ops.rigidbody.world_add()",
                ""
            ])
            
            # Set gravity
            gravity = physics_params.get('gravity', -9.81)
            code_lines.extend([
                "# Set gravity",
                f"bpy.context.scene.rigidbody_world.gravity = (0, 0, {gravity})",
                ""
            ])
            
            # Create objects
            for obj_name in objects:
                if 'cube' in obj_name.lower():
                    code_lines.extend([
                        "# Create cube",
                        f"bpy.ops.mesh.primitive_cube_add(size=2, location={position})",
                        ""
                    ])
                elif 'sphere' in obj_name.lower():
                    code_lines.extend([
                        "# Create sphere",
                        f"bpy.ops.mesh.primitive_uv_sphere_add(radius=1, location={position})",
                        ""
                    ])
                elif 'plane' in obj_name.lower():
                    code_lines.extend([
                        "# Create plane",
                        f"bpy.ops.mesh.primitive_plane_add(size=10, location={position})",
                        ""
                    ])
            
            # Add rigid body physics
            mass = physics_params.get('mass', 1.0)
            code_lines.extend([
                "# Add rigid body physics to object",
                f"bpy.ops.rigidbody.object_add(type='ACTIVE')",
                f"bpy.context.object.rigid_body.mass = {mass}",
                ""
            ])
            
        elif instruction_type == 'object_creation':
            for obj_name in objects:
                if 'cube' in obj_name.lower():
                    code_lines.extend([
                        "# Create cube",
                        f"bpy.ops.mesh.primitive_cube_add(size=2, location={position})",
                        ""
                    ])
                elif 'sphere' in obj_name.lower():
                    code_lines.extend([
                        "# Create sphere",
                        f"bpy.ops.mesh.primitive_uv_sphere_add(radius=1, location={position})",
                        ""
                    ])
        
        # Add comments explaining used APIs
        if relevant_apis:
            code_lines.extend([
                "# Related API documentation:",
            ])
            for api in relevant_apis[:3]:  # Show only first 3
                code_lines.append(f"# {api['name']}: {api['description']}")
        
        return "\n".join(code_lines)
    
    def enhanced_generation(self, instruction: str) -> str:
        """Use OpenAI for enhanced code generation (if available)"""
        if not self.openai_client:
            return "OpenAI API not available for enhanced generation"
        
        try:
            parsed = self.parse_instruction(instruction)
            relevant_apis = self.search_knowledge_base(parsed)
            
            # Search official documentation for additional context
            doc_results = self.search_documentation(instruction, max_results=5)
            doc_context = ""
            if doc_results:
                doc_context = "\n\nOfficial Documentation References:\n"
                for result in doc_results:
                    doc_context += f"- {result['title']} ({result['source']}): {result['snippet']}\n"
                    doc_context += f"  URL: {result['url']}\n"
            
            # Build prompt
            api_info = "\n".join([f"- {api['name']}: {api['description']}" for api in relevant_apis[:5]])
            
            prompt = f"""
Based on the following user instruction, relevant API information, and official documentation references, generate Blender Python code:

User instruction: {instruction}

Relevant APIs from knowledge base:
{api_info}
{doc_context}

Requirements:
1. Generate complete, executable Blender Python code
2. Include necessary import statements
3. Add detailed English comments
4. Ensure code follows Blender Python API best practices
5. Use the most current and accurate API methods from the official documentation
6. Reference the official Blender Python API documentation at https://docs.blender.org/api/current/index.html

Please return only the code, no additional explanations.
"""
            
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logging.error(f"Enhanced generation failed: {e}")
            return f"Enhanced generation failed: {str(e)}"
    
    def rag_query(self, instruction: str, use_enhanced: bool = False, use_doc_search: bool = True) -> Dict:
        """Main RAG query function"""
        try:
            # Parse instruction
            parsed_instruction = self.parse_instruction(instruction)
            
            # Search relevant knowledge
            relevant_apis = self.search_knowledge_base(parsed_instruction)
            
            # Search official documentation if requested
            doc_results = []
            if use_doc_search:
                # 1) query-time live search of official docs + enrichment
                raw_results = self.search_documentation(instruction, max_results=6)
                enriched = self._enrich_doc_results(raw_results, instruction)
                # 2) authoritative seeds fallback to ensure coverage
                seeded = self.search_seeded_docs(instruction, max_results=6)
                # Deduplicate by URL, merge keeping enriched excerpts
                seen = set()
                doc_results = []
                for item in enriched + seeded:
                    url = item.get('url')
                    if not url or url in seen:
                        continue
                    seen.add(url)
                    doc_results.append(item)
                    if len(doc_results) >= 6:
                        break
            
            # Generate code example
            if use_enhanced and self.openai_client:
                code_example = self.enhanced_generation(instruction)
            else:
                code_example = self.generate_code_example(parsed_instruction, relevant_apis)
            
            return {
                'status': 'success',
                'instruction': instruction,
                'parsed_instruction': parsed_instruction,
                'relevant_apis': relevant_apis,
                'doc_results': doc_results,
                'code_example': code_example,
                'generation_method': 'enhanced' if use_enhanced else 'template'
            }
            
        except Exception as e:
            logging.error(f"RAG query failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'instruction': instruction
            }


# Global instance
_rag_tool = None


def initialize_rag_tool(openai_api_key: str = None):
    """Initialize RAG tool"""
    global _rag_tool
    _rag_tool = BlenderInfinigenRAG(openai_api_key)
    return _rag_tool


def rag_query(instruction: str, use_enhanced: bool = False, use_doc_search: bool = True) -> Dict:
    """RAG query interface"""
    global _rag_tool
    if _rag_tool is None:
        _rag_tool = BlenderInfinigenRAG()
    
    return _rag_tool.rag_query(instruction, use_enhanced, use_doc_search)


# MCP tool interface
from mcp.server.fastmcp import FastMCP
mcp = FastMCP("rag-tool")

@mcp.tool()
def rag_query_tool(instruction: str, use_enhanced: bool = False, use_doc_search: bool = True) -> dict:
    """
    Search for bpy and Infinigen related documentation and generate code examples based on input instruction
    
    Args:
        instruction: User instruction, e.g., "place an object following physics laws"
        use_enhanced: Whether to use OpenAI enhanced generation (requires OpenAI API key)
        use_doc_search: Whether to search official Blender and Infinigen documentation
        
    Returns:
        dict: Dictionary containing parsed results, relevant APIs, documentation results, and generated code
    """
    return rag_query(instruction, use_enhanced, use_doc_search)

@mcp.tool()
def search_blender_docs_tool(query: str, max_results: int = 5) -> dict:
    """
    Search Blender Python API documentation for relevant information
    
    Args:
        query: Search query
        max_results: Maximum number of results to return
        
    Returns:
        dict: Blender documentation search results
    """
    try:
        global _rag_tool
        if _rag_tool is None:
            _rag_tool = BlenderInfinigenRAG()
        
        results = _rag_tool.search_blender_docs(query, max_results)
        return {
            "status": "success",
            "query": query,
            "results": results
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

@mcp.tool()
def search_infinigen_docs_tool(query: str, max_results: int = 5) -> dict:
    """
    Search Infinigen documentation for relevant information
    
    Args:
        query: Search query
        max_results: Maximum number of results to return
        
    Returns:
        dict: Infinigen documentation search results
    """
    try:
        global _rag_tool
        if _rag_tool is None:
            _rag_tool = BlenderInfinigenRAG()
        
        results = _rag_tool.search_infinigen_docs(query, max_results)
        return {
            "status": "success",
            "query": query,
            "results": results
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

@mcp.tool()
def search_documentation_tool(query: str, max_results: int = 5) -> dict:
    """
    Search both Blender and Infinigen documentation for relevant information
    
    Args:
        query: Search query
        max_results: Maximum number of results to return
        
    Returns:
        dict: Combined documentation search results from both sources
    """
    try:
        global _rag_tool
        if _rag_tool is None:
            _rag_tool = BlenderInfinigenRAG()
        
        results = _rag_tool.search_documentation(query, max_results)
        return {
            "status": "success",
            "query": query,
            "results": results
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

@mcp.tool()
def initialize_rag_tool(openai_api_key: str = None) -> dict:
    """
    Initialize RAG tool
    
    Args:
        openai_api_key: OpenAI API key (optional)
        
    Returns:
        dict: Initialization result
    """
    try:
        global _rag_tool
        _rag_tool = BlenderInfinigenRAG(openai_api_key)
        return {"status": "success", "message": "RAG tool initialized successfully"}
    except Exception as e:
        return {"status": "error", "error": str(e)}

def main():
    """Run MCP server or test the RAG tool when --test is provided"""
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        # Test instruction (Chinese as requested)
        test_instruction = "Move the bed to the corner of the room"
        # Ensure tool is initialized
        global _rag_tool
        if _rag_tool is None:
            _rag_tool = BlenderInfinigenRAG()
        # Run query with official doc search enabled
        result = _rag_tool.rag_query(test_instruction, use_enhanced=False, use_doc_search=True)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
        # Default: run as MCP server
    else:
        mcp.run(transport="stdio")

if __name__ == "__main__":
    main()
