# SlidesLib

SlidesLib is a Python library for slide generation, providing APIs for image generation, Google search, and slide customization.

## Features
- **Image Generation**: Create images using the DALL-E API.
- **Search Integration**: Perform Google searches, save screenshots, and retrieve images.
- **Slide Customization**: Add text, bullet points, images, and set slide backgrounds.

## Installation
1. **Dependencies**: Install required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```
2. **Google Chrome**: Required for search functionality:
   ```bash
   wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
   sudo dpkg -i google-chrome-stable_current_amd64.deb
   sudo apt-get install -f
   ```
3. **OpenAI API Key**: Export your API key:
   ```bash
   export OPENAI_API_KEY="your_api_key"
   ```

## Quick Start
- **Image Generation**:
   ```python
   from slidesLib.image_gen import Dalle3
   Dalle3.generate_image("A futuristic cityscape", save_path="cityscape.png")
   ```

- **Search Integration**:
   ```python
   from slidesLib.search import GoogleSearch
   GoogleSearch.search_result("Tallest building in the world", "result.png")
   ```

- **Slide Customization**:
   ```python
   from slidesLib.ppt_gen import add_title
   add_title(slide, text="Welcome to SlidesLib")
   ```

For more examples, refer to the code in this folder.
```