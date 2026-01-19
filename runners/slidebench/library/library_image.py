"""Image Helper Functions for AutoPresent Slide Generation.

Provides Google search and DALL-E image generation functions for slides.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "utils", "third_party", "slides"))

from SlidesLib import GoogleSearch, Dalle3

def google_search_screenshot(question: str, save_path: str = "screenshot.png") -> str:
    """
    Search a question on Google, and take a screenshot of the search result.
    Save the screenshot to save_path, and return the path.
    Args:
        question: str, The question to search on Google.
        save_path: str, The path to save the screenshot.
    Returns:
        The path of the saved screenshot.
    """
    return GoogleSearch.search_result(question, save_path)


def search_image(query: str, save_path: str = "image.png") -> str:
    """
    Search for an image on Google and download the result to save_path.
    Args:
        query: str, The query to search for.
        save_path: str, The path to save the downloaded image.
    Rets:
        the save_path.
    """
    return GoogleSearch.search_image(query, save_path)


def generate_image(query: str, save_path: str = "image.png") -> str:
    """
    Generate an image using diffusion model based on a text query, and save the image to the path.
    Args:
        query: str, The text query to generate the image.
        save_path: str, The path to save the generated image.
    Rets:
        The path of the saved image
    """
    return Dalle3.generate_image(query, save_path)