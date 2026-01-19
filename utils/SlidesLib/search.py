"""Google and Bing image search utilities."""
import os
import shutil
import time
from typing import Tuple

import requests
from bing_image_downloader import downloader
from chromedriver_py import binary_path
from google_images_download import google_images_download
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


class GoogleSearch:
    """Google search utilities for images and web results."""

    @classmethod
    def _init_driver(cls) -> Tuple[webdriver.Chrome, WebDriverWait]:
        """Initialize Chrome WebDriver with headless options."""
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        service = Service(binary_path)
        driver = webdriver.Chrome(service=service, options=chrome_options)
        wait = WebDriverWait(driver, 100)
        return driver, wait

    @classmethod
    def search_result(cls, question: str, screenshot_path: str = "screenshot.png") -> str:
        """Search a question on Google and return a screenshot of the search result."""
        driver, wait = cls._init_driver()

        if not question:
            raise ValueError("Please provide a question")

        # Perform Google search
        search_url = f"https://www.google.com/search?q={question}"
        driver.get(search_url)

        # Give some time for the page to load
        time.sleep(3)

        # Take a screenshot
        driver.save_screenshot(screenshot_path)

        driver.quit()
        return screenshot_path

    @classmethod
    def search_image_org(cls, query: str, download_path: str = 'top_image.png') -> str:
        """Search for an image on Google and download the top result."""
        driver, wait = cls._init_driver()

        if not query:
            raise ValueError("Please provide a query")

        # Perform Google image search
        search_url = f"https://www.google.com/search?tbm=isch&q={query}"
        driver.get(search_url)

        # Find all image elements
        image_elements = driver.find_elements(By.CSS_SELECTOR, "img")

        # Filter out Google icon images and get the first valid image URL
        image_url = None
        for img in image_elements:
            src = img.get_attribute("src")
            if src and "googlelogo" not in src:
                image_url = src
                try:
                    response = requests.get(image_url)
                    with open(download_path, 'wb') as file:
                        file.write(response.content)

                    driver.quit()
                    print(image_url)
                    return download_path
                except (requests.RequestException, IOError):
                    print("Error downloading image, skipping.")
                    continue

        driver.quit()
        raise RuntimeError("No valid image found")

    @classmethod
    def search_image_prev(cls, query: str, output_dir: str = './downloads', limit: int = 10) -> str:
        """Search for images using Bing Image Downloader (deprecated)."""
        # Download images using Bing Image Downloader
        downloader.download(query, limit=limit, output_dir=output_dir, adult_filter_off=True, force_replace=False, timeout=60)
        # List the files in the output directory
        image_dir = os.path.join(output_dir, query)
        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"No images found for query '{query}' in directory '{output_dir}'")

        # Collect all image paths
        image_paths = [os.path.join(image_dir, file) for file in os.listdir(image_dir) if file.endswith(('jpg', 'jpeg', 'png'))]

        # Return the first image
        return image_paths[0]

    @classmethod
    def search_image(cls, query: str, save_path: str) -> str:
        """Search for an image based on the query and save the result to the specified path.

        Args:
            query: The query to search for.
            save_path: The path to save the downloaded image.

        Returns:
            The path where the image was saved.
        """
        # Create a temporary directory for storing downloaded images
        temp_dir = "./temp_download"
        os.makedirs(temp_dir, exist_ok=True)

        # Download only the top image result
        downloader.download(query, limit=1, output_dir=temp_dir, adult_filter_off=True, force_replace=True, timeout=60)

        # Construct the expected directory and image path
        image_dir = os.path.join(temp_dir, query)
        image_files = [file for file in os.listdir(image_dir) if file.endswith(('jpg', 'jpeg', 'png'))]

        # Check if any image files were downloaded
        if not image_files:
            raise FileNotFoundError(f"No images found for query '{query}'.")

        # Copy the top image to the desired save path
        top_image_path = os.path.join(image_dir, image_files[0])
        shutil.move(top_image_path, save_path)

        # Clean up temporary directory
        shutil.rmtree(temp_dir)

        return save_path
