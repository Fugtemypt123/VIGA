from openai import OpenAI
import requests

class Dalle3():
    @classmethod
    def __init_dalle__(cls):
        client = OpenAI()
        return client
        
    @classmethod
    def generate_image(cls, query: str, save_path: str = "downloaded_image.png"):
        """Generate an image based on a text query, save the image to the save_path"""
        client = cls.__init_dalle__()
        response = client.images.generate(
            model="dall-e-3",
            prompt=query,
            size="1024x1024",
            quality="standard",
            n=1,
        )
        image_url = response.data[0].url
        # Send a GET request to the URL
        response = requests.get(image_url)

        # Open a file in binary write mode and write the content of the response
        with open(save_path, "wb") as file:
            file.write(response.content)
        return save_path