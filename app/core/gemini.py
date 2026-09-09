from google import genai
import os


class GeminiClient:

    def __init__(self):
        self.client = genai.Client(
            api_key=os.environ["GEMINI_API_KEY"]
        )

        self.model = "gemini-2.5-flash"