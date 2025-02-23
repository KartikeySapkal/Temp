import os
import logging


class ChatConfig:
    UPLOAD_FOLDER = 'uploads'
    OLLAMA_HOST = 'https://lamprey-useful-slug.ngrok-free.app/'
    OLLAMA_HEADERS = {'x-some-header': 'some-value'}

    @classmethod
    def setup(cls):
        # Create upload directory
        os.makedirs(cls.UPLOAD_FOLDER, exist_ok=True)

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('chatbot.log'),
                logging.StreamHandler()
            ]
        )
