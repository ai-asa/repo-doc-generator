# %%
import configparser
import google.generativeai as genai
from pathlib import Path
import os
from dotenv import load_dotenv
from utils.logging_config import setup_logger

logger = setup_logger(__name__)

class GeminiAdapter:

    def __init__(self):
        # .envファイルを読み込む
        load_dotenv()
        
        settings_path = Path(__file__).parent.parent.parent / 'settings.ini'
        config = configparser.ConfigParser()
        config.read(settings_path, encoding='utf-8')
        gemini_api_key = os.getenv('GEMINI_API_KEY')
        gemini_selected_model = config.get('SYSTEM', 'gen_ai_model',fallback='gemini-1.5-flash')
        self.call_attempt_limit = int(config.get('SYSTEM', 'call_limite', fallback=5))
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel(gemini_selected_model)


    def gemini_chat(self, user_text):
        for i in range(self.call_attempt_limit):
            try:
                response = self.model.generate_content(user_text)
                return response.text
            except Exception as error:
                logger.error(f"gemini呼び出し時にエラーが発生しました: {error}")
                if i == self.call_attempt_limit - 1:
                    return None
                logger.warning(f"リトライ {i+1}/{self.call_attempt_limit}")
                continue
    
    def gemini_streaming(self, user_text):
        for i in range(self.call_attempt_limit):
            try:
                response = self.model.generate_content(user_text, stream=True)
                for chunk in response:
                    if hasattr(chunk, 'parts'):
                        texts = [part.text for part in chunk.parts if hasattr(part, 'text')]
                        yield ''.join(texts)
                break
            except Exception as error:
                logger.error(f"gemini呼び出し時にエラーが発生しました: {error}")
                if i == self.call_attempt_limit - 1:
                    return None
                logger.warning(f"リトライ {i+1}/{self.call_attempt_limit}")
                continue
