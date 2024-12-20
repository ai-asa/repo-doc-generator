# %%
import configparser
from openai import OpenAI
from pathlib import Path

class OpenaiAdapter:

    def __init__(self):
        settings_path = Path(__file__).parent.parent.parent / 'settings.ini'
        config = configparser.ConfigParser()
        config.read(settings_path, encoding='utf-8')
        self.openai_api_key = config.get('ENVIRONMENT', 'openai_api_key', fallback="")
        self.call_attempt_limit = int(config.get('SYSTEM', 'call_limite', fallback=5))
        self.openai_selected_model = config.get('SYSTEM', 'openai_model', fallback="gpt-4o")
        self.client = OpenAI(
            api_key = self.openai_api_key
        )
    
    def openai_chat(self, prompt, temperature=1):
        system_prompt = [{"role": "system", "content": prompt}]
        for i in range(self.call_attempt_limit):
            try:
                response = self.client.chat.completions.create(
                    messages=system_prompt,
                    model=self.openai_selected_model,
                    temperature=temperature
                )
                text = response.choices[0].message.content
                return text
            except Exception as error:
                print(f"GPT呼び出し時にエラーが発生しました:{error}")
                if i == self.call_attempt_limit - 1:
                    return None  # エラー時はNoneを返す
                continue
    
    def openai_streaming(self, prompt, temperature=1):
        system_prompt = [{"role": "system", "content": prompt}]
        for i in range(self.call_attempt_limit):
            try:
                stream = self.client.chat.completions.create(
                    model=self.openai_selected_model,
                    messages=system_prompt,
                    temperature=temperature,
                    stream=True
                )
                for chunk in stream:
                    if chunk.choices[0].delta.content is not None:
                        yield chunk.choices[0].delta.content
                break
            except Exception as error:
                print(f"GPT呼び出し時にエラーが発生しました:{error}")
                if i == self.call_attempt_limit - 1:
                    return None  # エラー時はNoneを返す
                continue