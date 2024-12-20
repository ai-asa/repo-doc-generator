# リポジトリ分析

## 概要
# Pythonプロジェクト概要

## プロジェクトの主な目的
このプロジェクトは、指定されたディレクトリ内のコードや設定ファイルを自動的に分析し、その内容を包括的なMarkdownドキュメントとして出力するリポジトリ分析ツールを提供することを目的としています。これにより、開発者はプロジェクトの全体像を把握しやすくし、ドキュメント作成の手間を省くことができます。

## 主要なコンポーネントとその役割
- **`main.py`**: コマンドライン引数を解析し、分析対象のディレクトリを指定するインターフェイスを提供します。このスクリプトが起動点となり、`RepositoryAnalyzer`を呼び出してリポジトリの分析結果を出力します。
- **`repository_analyzer.py`**: リポジトリを分析する主要なクラスが含まれています。ファイルの無視パターンを定義し、ディレクトリ構造やファイルの読み込みを通じて、プロンプト生成やOpenAIを使って内容を解析します。
- **`openai_adapter.py`**: OpenAIのAPIと通信するクラスで、模倣されたAIモデルを使ってファイルの内容からプロジェクトの概要を生成します。
- **`gemini_adapter.py`**: GoogleのGenerative AI（Gemini）サービスと通信するクラスで、別のAIモデルを用いた分析機能を提供します。
- **`get_prompt.py`**: プロンプト生成を行うクラスで、リポジトリの概要をAIに生成させるための指示を定義します。
- **`logging_config.py`**: ログの設定を行うユーティリティで、分析中のエラーや情報を記録します。

## 使用している主要な外部APIやライブラリ
- **OpenAI**: ファイル内容からプロジェクトの概要を生成するために使用しています。具体的にはChat APIを活用しています。
- **Google Generative AI (Gemini)**: 別のAIモデルを活用するためにGemini APIを使用しています。
- **dotenv**: 環境変数を.loadし、APIキーなどの機密情報を安全に管理するために使用しています。

## アーキテクチャの特徴
- **モジュラー構造**: 各機能が独立したモジュールとして実装されており、拡張性や保守性に優れています。
- **プラグイン可能なAIモデル**: `openai_adapter.py`と`gemini_adapter.py`により、異なるAIサービスを簡単に切り替えて利用することが可能です。
- **エラーハンドリングとログ**: `logging_config.py`を用いた詳細なログ設定により、異常発生時のトラブルシューティングを容易にしています。
- **パターンマッチングとフィルタリング**: 無視するファイルパターンやサイズ制限を定義し、効率的に分析するためのファイル選定を行っています。

## ディレクトリ構造
├── LICENSE
├── requirements.txt
├── settings.ini
└── src/
    ├── chat/
    │   ├── gemini_adapter.py
    │   └── openai_adapter.py
    ├── config/
    ├── main.py
    ├── output/
    ├── prompt/
    │   └── get_prompt.py
    ├── repository_analyzer.py
    └── utils/
        └── logging_config.py

## ソースコード
### settings.ini
```ini
[SYSTEM]
call_limite = 5
gen_ai_model = gpt-4o
```

### src\main.py
```py
from repository_analyzer import RepositoryAnalyzer
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='リポジトリ分析ツール')
    parser.add_argument('--dir', '-d', type=str, default='.',
                       help='分析対象のディレクトリパス（デフォルト: カレントディレクトリ）')
    parser.add_argument('--output', '-o', type=str, default='output',
                       help='分析結果の出力ディレクトリパス（デフォルト: output）')
    args = parser.parse_args()

    # パスを正規化
    target_dir = Path(args.dir).resolve()
    output_dir = Path(args.output).resolve()
    
    # 分析を実行して結果をファイルに保存
    repo_analyzer = RepositoryAnalyzer(target_dir)
    repo_analyzer.save_analysis(output_dir)

if __name__ == '__main__':
    main()
```

### src\repository_analyzer.py
```py
import re
from chat.openai_adapter import OpenaiAdapter
from prompt.get_prompt import GetPrompt
from pathlib import Path
from utils.logging_config import setup_logger

logger = setup_logger(__name__)

class RepositoryAnalyzer:
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.openai = OpenaiAdapter()
        self.gp = GetPrompt()
        # 共通の無視パターンをクラス変数として定義
        self.ignore_patterns = [
            r'__pycache__',
            r'\.git',
            r'\.pytest_cache',
            r'\.vscode',
            r'\.idea',
            r'\.env',
            r'venv',
            r'build',
            r'dist',
            r'__init__\.py$',
            r'node_modules',
            r'\.next',
            r'\.cache',
            r'coverage',
            r'\.husky',
            r'\.gradle',
            r'target',
            r'bin',
            r'obj',
            r'\.vs',
            r'vendor',
            r'\.DS_Store',
            r'\.env\.local',
            r'\.env\.development\.local',
            r'\.env\.test\.local',
            r'\.env\.production\.local'
        ]
        
        # 大きすぎるファイルを無視するサイズ制限（例：1MB）
        self.max_file_size = 1024 * 1024  # 1MB in bytes

    def analyze_repository(self) -> str:
        """リポジトリを分析してmarkdownドキュメントを生成"""
        # 1. 概要の生成
        overview = self._generate_overview()
        
        # 2. ディレクトリ構造の生成
        directory_structure = self._generate_directory_structure()
        
        # 3. ソースコードの取得
        source_files = self._get_source_files()
        
        # markdownドキュメントの組み立て
        markdown = f"""# リポジトリ分析

## 概要
{overview}

## ディレクトリ構造
{directory_structure}

## ソースコード
{source_files} 
"""
        return markdown

    def _generate_overview(self) -> str:
        """リポジトリの概要を生成"""
        files_content = self._get_all_relevant_files()
        prompt = self.gp.overview_prompt(files_content)
        return self.openai.openai_chat(prompt)

    def _generate_directory_structure(self) -> str:
        """ディレクトリ構造を生成"""
        def create_tree(start_path: Path, prefix: str = '') -> str:
            if self.should_ignore(start_path):  # 共通の判定メソッドを使用
                return ''
                
            content = []
            try:
                items = sorted(start_path.iterdir())
                
                for i, item in enumerate(items):
                    is_last = i == len(items) - 1
                    
                    if self.should_ignore(item):
                        continue
                        
                    if item.is_file():
                        content.append(f"{prefix}{'└── ' if is_last else '├── '}{item.name}")
                    elif item.is_dir():
                        content.append(f"{prefix}{'└── ' if is_last else '├── '}{item.name}/")
                        content.append(create_tree(item, prefix + ('    ' if is_last else '│   ')))
                        
                return '\n'.join(filter(None, content))
            except PermissionError:
                return f"{prefix}[アクセス拒否]"
                
        return create_tree(self.base_dir)

    def _get_source_files(self) -> str:
        """ソースコードファイルの内容を取得してmarkdown形式で返す"""
        source_files = []
        
        # 分析対象の拡張子を定義（設定ファイルを含む）
        relevant_extensions = {
            # ソースコード
            '.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.cpp', '.h', '.cs',
            # ドキュメント
            '.md', '.rst',
            # 設定ファイル
            '.json', '.yaml', '.yml', '.toml', '.ini', '.conf',
            # その他の重要な設定ファイル
            '.gitignore', '.dockerignore', '.env.example'
        }
        
        # 特定のファイル名を直接指定
        important_filenames = {
            'package.json', 'tsconfig.json', 'Dockerfile', 'Makefile', 'README.md'
        }
        
        for path in self.base_dir.rglob('*'):
            if self.should_ignore(path):
                continue
                
            if path.is_file():
                # 拡張子またはファイル名で判定
                if path.suffix in relevant_extensions or path.name in important_filenames:
                    try:
                        content = path.read_text(encoding='utf-8')
                        rel_path = path.relative_to(self.base_dir)
                        
                        # 拡張子がない場合はファイル名をそのまま使用
                        extension = path.suffix[1:] if path.suffix else path.name
                        source_files.append(f"### {rel_path}\n```{extension}\n{content}\n```\n")
                    except Exception as e:
                        logger.error(f"ファイル読み込みエラー {path}: {e}")
                        
        return '\n'.join(source_files)

    def should_ignore(self, path: Path) -> bool:
        """パスが無視すべきかどうかを判定"""
        path_str = str(path)
        
        # パターンマッチによる判定
        if any(re.search(pattern, path_str) for pattern in self.ignore_patterns):
            return True
            
        # ファイルサイズによる判定
        if path.is_file():
            try:
                if path.stat().st_size > self.max_file_size:
                    logger.warning(f"大きすぎるファイルをスキップします: {path} ({path.stat().st_size / 1024 / 1024:.2f}MB)")
                    return True
            except OSError:
                logger.error(f"ファイルアクセスエラー: {path}")
                return True
                
        return False

    def _get_all_relevant_files(self) -> str:
        """分析に必要な全ファイルの内容を取得"""
        files_content = []
        
        # 分析対象の拡張子を定義
        relevant_extensions = {'.py', '.md', '.js', '.jsx', '.ts', '.tsx', '.java', '.cpp', '.h', '.cs'}
        
        for path in self.base_dir.rglob('*'):
            if self.should_ignore(path):
                continue
                
            if path.is_file() and path.suffix in relevant_extensions:
                try:
                    content = path.read_text(encoding='utf-8')
                    rel_path = path.relative_to(self.base_dir)
                    files_content.append(f"=== {rel_path} ===\n{content}\n")
                except Exception as e:
                    print(f"Error reading {path}: {e}")
                    
        return '\n'.join(files_content)

    def save_analysis(self, output_dir: str = 'output'):
        """分析結果をファイルに保存"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        analysis_file = output_path / 'repository_analysis.md'
        markdown = self.analyze_repository()
        analysis_file.write_text(markdown, encoding='utf-8')
```

### src\chat\gemini_adapter.py
```py
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
        self.gemini_model = config.get('CONFIG', 'gemini_model',fallback='gemini-1.5-flash')
        gemini_selected_model = config.get('SYSTEM', 'gemini_selected_model',fallback='gemini-1.5-flash')
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

```

### src\chat\openai_adapter.py
```py
# %%
import configparser
from openai import OpenAI
from pathlib import Path
import os
from dotenv import load_dotenv
from utils.logging_config import setup_logger

logger = setup_logger(__name__)

class OpenaiAdapter:

    def __init__(self):
        # .envファイルを読み込む
        load_dotenv()
        
        settings_path = Path(__file__).parent.parent.parent / 'settings.ini'
        config = configparser.ConfigParser()
        config.read(settings_path, encoding='utf-8')
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
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
                logger.error(f"GPT呼び出し時にエラーが発生しました: {error}")
                if i == self.call_attempt_limit - 1:
                    return None
                logger.warning(f"リトライ {i+1}/{self.call_attempt_limit}")
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
                logger.error(f"GPT呼び出し時にエラーが発生しました: {error}")
                if i == self.call_attempt_limit - 1:
                    return None
                logger.warning(f"リトライ {i+1}/{self.call_attempt_limit}")
                continue

```

### src\prompt\get_prompt.py
```py

class GetPrompt:
    def __init__(self):
        pass

    def overview_prompt(self, files_content: str) -> str:
        """リポジトリの概要を生成するためのプロンプトを返す
        
        Args:
            files_content (str): 分析対象のファイル群の内容
            
        Returns:
            str: 生成用プロンプト
        """
        return f"""以下のファイル群からなるPythonプロジェクトの概要を生成してください。
特に以下の点に注目してください：
- プロジェクトの主な目的
- 主要なコンポーネントとその役割
- 使用している主要な外部APIやライブラリ
- アーキテクチャの特徴

ファイル内容:
{files_content}

レスポンスは日本語のmarkdown形式で記述してください。
"""


```

### src\utils\logging_config.py
```py
import logging
import sys

def setup_logger(name: str) -> logging.Logger:
    """
    ロガーの設定を行う
    
    Args:
        name (str): ロガー名（通常は__name__を使用）
        
    Returns:
        logging.Logger: 設定済みのロガーインスタンス
    """
    logger = logging.getLogger(name)
    
    # ロガーが既に設定されている場合は、既存のロガーを返す
    if logger.handlers:
        return logger
        
    logger.setLevel(logging.DEBUG)
    
    # コンソールハンドラの設定
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    
    # フォーマッターの設定
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    
    return logger 
```
 
