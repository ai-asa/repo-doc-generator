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