
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

