# ChatGPT o1 用リポジトリ分析ツール

## 概要
このツールは、GitHubリポジトリのソースコードを自動的に分析し、プロジェクトの構造や概要をMarkdown形式のドキュメントとして生成します。OpenAIのAIモデルを活用して、コードの理解と文書化を支援します。o1にリポジトリの情報を渡すプロンプトとして活用します。

## 主な機能
- プロジェクトディレクトリの自動スキャンと分析
- ソースコードファイルの内容解析
- AIを活用したプロジェクト概要の自動生成
- Markdown形式でのドキュメント出力

## 必要要件
- Python 3.7以上
- 必要なPythonパッケージ:
  ```
  openai>=1.0.0
  python-dotenv
  pathlib
  configparser
  logging
  ```

## インストール方法
1. リポジトリをクローン:
```bash
git clone [リポジトリURL]
cd [プロジェクトディレクトリ]
```

2. 必要なパッケージをインストール:
```bash
pip install -r requirements.txt
```

3. 環境変数の設定:
- `.env.example`を`.env`にコピーし、必要なAPI keyを設定:
  ```
  OPENAI_API_KEY=your_openai_api_key_here
  ```

## 使用方法
1. 基本的な使用方法:
```bash
python src/main.py --dir [分析対象のディレクトリパス] --output [出力先ディレクトリパス]
```

2. オプション:
- `--dir`, `-d`: 分析対象のディレクトリパス（デフォルト: カレントディレクトリ）
- `--output`, `-o`: 分析結果の出力先ディレクトリパス（デフォルト: output）

## 設定
`settings.ini`で以下の設定が可能です:
```ini
[SYSTEM]
call_limite = 5  # APIコール試行回数の制限
gen_ai_model = gpt-4o  # 使用するAIモデル
```

## プロジェクト構造
```
├── LICENSE
├── README.md
├── requirements.txt
├── settings.ini
└── src/
    ├── chat/
    │   └── openai_adapter.py
    ├── main.py
    ├── output/
    ├── prompt/
    │   └── get_prompt.py
    ├── repository_analyzer.py
    └── utils/
        └── logging_config.py
```

## ライセンス
このプロジェクトはApache License 2.0のもとで公開されています。詳細は[LICENSE](LICENSE)ファイルを参照してください。

## 注意事項
- OpenAIのAPIキーが必要です
- 大きなファイル（1MB以上）は自動的にスキップされます
- 特定のディレクトリ（.git, node_modules等）は解析から除外されます 