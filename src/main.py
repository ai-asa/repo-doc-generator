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