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