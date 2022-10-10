import logging
import os
from logging.config import dictConfig

import tqdm


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.DEBUG):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)

def configure_logger(name, log_path):
    """配置logger

    提供两种logger(name): debug可记录debug日志，info可记录info日志。注意本程序会根据传入的
    `log_path`生成两个日志文件，一个是`ERR`开头的日志，记录错误日志；另一个是`INFO`或者`DEBUG`
    开头，内容自明。

    Args:
        name (:str): logger handler名称，`info`或者`debug`之一。
        log_path (:str): 日志位置，绝对路径
    """
    try:
        log_dir, log_file = os.path.split(log_path)
        if log_dir:
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
    except Exception as e:
        raise ValueError("log_path error, MUST be absolute path")
    dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s - %(module)s - %(levelname)s - %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                }
            },
            "handlers": {
                "console_handler": {
                    "level": logging.DEBUG,
                    "class": "logging.StreamHandler",
                    "formatter": "default",
                    "stream": "ext://sys.stdout",
                },
                "info_handler": {
                    "level": logging.INFO,
                    "class": "logging.handlers.TimedRotatingFileHandler",
                    "formatter": "default",
                    "when": "D",
                    "interval": 1,
                    "filename": os.path.join(log_dir, log_file),
                    "backupCount": 7,
                    "encoding": "utf8",
                },
                "debug_handler": {
                    "level": logging.DEBUG,
                    "class": "logging.handlers.TimedRotatingFileHandler",
                    "formatter": "default",
                    "when": "D",
                    "interval": 1,
                    "filename": os.path.join(log_dir, log_file),
                    "backupCount": 7,
                    "encoding": "utf8",
                },
                "error_handler": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": logging.ERROR,
                    "formatter": "default",
                    "filename": os.path.join(log_dir, "error_" + log_file),
                    "maxBytes": 10485760,
                    "backupCount": 7,
                    "encoding": "utf8",
                }
            },
            "loggers": {
                "": {  # root logger
                    "level": logging.DEBUG,
                    "handlers": ["console_handler","debug_handler", "error_handler"],
                },
                "debug": {
                    "level": logging.DEBUG,
                    "handlers": ["debug_handler", "error_handler"],
                    "propagate": False,
                },
                "info": {
                    "level": logging.INFO,
                    "handlers": ["info_handler", "error_handler"],
                    "propagate": False,
                },
                "file": {
                    "level": logging.INFO,
                    "handlers": ["info_handler", "error_handler"],
                    "propagate": False,
                }
            },
        }
    )

    logging.getLogger("info").addHandler(TqdmLoggingHandler())
    logging.getLogger("debug").addHandler(TqdmLoggingHandler())
    logger = logging.getLogger(name)

    return logger


if __name__ == "__main__":
    l1 = configure_logger("info", "./a3.log")
    # logging.config.dictConfig(LOGGING_DICT)  # 导入上面定义的logging配置
    # l1=logging.getLogger('info')
    l1.info("测试")
