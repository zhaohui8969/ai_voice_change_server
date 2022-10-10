import logging
import os
import time
from datetime import datetime

from flask import Flask, request
from flask import send_from_directory
from flask_cors import CORS

from config import config
from logger_config import configure_logger

from model import ModelWrapper

app = Flask(__name__)
CORS(app)

def get_timestamp():
    timestamp = datetime.now()
    return timestamp.strftime("%Y%m%d_%H%M%S")


@app.route("/voiceChangeModel", methods=["POST"])
def voiceChangeModel():
    """
    :return:
    """
    starttime = time.time()
    try:
        request_form = request.form
        request_files = request.files
        logger.info(request_form)
        logger.info(request_files)
        wave_file = request_files.get("sample", None)
        f_pitch_change = int(float(request_form.get("fPitchChange", 0)))
        save_file_name = os.path.join("raw", "{}.wav".format(get_timestamp()))
        print("save_file_name:{}".format(save_file_name))
        with open(save_file_name, 'wb') as fop:
            fop.write(wave_file.stream.read())
        result_file = model.predict(save_file_name, f_pitch_change)
        endtime = time.time()
        runtime = round((endtime - starttime), 2)
        logger.info("runtime:{}".format(runtime))
        return send_from_directory("./results", os.path.basename(result_file), as_attachment=True)
    except Exception as e:
        if request.content_length < 1024:
            request_data_log = f"request body: {request.get_data(as_text=True)}"
        else:
            request_data_log = "request body 太大，不予输出"
        logger.error(f"/np [POST]接口错误。{request_data_log}", exc_info=True)


def build_gunicorn(config_file):
    """
    gunicorn的启动入口
    也可以当做CLI的启动入口，可以从这里读取配置文件，初始化模型
    :param config_file: 配置文件，将CLI的入参按行拆分写在txt文件里
    :return:
    """
    global args
    global model
    global logger
    with open(config_file, 'r', encoding='utf-8') as fop:
        lines = [i.strip() for i in fop.readlines()]
        args = config(cli_lines=lines)
        logging.info(args)
    # gunicorn
    # 必须从build_gunicorn方法获取APP实例，用于读取配置文件
    gunicorn_logger = logging.getLogger('gunicorn.info')
    logger = configure_logger("info", "logs/prediction.log")
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)
    logger.info(args)

    logger.info("load model")
    # model = FewShotModuleWrapper(gpt_api_url=args.gpt_srv_url,
    #                              terms_api_url=args.terms_srv_url,
    #                              connection_timeout=3,
    #                              read_timeout=30)

    model = ModelWrapper(model_dir=args.model_dir)
    return app


if __name__ == '__main__':
    build_gunicorn("config_for_gunicorn_start.ini")
    app.run(port=args.port, host="0.0.0.0", debug=False, threaded=False)
