from src.utils.all_utils import read_yaml, create_directory
from src.utils.models import get_VGG_16_model, prepare_model
import argparse
import os
import logging
import io

logging_str = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, 'running_logs.log'), level=logging.INFO, format=logging_str,
                    filemode="a")

def prepare_base_model(config_path, params_path):
    config = read_yaml(config_path)
    params = read_yaml(params_path)
    # Loading the artifacts dir
    artifacts = config["artifacts"]
    artifacts_dir = artifacts["ARTIFACTS_DIR"]
    # Loading the base model directory and base model name
    base_model_dir = artifacts["BASE_MODEL_DIR"]
    base_model_name = artifacts["BASE_MODEL_NAME"]
    # Creating the base model directory and base model dir path 
    base_model_dir_path = os.path.join(artifacts_dir, base_model_dir)

    create_directory([base_model_dir_path])
    # Base model path
    base_model_path = os.path.join(base_model_dir_path, base_model_name)
    # Loading tne base model VGG 16 for the transfer learning
    model = get_VGG_16_model(input_shape=params["IMAGE_SIZE"], model_path=base_model_path)
    # Preparing the final model taht will take i/p base model, num of class is 2, freeze
    #every thing in base model,freeze_till as None, Learning rate
    full_model = prepare_model(
        model,
        CLASSES=params["CLASSES"],
        freeze_all=True,
        freeze_till=None,
        learning_rate=params["LEARNING_RATE"]
    )
   # Newly updated custom model to updated/added in new directory
    update_base_model_path = os.path.join(
        base_model_dir_path,
        artifacts["UPDATED_BASE_MODEL_NAME"]
    )
    # Logging the updated base model summary
    def _log_model_summary(full_model):
        with io.StringIO() as stream:
            full_model.summary(print_fn=lambda x: stream.write(f"{x}\n"))
            summary_str = stream.getvalue()
        return summary_str

    logging.info(f"full model summary: \n{_log_model_summary(full_model)}")
    # Saving the model to the dir location
    full_model.save(update_base_model_path)
    


if __name__ == '__main__':
    args = argparse.ArgumentParser()

    args.add_argument("--config", "-c", default="config/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")

    parsed_args = args.parse_args()

    try:
        logging.info(">>>>> stage two started")
        prepare_base_model(config_path=parsed_args.config, params_path=parsed_args.params)
        logging.info("stage two completed! base model is created >>>>>\n")
    except Exception as e:
        logging.exception(e)
        raise 