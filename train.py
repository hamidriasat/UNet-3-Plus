import hydra
import data_generator
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # Train the YoloV4_Model using sequence generator
    train_generator = data_generator.DataGenerator(cfg, shuffle=True)

    # # in case of Sequence generator use for loop
    for temp_batch_img, temp_batch_mask in train_generator:
        print(len(temp_batch_img))

if __name__ == "__main__":
    main()
