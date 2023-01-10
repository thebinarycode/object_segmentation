import os
import sys
from tensorflow.keras.callbacks import ModelCheckpoint
from data_generator import DataGenerator
from tensorboard_callbacks import TrainValTensorBoard, TensorBoardMask
from utils import generate_missing_json
from config import model_name, n_classes
from models import unet, fcn_8


def sorted_fns(dir):
    return sorted(os.listdir(dir), key=lambda x: int(x.split('.')[0]))

if len(os.listdir('dataset/shapes/train/images')) != len(os.listdir('dataset/shapes/train/annotated')):
    generate_missing_json()
if len(os.listdir('dataset/shapes/validation/images')) != len(os.listdir('dataset/shapes/validation/annotated')):
    generate_missing_json()

train_image_paths = [os.path.join('dataset/shapes/train/images', x) for x in sorted_fns('dataset/shapes/train/images')]
train_annot_paths = [os.path.join('dataset/shapes/train/annotated', x) for x in sorted_fns('dataset/shapes/train/annotated')]

val_image_paths = [os.path.join('dataset/shapes/validation/images', x) for x in sorted_fns('dataset/shapes/validation/images')]
val_annot_paths = [os.path.join('dataset/shapes/validation/annotated', x) for x in sorted_fns('dataset/shapes/validation/annotated')]


def train(model_name, epochs, save_dir):
    if 'unet' in model_name:
        model = unet(pretrained=False, base=4)
    elif 'fcn_8' in model_name:
        model = fcn_8(pretrained=False, base=4)
    tg = DataGenerator(image_paths=train_image_paths, annot_paths=train_annot_paths,
                    batch_size=5, augment=True)
    vg = DataGenerator(image_paths=val_image_paths, annot_paths=val_annot_paths,
                        batch_size=5, augment=True)

    checkpoint = ModelCheckpoint(os.path.join(save_dir, model_name+'.model'), monitor='dice', verbose=1, mode='max',
                                save_best_only=True, save_weights_only=False, period=10)

    train_val = TrainValTensorBoard(write_graph=True)
    tb_mask = TensorBoardMask(log_freq=10)

    model.fit_generator(generator=tg, validation_data=vg,
                        steps_per_epoch=len(tg),
                        epochs=epochs, verbose=1,
                        callbacks=[checkpoint, train_val, tb_mask])


if __name__ == "__main__":
    # efficientdet_lite0 4 10 saved_model
    print("Inside train model script")
    print(f"args received: {sys.argv[1:]}")
    model_name = sys.argv[1]
    epochs = int(sys.argv[2])
    save_dir = sys.argv[3]

    #train("unet", 500, "saved_model")
    train(model_name, epochs, save_dir)