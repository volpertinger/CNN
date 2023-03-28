import CNNUtils.convolutionalNeuralNetwork as cnn
import Configuration.confCNN as cc
import CNNUtils.imageProcessing as ip
import TelegramUtils.server as ss

if __name__ == '__main__':
    img_proc = ip.ImageProc(cc.TRANSFORM_MATRIX, cc.SHIFT_SCALE, cc.DATA_SCALE, cc.INPUT_SHAPE,
                            cc.MIN_COLOR_SENSITIVITY, cc.MAX_COLOR_SENSITIVITY)

    model = cnn.Model(cc.DATASET, img_proc, cc.CLASSES_NUMBER, cc.INPUT_SHAPE, cc.BATCH_SIZE, cc.EPOCHS,
                      cc.SAVE_FILENAME, cc.TRAIN_HISTORY_FILENAME, cc.COMMON_ACTIVATION, cc.LAST_ACTIVATION,
                      cc.KERNEL_INITIALIZER, cc.LEARNING_RATE, cc.MOMENTUM, cc.LOSS, cc.METRICS, cc.SMALL_DOT,
                      cc.LARGE_DOT, cc.SMALL_DENSE, cc.LARGE_DENSE, cc.SMALL_FILTER, cc.LARGE_FILTER, cc.CMAP,
                      cc.POSSIBILITY_PRECISION, cc.WITH_INFO, cc.VERBOSE)
    model.train()
    telegram_bot = ss.TelegramBot(model)
    telegram_bot.start()
