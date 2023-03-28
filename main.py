import CNNUtils.convolutionalNeuralNetwork as cnn
import Configuration.confCNN as cc

if __name__ == '__main__':
    model = cnn.Model(cc.DATASET, cc.CLASSES_NUMBER, cc.INPUT_SHAPE, cc.COLOR_SCALE, cc.BATCH_SIZE, cc.EPOCHS,
                      cc.SAVE_FILENAME, cc.TRAIN_HISTORY_FILENAME, cc.COMMON_ACTIVATION, cc.LAST_ACTIVATION,
                      cc.KERNEL_INITIALIZER, cc.LEARNING_RATE, cc.MOMENTUM, cc.LOSS, cc.METRICS, cc.SMALL_DOT,
                      cc.LARGE_DOT, cc.SMALL_DENSE, cc.LARGE_DENSE, cc.SMALL_FILTER, cc.LARGE_FILTER, cc.CMAP,
                      cc.POSSIBILITY_PRECISION, cc.WITH_INFO, cc.VERBOSE)
    print(model.get_log())
    model.train()
    # model.predict_test(0)
    print(model.predict_image("media/Figure_1.png"))
