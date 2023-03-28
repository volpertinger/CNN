import logging
import Configuration.confTG as ct
import CNNUtils.convolutionalNeuralNetwork as cnn
import os
from telegram import Update
from telegram.ext import filters, MessageHandler, ApplicationBuilder, CommandHandler, ContextTypes, CallbackContext


class TelegramBot:
    def __init__(self, model: cnn.Model):
        logging.basicConfig(
            format=ct.LOG_FORMAT,
            level=logging.INFO
        )
        self.__application = ApplicationBuilder().token(ct.TOKEN).build()
        self.__application.add_handler(CommandHandler(ct.HELP_C, self.__user_guide))
        self.__application.add_handler(MessageHandler(filters.PHOTO, self.__image))
        self.__application.add_handler(MessageHandler(filters.TEXT, self.__text))

        self.__model = model

    # ------------------------------------------------------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------------------------------------------------------
    def start(self):
        self.__application.run_polling()

    def stop(self):
        self.__application.stop()

    # ------------------------------------------------------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------------------------------------------------------

    def __clear_predict(self, filepath):
        result = self.__model.get_predict_string(filepath)
        os.remove(filepath)
        return result

    async def __image(self, update: Update, context: CallbackContext):
        if update.effective_message.photo is None:
            await context.bot.send_message(chat_id=update.effective_chat.id, text=ct.UNSUPPORTED_M)
        else:
            # getting the high quality variant of photo
            photo = update.effective_message.photo[-1]
            file = await context.bot.get_file(photo)
            path = f"{ct.DOWNLOAD_DIR}{photo.file_id}.png"
            await file.download_to_drive(path)
            await context.bot.send_message(chat_id=update.effective_chat.id, text=ct.PROCESSING_M)
            prediction = self.__clear_predict(path)
            await context.bot.send_message(chat_id=update.effective_chat.id, text=prediction)

    @staticmethod
    async def __text(update: Update, context: CallbackContext):
        await context.bot.send_message(chat_id=update.effective_chat.id, text=ct.UNSUPPORTED_M)

    @staticmethod
    async def __user_guide(update: Update, context: ContextTypes.DEFAULT_TYPE):
        await context.bot.send_message(chat_id=update.effective_chat.id, text=ct.HELP_M)
