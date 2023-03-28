import logging
import Configuration.confTG as ct
from telegram import Update
from telegram.ext import filters, MessageHandler, ApplicationBuilder, CommandHandler, ContextTypes, CallbackContext


class TelegramBot:
    def __init__(self):
        logging.basicConfig(
            format=ct.LOG_FORMAT,
            level=logging.INFO
        )
        self.__application = ApplicationBuilder().token(ct.TOKEN).build()
        self.__application.add_handler(CommandHandler(ct.HELP_C, self.__user_guide))
        self.__application.add_handler(MessageHandler(filters.PHOTO, self.__image))
        self.__application.add_handler(MessageHandler(filters.TEXT, self.__text))

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

    async def __image(self, update: Update, context: CallbackContext):
        if update.effective_message.photo is None:
            await context.bot.send_message(chat_id=update.effective_chat.id, text=ct.UNSUPPORTED_M)
        else:
            # getting the smallest variant of photo
            photo = update.effective_message.photo[0]
            file = await context.bot.get_file(photo)
            await file.download_to_drive(f"{ct.DOWNLOAD_DIR}{photo.file_id}.png")
            await context.bot.send_message(chat_id=update.effective_chat.id, text=ct.PROCESSING_M)

    @staticmethod
    async def __text(update: Update, context: CallbackContext):
        await context.bot.send_message(chat_id=update.effective_chat.id, text=ct.UNSUPPORTED_M)

    @staticmethod
    async def __user_guide(update: Update, context: ContextTypes.DEFAULT_TYPE):
        await context.bot.send_message(chat_id=update.effective_chat.id, text=ct.HELP_M)
