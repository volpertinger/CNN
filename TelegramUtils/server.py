import Configuration.confTG as ct
import TelegramUtils.handlers as h
import logging
from telegram.ext import filters, MessageHandler, ApplicationBuilder, CommandHandler, ContextTypes, CallbackContext


class TelegramBot:
    def __init__(self):
        logging.basicConfig(
            format=ct.LOG_FORMAT,
            level=logging.INFO
        )
        self.__application = ApplicationBuilder().token(ct.TOKEN).build()
        self.__application.add_handler(CommandHandler(ct.HELP_C, h.user_guide))
        self.__application.add_handler(MessageHandler(filters.PHOTO, h.image))
        self.__application.add_handler(MessageHandler(filters.TEXT, h.text))

    def start(self):
        self.__application.run_polling()

    def stop(self):
        self.__application.stop()
