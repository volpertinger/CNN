import Configuration.confTG as ct
from telegram import Update
from telegram.ext import ContextTypes, CallbackContext


async def image(update: Update, context: CallbackContext):
    if update.effective_message.photo is None:
        await context.bot.send_message(chat_id=update.effective_chat.id, text=ct.UNSUPPORTED_M)
    else:
        # getting the smallest variant of photo
        photo = update.effective_message.photo[0]
        file = await context.bot.get_file(photo)
        await file.download_to_drive(f"{ct.DOWNLOAD_DIR}{photo.file_id}.png")
        await context.bot.send_message(chat_id=update.effective_chat.id, text=ct.PROCESSING_M)


async def text(update: Update, context: CallbackContext):
    await context.bot.send_message(chat_id=update.effective_chat.id, text=ct.UNSUPPORTED_M)


async def user_guide(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id, text=ct.HELP_M)
