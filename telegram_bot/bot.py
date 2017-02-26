#!python3
import os
import requests
import time
import telebot
import uuid
import cv2
from telebot import types
import face_to_vec.api as api
import face_to_vec.porn_creator as porn_creator

TOKEN = '246763002:AAHfGctwhdHPExiyRz39FhVHTLj5kqM2QkQ'
bot = telebot.TeleBot(TOKEN)

def send_image(path, chat_id):
    url = "https://api.telegram.org/bot{}/sendPhoto".format(TOKEN);
    files = {'photo': open(path, 'rb')}
    data = {'chat_id' : chat_id}
    r = requests.post(url, files=files, data=data)
    print(r.status_code, r.reason, r.content)


@bot.message_handler(content_types=['text'])
def handle_text(message):
    pass


@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    file_info = bot.get_file(message.photo[-1].file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    src = '/home/www/flask_project/static/images/{}.jpg'.format(uuid.uuid4())
    print(src)
    with open(src, 'wb') as new_file:
        new_file.write(downloaded_file)
    image = cv2.imread(src, cv2.IMREAD_COLOR)
    res = porn_creator.process_image(image)
    new_pk = uuid.uuid4()
    new_uri = '/home/www/flask_project/static/images/{}.jpg'.format(new_pk)
    cv2.imwrite(new_uri, res)
    send_image(new_uri, message.chat.id)
    markup = types.ReplyKeyboardMarkup()
    markup.row('yes')
    markup.row('no')
    bot.send_message(message.chat.id, "Did you like it?", reply_markup=markup)


@bot.message_handler(content_types=['document'])
def handle_doc(message):
    bot.send_message(message.chat.id, 'Please send as a photo not as a doc')


if __name__ == '__main__':
    api.init()
    bot.polling(none_stop=True)
