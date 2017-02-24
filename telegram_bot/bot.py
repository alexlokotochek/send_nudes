#!python3
import os
import requests
import time
import telebot
import uuid

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
    send_image('/home/www/flask_project/static/images/8bf122b7-6fc8-4e11-ac78-07d95d13c4d1.jpg', message.chat.id)


@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    file_info = bot.get_file(message.photo[-1].file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    src = '/home/www/flask_project/static/images/{}.jpg'.format(uuid.uuid4())
    print(src)
    with open(src, 'wb') as new_file:
        new_file.write(downloaded_file)
    send_image('/home/www/flask_project/static/images/f33edae9-3c95-455e-990d-e65d1199ce7e.jpg', message.chat.id)

@bot.message_handler(content_types=['document'])
def handle_doc(message):
    bot.send_message(message.chat.id, 'Please send as a photo not as a doc')


if __name__ == '__main__':
    bot.polling(none_stop=True)
