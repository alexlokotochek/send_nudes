import cognitive_face as CF
import os


def detect_face(img):
    return CF.face.detect(img, landmarks=True, attributes='headPose')


def create_set(path='data'):
    print CF.face_list.create('my-list-1')
    for f in os.listdir(path):
        print f, CF.face_list.add_face(os.path.join(path, f), 'my-list-1')


def find_similars(img_id):
    return CF.face.find_similars(img_id, face_list_id='my-list-1', mode='matchFace')

if __name__ == '__main__':
    CF.Key.set('dfb1d3b95f5e4041adeb4265b4601f55')
    # print detect_face('/home/user/iron_hack/send_nudes/face_detect/test/Simasia_Helen-H_0003.jpg')
    # print find_similars('e43e9381-c010-4818-aeca-3e9af5f326a7')
    print find_similars('4ccb2184-1882-46be-8bd4-3c7833681845')
    # create_set()
    # img_url if file zbs
    # img_url = 'https://raw.githubusercontent.com/Microsoft/Cognitive-Face-Windows/master/Data/detection1.jpg'
    #
    # print detect_face(img_url)

