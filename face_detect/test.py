import cognitive_face as CF

if __name__ == '__main__':
    CF.Key.set('dfb1d3b95f5e4041adeb4265b4601f55')

    # img_url if file zbs
    img_url = 'https://raw.githubusercontent.com/Microsoft/Cognitive-Face-Windows/master/Data/detection1.jpg'
    result = CF.face.detect(img_url, landmarks=True, attributes='headPose')
    print result