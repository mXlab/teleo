import malaya_speech
from malaya_speech import Pipeline

webrtc = malaya_speech.vad.webrtc()

gender_model = malaya_speech.gender.deep_model(model = 'vggvox-v2')
language_detection_model = malaya_speech.language_detection.deep_model(model = 'vggvox-v2')
age_model = malaya_speech.age_detection.deep_model(model = 'vggvox-v2')
emotion_model = malaya_speech.emotion.deep_model(model = 'vggvox-v2')

p_classification = Pipeline()
to_float = p_classification.map(malaya_speech.astype.to_ndarray).map(malaya_speech.astype.int_to_float)
gender = to_float.map(gender_model)
#language_detection = to_float.map(language_detection_model)
emotion_detection = to_float.map(emotion_model)
age_detection = to_float.map(age_model)

combined = gender.zip(emotion_detection).zip(age_detection).flatten()
combined.map(lambda x: x, name = 'classification')

#p_classification.visualize()

file, samples = malaya_speech.streaming.record(webrtc, classification_model = p_classification)
#samples = malaya_speech.streaming.pyaudio.stream(webrtc, classification_model = p_classification)
