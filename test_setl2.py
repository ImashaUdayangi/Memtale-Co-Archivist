from src.setl2 import setl2_emotion_vector

text = """
I left my hometown with mixed feelings. I was scared and sad to leave my family,
but also excited about starting a new chapter in my life.
"""

emotion_vec = setl2_emotion_vector(text)
print(emotion_vec)