from src.setl2 import setl2_emotion_vector
from src.summarisation import naive_summary, emotion_aware_summary
from src.metrics import js_divergence

text = """
I left my hometown with mixed feelings. I was scared and sad to leave my family,
but also excited about starting a new chapter in my life. I still remember the last
day at home and how my mother hugged me tightly. Even now, I feel both pain and pride.
"""

full_vec = setl2_emotion_vector(text)

naive = naive_summary(text)
naive_vec = setl2_emotion_vector(naive)

emo = emotion_aware_summary(text)
emo_vec = setl2_emotion_vector(emo)

print("\n=== NAIVE SUMMARY ===\n", naive)
print("\n=== EMOTION-AWARE SUMMARY ===\n", emo)

print("\nFull emotion:", full_vec)
print("Naive emotion:", naive_vec)
print("Aware emotion:", emo_vec)

print("\nJS Divergence (Full vs Naive):", js_divergence(full_vec, naive_vec))
print("JS Divergence (Full vs Aware):", js_divergence(full_vec, emo_vec))