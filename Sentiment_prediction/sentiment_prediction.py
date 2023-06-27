import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch.nn.functional as F

model_path = '/content/drive/MyDrive/Sentiment Analysis/simplet5-epoch-2-train-loss-0.0447-val-loss-0.0571'
model = T5ForConditionalGeneration.from_pretrained(model_path)
tokenizer = T5Tokenizer.from_pretrained(model_path)

def analyze_sentiment(sentence):
    input_text = "sentiment: " + sentence
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    input_embeds = model.get_input_embeddings()(input_ids)

    # T5 모델을 사용하여 감정 분석 수행
    output = model.generate(input_ids=input_ids, decoder_inputs_embeds=input_embeds, max_length=200)
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)

    # 확률을 얻기 위해 로짓 계산
    logits = model(input_ids=input_ids, decoder_inputs_embeds=input_embeds).logits
    probabilities = F.softmax(logits, dim=-1).squeeze()

    sentiment = decoded_output
    emotions = ["Positive", "Negative"]
    emotion_probabilities = probabilities[0].tolist()  # 첫 번째 원소를 추출하여 확률값으로 사용
    result = {"sentiment": sentiment, "emotions": dict(zip(emotions, emotion_probabilities))}
    return result

sentences = ['I took a sip and immediately discarded it. How could a beer have such a strong cinnamon flavor?',
             "You can't perceive any hop aroma in this beer, and it feels like drinking water.",
             'The taste of this beer embodies the style description quite well.',
             "This beer successfully captures the bitter aroma of coffee, and when you're about to finish a glass, there's a slight dominant sweetness that lingers in your mouth.",
             "The taste of this Barrel-Aged Imperial Stout is not overpowering, and the subtle sweetness reminiscent of banana can be detected",
             "This beer goes down very smoothly, like water. It captures the essence of a Helles beer very well.",
             "Although the hop aroma is not strong, the rich taste of malt is well pronounced.",
             "This beer goes down like water."]
for sentence in sentences:
    predicted_sentiment = analyze_sentiment(sentence)
    print("문장:", sentence)
    print("감정:", predicted_sentiment["sentiment"])